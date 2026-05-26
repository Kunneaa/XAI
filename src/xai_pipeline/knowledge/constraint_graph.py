"""Principle-family equation selection boundary."""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from dataclasses import dataclass
from typing import Iterable, List

from .registries import FORMULA_IDS, FORMULA_REGISTRY, PRINCIPLE_IDS, TASK_TYPES, FormulaSpec, formula_execution_branch
from .units import unit_info
from .language import has_change_factor_cue, has_frequency_transform_cue


COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
}


@dataclass(frozen=True)
class PrincipleSelection:
    ok: bool
    formula_ids: List[str]
    issues: List[str]
    trace: dict

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "formula_ids": list(self.formula_ids),
            "issues": list(self.issues),
            "trace": dict(self.trace),
        }


@dataclass(frozen=True)
class RouteResult:
    task_type: str
    answer_type: str
    confidence: float
    reasons: List[str]

    def to_dict(self):
        return {
            "task_type": self.task_type,
            "answer_type": self.answer_type,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ConstraintGraph:
    variables: List[dict]
    formulas: List[dict]
    edges: List[dict]
    target_dimensions: List[str]
    reachable_formula_ids: List[str]
    derived_variables: List[dict]
    selected_formula_ids: List[str]
    solvable_target_dimensions: List[str]
    issues: List[str]

    def to_dict(self) -> dict:
        return {
            "variables": [dict(item) for item in self.variables],
            "formulas": [dict(item) for item in self.formulas],
            "edges": [dict(item) for item in self.edges],
            "target_dimensions": list(self.target_dimensions),
            "reachable_formula_ids": list(self.reachable_formula_ids),
            "derived_variables": [dict(item) for item in self.derived_variables],
            "selected_formula_ids": list(self.selected_formula_ids),
            "solvable_target_dimensions": list(self.solvable_target_dimensions),
            "issues": list(self.issues),
        }


def route(front_payload: dict) -> RouteResult:
    """Build the first constraint-graph route from semantic IR.

    This is deterministic-first graph navigation. It scores code-owned formula
    cards by reachable dimensions and domain context; it never asks a model to
    choose the executable equation.
    """

    text = str(front_payload.get("canonical_question") or "").lower()
    answer_type = front_payload.get("answer_type_hint") or "unknown"
    concepts = set(front_payload.get("concepts") or [])
    target_dimensions = infer_target_dimensions(front_payload)
    available_dimensions = _front_available_dimensions(front_payload)
    available_counts = Counter(available_dimensions)
    branch_context = _branch_context(front_payload)

    if answer_type == "multi_output":
        return RouteResult("multi_output", answer_type, 0.72, ["multi-output target detected"])

    if _symbolic_or_graph_theory_query(text, answer_type, target_dimensions):
        return RouteResult("conceptual", answer_type, 0.68, ["formula/expression/graph-shape theory query detected"])

    if _measurement_error_query(text):
        return RouteResult("measurement_error", _numeric_answer_type(answer_type), 0.72, ["measurement uncertainty/error query detected"])

    if _capacitor_breakdown_charge_query(text, target_dimensions):
        return RouteResult("capacitor_charge", _numeric_answer_type(answer_type), 0.82, ["parallel-plate breakdown charge query detected"])

    if _electric_field_zero_unknown_charge_query(front_payload, text, target_dimensions, available_counts):
        routed_answer_type = "numeric" if available_counts["charge"] >= 1 else answer_type
        return RouteResult("electric_field_point", routed_answer_type, 0.76, ["unknown charge from zero-field constraint detected"])

    if _inverse_square_midpoint_field_expression_query(front_payload, text):
        return RouteResult("electric_field_point", "symbolic", 0.76, ["inverse-square field-line midpoint expression detected"])

    if _symmetry_zero_query(front_payload, text, target_dimensions):
        task_type = "coulomb_force" if "force" in target_dimensions else "electric_field_point"
        return RouteResult(task_type, answer_type, 0.72, ["symmetry-zero vector query detected"])

    if _midpoint_equal_source_zero_query(front_payload, text, target_dimensions):
        task_type = "coulomb_force" if "force" in target_dimensions else "electric_field_point"
        return RouteResult(task_type, answer_type, 0.7, ["midpoint equal-source cancellation detected"])

    if _resultant_force_query(text, available_dimensions, target_dimensions):
        return RouteResult("resultant_force", _numeric_answer_type(answer_type), 0.82, ["resultant of explicit force vectors detected"])

    if _directional_vector_query(text, target_dimensions):
        task_type = "electric_field_point" if "electric_field" in target_dimensions else "coulomb_force"
        return RouteResult(task_type, answer_type, 0.74, ["direction-only vector superposition query detected"])

    if _relationship_rule_query(text):
        return RouteResult("conceptual", answer_type, 0.68, ["symbolic relationship rule detected"])

    rlc_quadrature_route = _rlc_quadrature_split_route(text, target_dimensions, answer_type, available_counts)
    if rlc_quadrature_route is not None:
        return rlc_quadrature_route

    additive_route = _branch_additive_route(text, target_dimensions, answer_type, available_counts)
    if additive_route is not None:
        return additive_route

    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    rlc_frequency_route = _rlc_frequency_transform_route(text, target_text, target_dimensions, answer_type)
    if rlc_frequency_route is not None:
        return rlc_frequency_route

    if answer_type in {"conceptual", "yes_no"} or (_conceptual_signal(text, concepts) and not target_dimensions):
        return RouteResult("conceptual", answer_type, 0.68, ["conceptual or yes/no signal detected"])

    topology_route = _route_canonical_topology(front_payload, target_dimensions, answer_type, text)
    if topology_route is not None:
        return topology_route

    if "rlc" in text and any(cue in text for cue in [" cos", " sin", "cos(", "sin("]):
        target_text = " ".join(front_payload.get("target_hints", [])).lower()
        if "angular_frequency" in target_dimensions or "angular frequency" in target_text or "omega" in target_text or "ω" in target_text:
            return RouteResult("lc_frequency", _numeric_answer_type(answer_type), 0.82, ["sinusoidal source angular-frequency query detected"])
        if "reactance" in target_text or re.search(r"\b(?:x_l|xl|x_c|xc)\b", target_text):
            task = "capacitive_reactance" if re.search(r"\b(?:x_c|xc|capacitive)\b", target_text) else "inductive_reactance"
            return RouteResult(task, _numeric_answer_type(answer_type), 0.82, ["sinusoidal source reactance query detected"])
        if "impedance" in target_text or re.search(r"\bz\b", target_text):
            return RouteResult("rlc_impedance", _numeric_answer_type(answer_type), 0.82, ["sinusoidal source impedance query detected"])
        if "voltage" in target_dimensions:
            return RouteResult("ohm_law", _numeric_answer_type(answer_type), 0.82, ["sinusoidal source voltage query detected"])

    if _rlc_current_ratio_reactance_query(text):
        return RouteResult("inductive_reactance", _numeric_answer_type(answer_type), 0.82, ["RLC resonance current-ratio reactance query detected"])

    if "solenoid" in text and ("energy" in target_dimensions or "energy density" in text):
        return RouteResult("inductor_energy", _numeric_answer_type(answer_type), 0.82, ["target asks for solenoid magnetic energy"])
    if "solenoid" in text and ("flux linkage" in text or "magnetic_flux" in target_dimensions):
        return RouteResult("magnetic_flux", _numeric_answer_type(answer_type), 0.82, ["target asks for solenoid flux/linkage"])
    if "solenoid" in text and ("magnetic_field" in target_dimensions or "flux density" in text):
        return RouteResult("solenoid_magnetic_field", _numeric_answer_type(answer_type), 0.82, ["target asks for solenoid magnetic field"])

    if (
        "rlc" in text
        or all(cue in text for cue in ["resistor", "capacitor", "inductor"])
        or all(cue in text for cue in ["resistance", "capacitance", "inductance"])
    ) and "resonan" in text and "voltage" in target_dimensions:
        return RouteResult("ohm_law", _numeric_answer_type(answer_type), 0.82, ["RLC resonance voltage query detected"])

    if "capacitance" not in target_dimensions and "capacitor" in text and re.search(r"\bfind\s+c['′]?\b|\bcalculate\s+c['′]?\b", text):
        return RouteResult("capacitance", _numeric_answer_type(answer_type), 0.78, ["target asks for an unknown capacitor capacitance symbol"])
    if ("voltage" in target_dimensions or re.search(r"\bwhat\s+voltage\b|\bvoltage\s+must\b", text)) and "capacitor" in text:
        return RouteResult("capacitor_final_voltage", _numeric_answer_type(answer_type), 0.82, ["target asks for capacitor voltage"])
    if "capacitance" in target_dimensions and "capacitor" in text:
        return RouteResult("capacitance", _numeric_answer_type(answer_type), 0.82, ["target asks for capacitor capacitance"])
    if "mass" in target_dimensions and "equilibrium" in text and available_counts["charge"] >= 1 and available_counts["electric_field"] >= 1:
        return RouteResult("force_in_electric_field", _numeric_answer_type(answer_type), 0.8, ["charged-particle equilibrium in a uniform electric field"])
    if "energy" in target_dimensions and "capacitor" in text and any(cue in text for cue in ["total oscillation energy", "oscillation energy", "connected to an inductor"]):
        return RouteResult("capacitor_energy", _numeric_answer_type(answer_type), 0.82, ["charged capacitor initial energy sets total LC energy"])
    if "energy" in target_dimensions and ("inductor" in text or "magnetic field energy" in text):
        return RouteResult("inductor_energy", _numeric_answer_type(answer_type), 0.82, ["target asks for inductor magnetic-field energy"])
    if "energy" in target_dimensions and ("capacitor" in text or "electric field energy" in text):
        return RouteResult("capacitor_energy", _numeric_answer_type(answer_type), 0.82, ["target asks for capacitor electric-field energy"])
    if "electric_field" in target_dimensions and available_counts["force"] >= 1 and available_counts["charge"] >= 1:
        return RouteResult("electric_field_force", _numeric_answer_type(answer_type), 0.84, ["target asks for electric field from force on test charge"])
    if "length" in target_dimensions and "electron" in text and "electric field" in text and any(cue in text for cue in ["reduces to zero", "comes to rest", "stops", "before its velocity"]):
        return RouteResult("charged_particle_motion", _numeric_answer_type(answer_type), 0.82, ["charged particle stopping distance in uniform electric field"])
    if "electric_field" in target_dimensions and available_counts["electric_field"] >= 1 and "dielectric" in text:
        return RouteResult("electric_field_point", _numeric_answer_type(answer_type), 0.78, ["target asks for dielectric-scaled electric field"])
    candidates = _rank_formula_candidates(
        available_dimensions=available_dimensions,
        target_dimensions=target_dimensions,
        text=text,
        branch_context=branch_context,
        front_payload=front_payload,
    )
    if candidates:
        score, spec, reason = candidates[0]
        task_type = spec.task_type if spec.task_type in TASK_TYPES else "unknown"
        confidence = max(0.45, min(0.9, 0.45 + score / 100.0))
        return RouteResult(task_type, _numeric_answer_type(answer_type), confidence, [reason])

    if concepts:
        return RouteResult("conceptual", answer_type, 0.55, ["front-end concept signal without numeric route"])

    return RouteResult("unknown", answer_type, 0.25, ["no registry-connected route"])


def _route_canonical_topology(
    front_payload: dict,
    target_dimensions: list[str],
    answer_type: str,
    text: str,
) -> RouteResult | None:
    topology = front_payload.get("topology_graph") or {}
    canonical = topology.get("canonical_form")
    if canonical not in {"series_topology", "parallel_topology"} or topology.get("ambiguity"):
        return None
    counts = Counter(
        quantity.get("dimension")
        for quantity in front_payload.get("quantities", [])
        if quantity.get("dimension")
    )
    relation = "series" if canonical == "series_topology" else "parallel"
    numeric_answer_type = _numeric_answer_type(answer_type)
    if "capacitance" in target_dimensions and counts["capacitance"] >= 2:
        return RouteResult(
            "capacitance",
            numeric_answer_type,
            0.82,
            [f"canonical {relation} capacitance topology"],
        )
    if "resistance" in target_dimensions and counts["resistance"] >= 2 and _target_requests_equivalent_resistance(text):
        return RouteResult(
            "equivalent_resistance",
            numeric_answer_type,
            0.82,
            [f"canonical {relation} resistance topology"],
        )
    if "current" in target_dimensions and counts["resistance"] >= 2 and counts["voltage"] == 1:
        return RouteResult(
            "ohm_law",
            numeric_answer_type,
            0.8,
            [f"canonical {relation} resistor topology plus Ohm law"],
        )
    if "voltage" in target_dimensions and counts["resistance"] >= 2 and counts["current"] == 1:
        return RouteResult(
            "ohm_law",
            numeric_answer_type,
            0.8,
            [f"canonical {relation} resistor topology plus Ohm law"],
        )
    return None


def build_constraint_graph(front_payload: dict, route_result=None) -> ConstraintGraph:
    """Build a bounded bipartite variable-formula reachability graph.

    Variable nodes are grounded dimensions observed in the semantic IR plus
    dimensions produced by reachable constraints. Formula nodes are executable
    registry formulas. The graph is expanded by deterministic greedy BFS over
    formula preconditions; it records the local subgraph that can reach the
    requested target dimensions without asking an LLM to pick formulas.
    """

    target_dimensions = infer_target_dimensions(front_payload)
    quantities = [*front_payload.get("quantities", []), *front_payload.get("symbolic_quantities", [])]
    variables: List[dict] = []
    known_counts: Counter[str] = Counter()
    for index, quantity in enumerate(quantities):
        dimension = quantity.get("dimension")
        if not dimension or dimension == "constant":
            continue
        node_id = f"var:{dimension}:{index}"
        variables.append(
            {
                "id": node_id,
                "kind": "variable",
                "dimension": dimension,
                "known": quantity in front_payload.get("quantities", []),
                "symbol": quantity.get("symbol"),
                "source": quantity.get("raw_text"),
            }
        )
        known_counts[dimension] += 1
    for index, fact in enumerate(_derived_dimension_facts(front_payload, known_counts), start=1):
        dimension = fact["dimension"]
        known_counts[dimension] += 1
        variables.append(
            {
                "id": f"var:{dimension}:derived_front:{index}",
                "kind": "variable",
                "dimension": dimension,
                "known": False,
                "source": fact["source"],
                "derived_by": fact["derived_by"],
            }
        )

    formulas: List[dict] = []
    edges: List[dict] = []
    formula_specs: list[FormulaSpec] = []
    text = str(front_payload.get("canonical_question") or "").lower()
    for formula_id, spec in FORMULA_REGISTRY.items():
        if formula_id not in FORMULA_IDS:
            continue
        if route_result is not None and route_result.task_type not in {"unknown", spec.task_type}:
            continue
        if not _formula_context_allowed(spec, text, front_payload):
            continue
        formula_node = {
            "id": f"formula:{formula_id}",
            "kind": "formula",
            "formula_id": formula_id,
            "task_type": spec.task_type,
            "principle_id": spec.principle_id,
            "required_dimensions": list(spec.required_dimensions),
            "target_dimension": spec.target_dimension,
        }
        formulas.append(formula_node)
        formula_specs.append(spec)
        for dimension in spec.required_dimensions:
            edges.append({"source": f"dimension:{dimension}", "target": formula_node["id"], "kind": "requires"})
        edges.append({"source": formula_node["id"], "target": f"dimension:{spec.target_dimension}", "kind": "produces"})

    reachable, derived_variables, selected_formula_ids, solvable_targets, expansion_trace = _expand_reachable_subgraph(
        formula_specs,
        known_counts,
        target_dimensions,
        branch_context=_branch_context(front_payload),
    )
    for variable in derived_variables:
        variables.append(variable)
    edges.extend(expansion_trace["derived_edges"])
    issues = []
    if target_dimensions and not solvable_targets:
        issues.append("target_not_reachable_from_known_dimensions")
    return ConstraintGraph(
        variables=variables,
        formulas=formulas,
        edges=edges,
        target_dimensions=target_dimensions,
        reachable_formula_ids=reachable,
        derived_variables=derived_variables,
        selected_formula_ids=selected_formula_ids,
        solvable_target_dimensions=solvable_targets,
        issues=issues,
    )


def _expand_reachable_subgraph(
    formula_specs: list[FormulaSpec],
    known_counts: Counter[str],
    target_dimensions: list[str],
    max_depth: int = 4,
    branch_context: dict | None = None,
) -> tuple[list[str], list[dict], list[str], list[str], dict]:
    available = Counter(known_counts)
    reachable: list[str] = []
    selected: list[str] = []
    derived_variables: list[dict] = []
    derived_edges: list[dict] = []
    seen_formulas: set[str] = set()
    produced_once: set[str] = set(known_counts)
    target_set = set(target_dimensions)

    for depth in range(max_depth):
        candidates: list[tuple[float, str, FormulaSpec]] = []
        for spec in formula_specs:
            if spec.formula_id in seen_formulas or _formula_is_metadata_only(spec):
                continue
            if _missing_required_dimensions(list(available.elements()), spec.required_dimensions):
                continue
            candidates.append((_constraint_cost(spec, target_set, depth, branch_context or {}), spec.formula_id, spec))
        if not candidates:
            break
        progressed = False
        for _, _, spec in sorted(candidates, key=lambda item: (item[0], item[1])):
            seen_formulas.add(spec.formula_id)
            reachable.append(spec.formula_id)
            if spec.target_dimension not in produced_once:
                produced_once.add(spec.target_dimension)
                available[spec.target_dimension] += 1
                node_id = f"var:{spec.target_dimension}:derived:{spec.formula_id}"
                derived_variables.append(
                    {
                        "id": node_id,
                        "kind": "variable",
                        "dimension": spec.target_dimension,
                        "known": False,
                        "produced_by": spec.formula_id,
                        "depth": depth + 1,
                    }
                )
                derived_edges.append({"source": f"formula:{spec.formula_id}", "target": node_id, "kind": "derives"})
                progressed = True
            if spec.target_dimension in target_set and spec.formula_id not in selected:
                selected.append(spec.formula_id)
        if target_set and target_set <= set(produced_once):
            break
        if not progressed:
            break

    if not selected and not target_set:
        selected = reachable[:]
    elif not selected and target_set:
        selected = [
            formula_id
            for formula_id in reachable
            if FORMULA_REGISTRY[formula_id].target_dimension in target_set
        ]
    selected = _prune_branch_conflicting_formulas(selected, branch_context or {})
    solvable_targets = sorted(target_set & set(produced_once), key=lambda dimension: target_dimensions.index(dimension))
    return reachable, derived_variables, selected, solvable_targets, {"derived_edges": derived_edges}


def _prune_branch_conflicting_formulas(selected_formula_ids: list[str], branch_context: dict) -> list[str]:
    """Keep the selected subgraph aligned with the structural branch evidence."""

    if not selected_formula_ids:
        return []
    selected_specs = [FORMULA_REGISTRY[formula_id] for formula_id in selected_formula_ids if formula_id in FORMULA_REGISTRY]
    preferred_targets: set[tuple[str, str]] = set()
    preferred_branches: set[str] = set()
    if branch_context.get("spatial_preferred"):
        preferred_branches.add("spatial_vector")
    if branch_context.get("topology_preferred"):
        preferred_branches.add("topology")
    for spec in selected_specs:
        if formula_execution_branch(spec.formula_id) in preferred_branches:
            preferred_targets.add((spec.task_type, spec.target_dimension))
    if not preferred_targets:
        return selected_formula_ids
    pruned: list[str] = []
    for formula_id in selected_formula_ids:
        spec = FORMULA_REGISTRY.get(formula_id)
        if spec is None:
            continue
        branch = formula_execution_branch(formula_id)
        if (spec.task_type, spec.target_dimension) in preferred_targets and branch not in preferred_branches:
            continue
        pruned.append(formula_id)
    return pruned


def _constraint_cost(spec: FormulaSpec, target_dimensions: set[str], depth: int, branch_context: dict | None = None) -> float:
    expression = spec.expression
    nonlinear_cost = expression.count("**") * 2 + expression.count("sqrt") * 3 + expression.count("/") * 0.5
    target_bonus = -20.0 if spec.target_dimension in target_dimensions else 0.0
    branch_bonus = _branch_cost_adjustment(spec, branch_context or {})
    return depth * 10.0 + len(spec.required_dimensions) + nonlinear_cost + target_bonus + branch_bonus


def infer_target_dimensions(front_payload: dict) -> list[str]:
    ordered: list[str] = []
    explicit_goal_dimensions: set[str] = set()

    def add(dimension: str | None) -> None:
        if dimension and dimension not in ordered:
            ordered.append(dimension)

    for goal in front_payload.get("goals", []) or []:
        dimension = goal.get("dimension")
        add(dimension)
        if dimension:
            explicit_goal_dimensions.add(dimension)
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    full_text = str(front_payload.get("canonical_question") or "").lower()
    haystack = f"{target_text} {full_text if not target_text else ''}"
    for keywords, dimension in ROUTER_TARGET_DIMENSION_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            if dimension not in explicit_goal_dimensions and _dimension_is_target_object_context(target_text, dimension):
                continue
            add(dimension)
    if not ordered:
        for quantity in front_payload.get("symbolic_quantities", []):
            add(quantity.get("dimension"))
    return ordered


def _dimension_is_target_object_context(target_text: str, dimension: str) -> bool:
    """Avoid adding dimensions for objects that the requested quantity acts on."""

    if not target_text:
        return False
    if dimension == "charge":
        return bool(
            re.search(
                r"\b(?:on|at|of|for|from|toward|towards)\s+(?:the\s+)?"
                r"(?:test\s+|probe\s+|target\s+|point\s+)?charge\b",
                target_text,
            )
            or re.search(r"\bcharge\s+(?:at|located\s+at|placed\s+at)\b", target_text)
        )
    return False


def _rank_formula_candidates(
    *,
    available_dimensions: list[str],
    target_dimensions: list[str],
    text: str,
    branch_context: dict | None = None,
    front_payload: dict | None = None,
) -> list[tuple[float, FormulaSpec, str]]:
    ranked: list[tuple[float, FormulaSpec, str]] = []
    for spec in FORMULA_REGISTRY.values():
        if spec.formula_id not in FORMULA_IDS:
            continue
        if _formula_is_metadata_only(spec):
            continue
        if not _formula_context_allowed(spec, text, front_payload):
            continue
        missing = _missing_required_dimensions(available_dimensions, spec.required_dimensions)
        if missing:
            continue
        score = 10.0
        if spec.target_dimension in target_dimensions:
            score += 45.0 - min(target_dimensions.index(spec.target_dimension), 4) * 5.0
        elif target_dimensions:
            score -= 60.0
        score += _domain_context_score(spec, text)
        score += _branch_score_adjustment(spec, branch_context or {})
        score -= max(0, len(spec.required_dimensions) - 2) * 1.5
        reason = (
            f"registry formula {spec.formula_id} connects "
            f"{list(spec.required_dimensions)} -> {spec.target_dimension}"
        )
        ranked.append((score, spec, reason))
    ranked.sort(key=lambda item: (-item[0], item[1].formula_id))
    return ranked


def _front_available_dimensions(front_payload: dict) -> list[str]:
    counts = Counter(
        quantity.get("dimension")
        for quantity in [*front_payload.get("quantities", []), *front_payload.get("symbolic_quantities", [])]
        if quantity.get("dimension") and quantity.get("dimension") != "constant"
    )
    for fact in _derived_dimension_facts(front_payload, counts):
        counts[fact["dimension"]] += 1
    return list(counts.elements())


def _relation_qualifiers(front_payload: dict) -> set[str]:
    return {
        str(relation.get("qualifier") or "").lower()
        for relation in front_payload.get("relations", []) or []
        if isinstance(relation, dict) and relation.get("qualifier")
    }


def _geometry_structures(front_payload: dict) -> dict:
    return ((front_payload.get("canonical_structures") or {}).get("geometry") or {})


def _geometry_cues(front_payload: dict) -> set[str]:
    """Return shape/location cues grounded by the semantic frontend.

    The knowledge layer uses these cues before raw wording so route selection
    follows normalized IR instead of one dataset sentence shape.
    """

    cues = set(_relation_qualifiers(front_payload))
    geometry = _geometry_structures(front_payload)
    triangles = geometry.get("triangles") or []
    if triangles:
        cues.add("triangle")
    if any(triangle.get("canonical_right_angle_at") or triangle.get("right_angle_at") for triangle in triangles):
        cues.add("right_triangle")
    if geometry.get("squares"):
        cues.add("square")
    return cues


def _has_geometry_cue(front_payload: dict, *names: str) -> bool:
    cues = _geometry_cues(front_payload)
    return any(name in cues for name in names)


def _has_structured_spatial_context(front_payload: dict) -> bool:
    cues = _geometry_cues(front_payload)
    if cues & {"triangle", "equilateral_triangle", "right_isosceles_triangle", "right_triangle", "square", "rectangle", "midpoint", "collinear"}:
        return True
    geometry = _geometry_structures(front_payload)
    return bool(geometry.get("triangles") or geometry.get("squares"))


def _derived_dimension_facts(front_payload: dict, known_counts: Counter[str]) -> list[dict]:
    """Return conservative dimension facts implied by structural knowledge.

    These facts do not add numeric values. They only tell the constraint graph
    that a deterministic constructor can provide repeated dimensions, such as
    three equal side lengths of an equilateral triangle or two identical source
    charges described by one value.
    """

    text = str(front_payload.get("canonical_question") or "").lower()
    facts: list[dict] = []
    length_completion = _triangle_length_completion(front_payload, text, known_counts)
    if length_completion is not None:
        for _ in range(max(0, length_completion["target_count"] - known_counts["length"])):
            facts.append(length_completion["fact"])
    repeated_charge_count = _repeated_charge_count(text)
    if repeated_charge_count and known_counts["charge"] >= 1:
        desired_charge_count = repeated_charge_count
        if _has_extra_target_charge_context(text):
            desired_charge_count = max(desired_charge_count, repeated_charge_count + 1)
        for _ in range(max(0, desired_charge_count - known_counts["charge"])):
            facts.append(
                {
                    "dimension": "charge",
                    "source": f"{repeated_charge_count} equal source charges share one stated value",
                    "derived_by": "multiplicity:repeated_equal_charges",
                }
            )
    if known_counts["length"] >= 2 and _collinear_two_source_metric_completion_context(front_payload, text):
        for _ in range(max(0, 3 - known_counts["length"])):
            facts.append(
                {
                    "dimension": "length",
                    "source": "collinear two-source geometry derives the missing source-target or source-source distance",
                    "derived_by": "geometry:collinear_metric_completion",
                }
            )
    if known_counts["force"] == 1 and re.search(r"\btwo\s+(?:electric\s+)?forces\b|\beach\s+with\s+(?:a\s+)?magnitude\b", text):
        facts.append(
            {
                "dimension": "force",
                "source": "two equal forces share one stated magnitude",
                "derived_by": "multiplicity:two_equal_forces",
            }
        )
    if known_counts["current"] == 0 and re.search(r"\b(?:i|current)\s*(?:\(\s*t\s*\))?\s*(?:=|is)\s*\d+(?:\.\d+)?(?:\s*√\s*2)?\s*(?:sin|cos)\s*\(?\s*\d", text):
        facts.append(
            {
                "dimension": "current",
                "source": "time-dependent current expression supplies current amplitude/function",
                "derived_by": "expression:time_dependent_current",
            }
        )
    if known_counts["angle"] == 0 and any(cue in text for cue in ["perpendicular", "right angle", "90 degrees", "90°", "out of phase"]):
        facts.append(
            {
                "dimension": "angle",
                "source": "orthogonal vector/phase wording implies a right angle",
                "derived_by": "language:orthogonal_angle",
            }
        )
    return facts


def _triangle_length_completion(front_payload: dict, text: str, known_counts: Counter[str]) -> dict | None:
    """Describe when geometry constructors can supply missing triangle lengths.

    This adds only dimension reachability, not a numerical side. It lets the
    constraint graph agree with the spatial engine for general triangle
    constructors such as equilateral equality, right-isosceles equality, or a
    right triangle where two sides determine the third by Pythagoras.
    """

    known_lengths = known_counts["length"]
    if known_lengths <= 0:
        return None
    reasons: list[str] = []
    target_count = known_lengths
    if _has_geometry_cue(front_payload, "equilateral_triangle") or "equilateral triangle" in text or "regular triangle" in text:
        target_count = max(target_count, 3)
        reasons.append("equilateral triangle side equality")
    if (
        _has_geometry_cue(front_payload, "right_isosceles_triangle")
        or "right isosceles triangle" in text
        or "isosceles right triangle" in text
    ) and known_lengths >= 1:
        target_count = max(target_count, 3)
        reasons.append("right-isosceles triangle side equality and Pythagoras")
    if _right_triangle_metric_context(front_payload, text) and known_lengths >= 2:
        target_count = max(target_count, 3)
        reasons.append("right triangle with two known sides")
    if target_count <= known_lengths:
        return None
    return {
        "target_count": target_count,
        "fact": {
            "dimension": "length",
            "source": "; ".join(reasons),
            "derived_by": "geometry:triangle_metric_completion",
        },
    }


def _right_triangle_metric_context(front_payload: dict, text: str) -> bool:
    if _has_geometry_cue(front_payload, "right_triangle", "right_isosceles_triangle"):
        return True
    return bool(
        "right-angled triangle" in text
        or "right angled triangle" in text
        or "right triangle" in text
        or "hypotenuse" in text
    )


def _two_identical_charges_cue(text: str) -> bool:
    return _repeated_charge_count(text) == 2


def _repeated_charge_count(text: str) -> int | None:
    """Infer repeated equal/identical charge multiplicity without row wording.

    The returned count is only a dimension-reachability fact. It never creates
    numeric values; deterministic engines still bind the actual charge values.
    """

    patterns = (
        r"\b(?P<count>one|two|three|four|\d+)\s+(?:identical|equal|same)\s+(?:point\s+|electric\s+)?charges?\b",
        r"\b(?P<count>one|two|three|four|\d+)\s+(?:point\s+|electric\s+)?charges?\s+(?:with\s+)?(?:equal|identical|same)\s+(?:magnitude|charge)\b",
        r"\b(?P<count>one|two|three|four|\d+)\s+[+-]?\s*(?:\d+(?:\.\d+)?|\.\d+)"
        r"(?:\s*(?:×|x|\*)\s*10\s*\^?\s*[+-]?\d+)?\s*(?:μc|µc|uc|nc|pc|mc|c)\s+"
        r"(?:point\s+|electric\s+)?charges?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        count = _parse_count(match.group("count"))
        if count is not None and count > 1:
            return count
    return None


def _parse_count(raw: str) -> int | None:
    lowered = str(raw or "").lower()
    if lowered in COUNT_WORDS:
        return COUNT_WORDS[lowered]
    try:
        value = int(lowered)
    except ValueError:
        return None
    return value if value > 0 else None


def _has_extra_target_charge_context(text: str) -> bool:
    if re.search(r"\b(?:remaining|last|third|other|unoccupied|empty)\s+(?:vertex|corner|point)\b", text):
        return True
    if re.search(
        r"\b(?:charge\s+[a-z]\w*\s*=|(?:a|an|the|test|probe|target)\s+(?:point\s+|electric\s+)?charge\b)"
        r".{0,180}\b(?:two|three|four|\d+)\s+"
        r"(?:[+-]?\s*(?:\d+(?:\.\d+)?|\.\d+)(?:\s*(?:×|x|\*)\s*10\s*\^?\s*[+-]?\d+)?\s*(?:μc|µc|uc|nc|pc|mc|c)\s+)?"
        r"(?:point\s+|electric\s+|positive\s+|negative\s+)?charges?\b",
        text,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:and|with|while)\s+(?:a|an|another|one|the|test|probe)\s+"
            r"(?:point\s+|electric\s+)?charge\b",
            text,
        )
    )


def _collinear_two_source_metric_completion_context(front_payload: dict, text: str) -> bool:
    line_context = any(
        cue in text
        for cue in [
            "collinear",
            "same line",
            "straight line",
            "opposite sides",
            "passing through",
            "line segment",
            "endpoints",
        ]
    ) or _has_geometry_cue(front_payload, "collinear")
    if not line_context:
        return False
    return bool(
        any(cue in text for cue in ["net", "resultant", "magnitude", "direction", "force", "field"])
        and any(cue in text for cue in ["distance", "distances", "from", "separated", "apart", "cm", " m"])
    )


def _branch_context(front_payload: dict) -> dict:
    text = str(front_payload.get("canonical_question") or "").lower()
    geometry = _has_structured_spatial_context(front_payload) or any(
        cue in text
        for cue in [
            "triangle",
            "vertices",
            "vertex",
            "right-angled",
            "right angled",
            "right triangle",
            "hypotenuse",
            "altitude",
            "midpoint",
            "perpendicular bisector",
            "square",
            "rectangle",
            "collinear",
            "straight line",
            "same line",
            "opposite sides",
            "passing through",
            "line segment",
            "distance",
            "distances",
            "separated",
            "apart",
        ]
    )
    superposition = _superposition_context(text)
    topology = ((front_payload.get("topology_graph") or {}).get("canonical_form") in {"series_topology", "parallel_topology"})
    return {
        "spatial_preferred": geometry and superposition,
        "topology_preferred": topology,
        "rlc_preferred": any(cue in text for cue in ["rlc", "impedance", "reactance", "resonance", "ac circuit", "z ="]),
    }


def _branch_cost_adjustment(spec: FormulaSpec, branch_context: dict) -> float:
    branch = formula_execution_branch(spec.formula_id)
    if branch_context.get("spatial_preferred"):
        if branch == "spatial_vector":
            return -35.0
        if spec.task_type in {"coulomb_force", "electric_field_point"} and branch == "scalar_equation":
            return 14.0
    if branch_context.get("topology_preferred"):
        if branch == "topology":
            return -25.0
        if spec.task_type in {"ohm_law", "capacitance", "equivalent_resistance"} and branch == "scalar_equation":
            return 8.0
    if branch_context.get("rlc_preferred"):
        if spec.principle_id == "rlc_core":
            return -28.0
        if spec.principle_id == "dc_circuit_core" and spec.task_type in {"electric_power", "ohm_law"}:
            return 12.0
    return 0.0


def _branch_score_adjustment(spec: FormulaSpec, branch_context: dict) -> float:
    return -_branch_cost_adjustment(spec, branch_context)


def _missing_required_dimensions(available_dimensions: Iterable[str], required_dimensions: tuple[str, ...]) -> list[str]:
    pool = Counter(available_dimensions)
    missing: list[str] = []
    for dimension in required_dimensions:
        if pool[dimension] <= 0:
            missing.append(dimension)
        else:
            pool[dimension] -= 1
    return missing


def _formula_is_metadata_only(spec: FormulaSpec) -> bool:
    expression = spec.expression.lower()
    if spec.formula_id in {"conceptual_direct", "yes_no_direct", "multi_output_direct", "measurement_error_direct"}:
        return True
    if not spec.required_dimensions:
        return True
    return any(cue in expression for cue in ["deterministic ", "vector_sum", "vector integral", " by symmetry", " at "])


def _domain_context_score(spec: FormulaSpec, text: str) -> float:
    score = 0.0
    context = {
        "capacitor_core": ("capacitor", "capacitance", "plate"),
        "dc_circuit_core": ("resistor", "resistance", "current", "voltage", "power"),
        "coulomb_core": ("charge", "coulomb", "electric field", "potential"),
        "field_core": ("electric field", "field strength", "uniform field"),
        "lc_core": ("lc", "oscillation", "resonance", "frequency", "period"),
        "rlc_core": ("rlc", "impedance", "reactance", "resonance"),
        "magnetic_core": ("magnetic", "solenoid", "inductor", "flux"),
        "induction_core": ("emf", "induced", "faraday", "flux"),
        "transformer_core": ("transformer", "turns", "primary", "secondary"),
    }
    for cue in context.get(spec.principle_id, ()):
        if cue in text:
            score += 4.0
    if spec.task_type.replace("_", " ") in text:
        score += 3.0
    if "impedance" in text and spec.task_type == "rlc_impedance":
        score += 25.0
    if "resonance" in text and spec.formula_id in {"lc_resonance_capacitance", "lc_resonance_inductance"}:
        score += 22.0
    if "reactance" in text and spec.task_type in {"inductive_reactance", "capacitive_reactance"}:
        score += 12.0
    if "angle" in text and spec.formula_id in {"magnetic_flux_angle", "rlc_phase_angle"}:
        score += 24.0
    has_impedance_symbol = bool(re.search(r"\bz\s*=", text))
    has_reactance_symbol = bool(re.search(r"\b(?:x\s*[lc]|x[lc])\s*=", text))
    has_ac_cue = any(cue in text for cue in ["rlc", "impedance", "reactance", "resonance", "ac circuit"]) or has_impedance_symbol or has_reactance_symbol
    if spec.formula_id in {"rlc_current_from_rlcf_voltage", "rlc_power_impedance", "rlc_power_resonance"} and has_ac_cue:
        score += 34.0
    if spec.formula_id in {"ohm_current", "power_u2r", "power_i2r", "power_ui"} and has_ac_cue:
        score -= 20.0
    if spec.formula_id == "coulomb_force_direction_superposition" and "direction" in text:
        score += 42.0
    return score


def _formula_context_allowed(spec: FormulaSpec, text: str, front_payload: dict | None = None) -> bool:
    formula_id = spec.formula_id
    front_payload = front_payload or {}
    geometry_cue_set = _geometry_cues(front_payload)
    geometry_cues = {
        "right_isosceles": (
            "right isosceles",
            "isosceles right",
            "right-angled isosceles",
            "right angled isosceles",
            "right-angle vertex",
        ),
        "equilateral": ("equilateral",),
        "triangle": (
            "triangle",
            "vertices",
            "vertex",
            "three points",
            "three fixed points",
            "side length",
            "side lengths",
            "distance between",
            "distances to",
            "separated by",
            "separated from",
            "remaining vertex",
            "from point",
            "from charge",
        ),
        "square": ("square",),
        "perpendicular": ("perpendicular", "bisector"),
        "long_wire": ("long wire", "straight wire", "straight conductor", "long straight"),
        "circular_loop": ("loop", "circular", "coil"),
        "loop_current": ("loop", "circular", "coil"),
        "zero_line": ("zero",),
        "ring": ("ring",),
        "semicircular": ("semicircle", "semicircular"),
        "rod": ("rod", "wire"),
        "disk": ("disk",),
        "sheets": ("sheet", "plate"),
        "dielectric": ("dielectric",),
    }
    structural_markers = {
        "right_isosceles": {"right_isosceles_triangle"},
        "equilateral": {"equilateral_triangle"},
        "triangle": {"triangle", "equilateral_triangle", "right_isosceles_triangle", "right_triangle"},
        "square": {"square"},
        "perpendicular": {"perpendicular", "perpendicular_bisector"},
        "zero_line": {"collinear"},
    }
    for marker, cues in geometry_cues.items():
        has_structural_cue = bool(geometry_cue_set & structural_markers.get(marker, set()))
        has_text_cue = any(cue in text for cue in cues)
        if marker in formula_id and not has_structural_cue and not has_text_cue:
            if marker == "triangle" and formula_id == "coulomb_force_triangle_sides" and _collinear_two_source_metric_completion_context(front_payload, text):
                continue
            return False
    if spec.task_type in {"inductive_reactance", "capacitive_reactance"} and "impedance" in text:
        return False
    if formula_id in {"rlc_power_resonance", "rlc_current_resonance", "rlc_voltage_resonance"} and "resonance" not in text and "resonant" not in text:
        return False
    if formula_id.startswith("rlc_resonance_") and not any(cue in text for cue in ["resonance", "resonant", "resonate"]):
        return False
    if formula_id.startswith("rlc_") and not (
        any(cue in text for cue in ["rlc", "impedance", "reactance", "resonance", "ac circuit", "lcω", "lcw", "quadrature", "uam", "u_am"])
        or re.search(r"\bz\s*=", text)
        or re.search(r"\b(?:x\s*[lc]|x[lc])\s*=", text)
    ):
        return False
    if formula_id == "capacitor_geometry_scaled_capacitance" and not (
        any(cue in text for cue in ["distance", "separation", "dielectric", "plate spacing", "plate area", "split in half", "cut in half"])
        and (has_change_factor_cue(text) or re.search(r"\bd\s*=\s*", text))
    ):
        return False
    if formula_id in {"resistance_resistivity", "resistivity_from_resistance"} and not any(cue in text for cue in ["resistivity", "rho", "ρ", "wire", "conductor"]):
        return False
    if formula_id == "coulomb_force_direction_superposition" and "direction" not in text:
        return False
    return True


def _directional_vector_query(text: str, target_dimensions: list[str]) -> bool:
    if "direction" not in text:
        return False
    if not any(dimension in {"force", "electric_field"} for dimension in target_dimensions):
        return False
    return _superposition_context(text) or any(cue in text for cue in ["test charge", "probe charge", "electric force", "electric field"])


def _superposition_context(text: str) -> bool:
    if any(cue in text for cue in ["net", "resultant", "superposition"]):
        return True
    return bool(
        re.search(r"\b(?:two|three|four|\d+|several|multiple)\s+(?:point\s+|electric\s+)?charges?\b", text)
        or re.search(r"\b(?:two|three|four|\d+|several|multiple)\s+(?:forces?|fields?)\b", text)
    )


def _measurement_error_query(text: str) -> bool:
    return any(
        cue in text
        for cue in [
            "uncertainty",
            "percentage error",
            "percent error",
            "relative error",
            "relative uncertainty",
            "measurement error",
            "random error",
            "absolute error",
            "least count",
        ]
    )


def _capacitor_breakdown_charge_query(text: str, target_dimensions: list[str]) -> bool:
    return (
        "charge" in target_dimensions
        and "capacitor" in text
        and any(cue in text for cue in ["breakdown", "maximum charge", "max charge", "without causing"])
    )


def _rlc_frequency_transform_query(text: str, target_dimensions: list[str]) -> bool:
    if not (
        any(dimension in target_dimensions for dimension in ["current", "power", "resistance", "dimensionless", "constant", "voltage"])
        or _frequency_scale_factor_query(text)
    ):
        return False
    if not has_frequency_transform_cue(text):
        return False
    has_xl = bool(re.search(r"\bx\s*_?\s*l\s*=", text) or re.search(r"\bxl\s*=", text))
    has_xc = bool(re.search(r"\bx\s*_?\s*c\s*=", text) or re.search(r"\bxc\s*=", text))
    if "inductive reactance" in text:
        has_xl = True
    if "capacitive reactance" in text:
        has_xc = True
    return has_xl and has_xc


def _rlc_frequency_transform_route(
    text: str,
    target_text: str,
    target_dimensions: list[str],
    answer_type: str,
) -> RouteResult | None:
    if not _rlc_frequency_transform_query(text, target_dimensions):
        return None
    numeric_answer_type = _numeric_answer_type(answer_type)
    if _frequency_scale_factor_query(text, target_text):
        return RouteResult("ohm_law", numeric_answer_type, 0.82, ["RLC resonance frequency multiplier query detected"])
    if "reactance" in target_text or re.search(r"\b(?:x_l|xl|z_l|zl)\b", target_text):
        return RouteResult("inductive_reactance", numeric_answer_type, 0.82, ["RLC frequency-transform reactance query detected"])
    if "impedance" in target_text or re.search(r"\bz\b", target_text):
        return RouteResult("rlc_impedance", numeric_answer_type, 0.82, ["RLC frequency-transform impedance query detected"])
    if "voltage" in target_dimensions:
        return RouteResult("ohm_law", numeric_answer_type, 0.82, ["RLC frequency-transform voltage query detected"])
    task_type = "electric_power" if "power" in target_dimensions else "ohm_law"
    return RouteResult(task_type, numeric_answer_type, 0.82, ["RLC frequency transform query detected"])


def _frequency_scale_factor_query(text: str, target_text: str = "") -> bool:
    """Detect structural frequency-factor/resonance requests.

    This intentionally avoids problem-specific placeholder names such as
    ``k``. The route should fire when the wording asks for a multiplicative
    change/ratio or for the frequency needed to reach resonance.
    """

    combined = f"{text} {target_text}".lower()
    return bool(
        has_change_factor_cue(combined)
        or re.search(r"\bby\s+what\s+factor\b", combined)
        or re.search(r"\bhow\s+many\s+times\b", combined)
        or re.search(r"\b(?:frequency|omega|ω)\s+(?:factor|multiplier|multiple|ratio)\b", combined)
        or any(
            cue in combined
            for cue in [
                "to achieve resonance",
                "to obtain resonance",
                "to reach resonance",
                "to be resonant",
                "to resonate",
                "resonant frequency",
            ]
        )
    )


def _rlc_current_ratio_reactance_query(text: str) -> bool:
    return bool(
        any(cue in text for cue in ["rlc", "ac circuit", "resonant current", "resonance"])
        and has_frequency_transform_cue(text)
        and "current" in text
        and re.search(r"\b(?:z_l|zl|x_l|xl|inductive reactance)\b", text)
    )


def _symmetry_zero_query(front_payload: dict, text: str, target_dimensions: list[str]) -> bool:
    if not any(dimension in {"electric_field", "force"} for dimension in target_dimensions):
        return False
    has_symmetry_point = any(
        cue in text
        for cue in ["center", "centre", "centroid", "intersection point of the square", "intersection point of the diagonals", "diagonals"]
    )
    if not has_symmetry_point:
        return False
    if not (
        _has_geometry_cue(front_payload, "square", "equilateral_triangle")
        or any(cue in text for cue in ["square", "equilateral triangle", "regular triangle"])
    ):
        return False
    return bool(
        re.search(r"\b(?:identical|equal|same)\s+(?:positive\s+|negative\s+)?charges\b", text)
        or re.search(r"\b(?:all|three|four)\s+(?:identical|equal)\b", text)
        or re.search(r"\bsame\s+magnitude\b", text)
    )


def _electric_field_zero_unknown_charge_query(front_payload: dict, text: str, target_dimensions: list[str], available_counts: Counter) -> bool:
    if "charge" not in target_dimensions:
        return False
    if available_counts["charge"] < 1:
        return False
    if "electric field" not in text and "field strength" not in text:
        return False
    if not re.search(r"\b(?:zero|vanish|cancel|is\s+0)\b|e\s*=\s*0", text):
        return False
    if not re.search(r"\b(?:what|find|determine|calculate)\b[^.?;]{0,80}\bcharge\b", text):
        return False
    return _has_structured_spatial_context(front_payload) or any(
        cue in text for cue in ["centroid", "center", "centre", "midpoint", "vertex", "vertices", "line"]
    )


def _inverse_square_midpoint_field_expression_query(front_payload: dict, text: str) -> bool:
    if "midpoint" not in text:
        return False
    if "field" not in text:
        return False
    if not any(goal.get("dimension") == "electric_field" for goal in front_payload.get("goals", []) or []):
        return False

    symbolic_fields = [
        quantity
        for quantity in front_payload.get("symbolic_quantities", []) or []
        if quantity.get("dimension") == "electric_field"
        or str(quantity.get("symbol") or "").lower().startswith("e")
    ]
    relation_text = " ".join(
        str(item.get("expression") or item.get("raw_text") or item.get("evidence") or "")
        for item in [
            *(front_payload.get("symbolic_relations", []) or []),
            *(front_payload.get("constraints", []) or []),
            *(front_payload.get("relations", []) or []),
        ]
    ).lower()
    combined = f"{text} {relation_text}"
    inverse_square_cue = any(cue in combined for cue in ["1/sqrt", "inverse-square", "inverse square", "proportional to distance"])
    endpoint_context = bool(
        len(symbolic_fields) >= 3
        or "field line" in combined
        or re.search(r"\b(?:endpoints?|two\s+points?|between\s+two\s+points?|at\s+two\s+locations?)\b", combined)
    )
    return inverse_square_cue and endpoint_context


def _midpoint_equal_source_zero_query(front_payload: dict, text: str, target_dimensions: list[str]) -> bool:
    if not any(dimension in {"electric_field", "force"} for dimension in target_dimensions):
        return False
    if "midpoint" not in text and not _has_geometry_cue(front_payload, "midpoint"):
        return False
    return bool(re.search(r"\bequal\s+magnitude\b|\bsame\s+sign\b|\bidentical\b|\bequal\s+charges\b", text))


def _resultant_force_query(text: str, available_dimensions: list[str], target_dimensions: list[str]) -> bool:
    if not any(dimension in {"force", "angle"} for dimension in target_dimensions):
        return False
    if Counter(available_dimensions)["force"] < 1:
        return False
    return bool(
        re.search(r"\bresultant\s+force\b", text)
        or re.search(r"\bnet\s+force\b", text)
        or re.search(r"\btwo\s+(?:electric\s+)?forces?\b", text)
    )


def _relationship_rule_query(text: str) -> bool:
    if "relationship" not in text and "relation between" not in text:
        return False
    return any(cue in text for cue in ["electric field", "field strength", "force", "charge", "current", "voltage"])


def _symbolic_or_graph_theory_query(text: str, answer_type: str, target_dimensions: list[str] | None = None) -> bool:
    if answer_type not in {"symbolic", "conceptual"}:
        return False
    if any(dimension in {"electric_field", "force"} for dimension in (target_dimensions or [])) and _superposition_context(text):
        return False
    if any(dimension in {"electric_field", "force"} for dimension in (target_dimensions or [])) and any(
        cue in text for cue in ["triangle", "square", "rectangle", "collinear", "perpendicular bisector", "altitude", "hypotenuse"]
    ):
        return False
    return any(
        cue in text
        for cue in [
            "formula for",
            "what is the formula",
            "expression for",
            "what is the expression",
            "shape of the graph",
            "graph representing",
            "as a function of",
        ]
    )


def _rlc_quadrature_split_route(
    text: str,
    target_dimensions: list[str],
    answer_type: str,
    available_counts: Counter[str],
) -> RouteResult | None:
    if available_counts["voltage"] < 1 or available_counts["resistance"] < 1:
        return None
    if not (
        "lcω" in text
        or "lcw" in text
        or "lc omega" in text
        or "lcω²" in text
        or re.search(r"\blc\s*(?:ω|w|omega)\s*\^?\s*2\s*=\s*1\b", text)
    ):
        return None
    if not any(cue in text for cue in ["quadrature", "90", "perpendicular", "out of phase"]):
        return None
    numeric_answer_type = _numeric_answer_type(answer_type)
    if "power factor" in text or "cos phi" in text or "cosφ" in text:
        return RouteResult("power_factor", numeric_answer_type, 0.84, ["two-section RLC quadrature gives zero phase angle"])
    if re.search(r"\br\s*2\b|value of r2|resistor r2", text) and available_counts["power"] >= 1:
        return RouteResult("ohm_law", numeric_answer_type, 0.84, ["two-section RLC quadrature unknown-resistance query detected"])
    if "power" in target_dimensions or "power consumed" in text or "total power" in text:
        return RouteResult("electric_power", numeric_answer_type, 0.84, ["two-section RLC quadrature reduces to R1+R2 for power"])
    if "current" in target_dimensions or "current in the circuit" in text or "rms current" in text:
        return RouteResult("ohm_law", numeric_answer_type, 0.84, ["two-section RLC quadrature reduces to R1+R2 for current"])
    if "voltage" in target_dimensions or re.search(r"\bu\s*_?\s*(?:am|mb)\b|voltage across (?:am|mb|segment)", text):
        return RouteResult("ohm_law", numeric_answer_type, 0.84, ["two-section RLC quadrature segment-voltage query detected"])
    return None


def _branch_additive_route(
    text: str,
    target_dimensions: list[str],
    answer_type: str,
    available_counts: Counter[str],
) -> RouteResult | None:
    if any(cue in text for cue in ["branch", "ammeter", "lamp", "bulb", "parallel"]):
        if "current" in target_dimensions and available_counts["current"] >= 1:
            return RouteResult("electric_current", _numeric_answer_type(answer_type), 0.74, ["branch current conservation/sum detected"])
        if "power" in target_dimensions and available_counts["power"] >= 2:
            return RouteResult("electric_power", _numeric_answer_type(answer_type), 0.74, ["load power sum detected"])
    if ("total current" in text or "main current" in text) and available_counts["current"] >= 1:
        return RouteResult("electric_current", _numeric_answer_type(answer_type), 0.72, ["total current from branch current statement detected"])
    if "total power" in text and available_counts["power"] >= 2:
        return RouteResult("electric_power", _numeric_answer_type(answer_type), 0.72, ["total power from load powers detected"])
    return None


def _conceptual_signal(text: str, concepts: set[str]) -> bool:
    if concepts and not any(cue in text for cue in ["calculate", "find", "determine", "what is", "what ", "how much", "needed", "required"]):
        return True
    return any(
        cue in text
        for cue in [
            "explain",
            "why",
            "graph shape",
            "relationship",
            "increases or decreases",
            "true or false",
        ]
    )


def _numeric_answer_type(answer_type: str) -> str:
    return answer_type if answer_type in {"numeric", "symbolic", "unknown"} else "numeric"


def _target_requests_equivalent_resistance(text: str) -> bool:
    if "impedance" in text or "reactance" in text:
        return False
    return any(cue in text for cue in ["equivalent resistance", "total resistance", "combined resistance", "effective resistance"])


def select_minimal_equation_subset(front_payload: dict, route_result, target_dimension: str | None = None) -> PrincipleSelection:
    """Select registry formulas that are dimension-connected to current inputs.

    This is intentionally conservative. It does not invoke SymPy; it only
    prepares the small connected subset that a future symbolic worker may use.
    """

    graph = build_constraint_graph(front_payload, route_result)
    if route_result.task_type == "unknown":
        return PrincipleSelection(False, [], ["unknown_route"], {"stage": "constraint_graph", "graph": graph.to_dict()})
    available = _front_available_dimensions(front_payload)
    selected: List[str] = []
    graph_selected = graph.selected_formula_ids or graph.reachable_formula_ids
    for formula_id in graph_selected:
        spec = FORMULA_REGISTRY.get(formula_id)
        if spec is None:
            continue
        if spec.task_type != route_result.task_type:
            continue
        if target_dimension and spec.target_dimension != target_dimension:
            continue
        if _has_required_dimensions(available, spec.required_dimensions) or formula_id in graph.selected_formula_ids:
            selected.append(formula_id)
    for formula_id in _structural_formula_overrides(front_payload, route_result):
        if formula_id not in selected:
            selected.append(formula_id)
    issues = [] if selected else ["no_connected_formula_subset"]
    principle_ids = sorted({FORMULA_REGISTRY[formula_id].principle_id for formula_id in selected})
    unknown_principles = [principle_id for principle_id in principle_ids if principle_id not in PRINCIPLE_IDS]
    issues.extend(f"unknown_principle:{principle_id}" for principle_id in unknown_principles)
    return PrincipleSelection(
        ok=not issues,
        formula_ids=selected,
        issues=issues,
        trace={
            "stage": "constraint_graph",
            "route_task_type": route_result.task_type,
            "available_dimensions": available,
            "target_dimension": target_dimension,
            "selected_count": len(selected),
            "graph": graph.to_dict(),
        },
    )


def _has_required_dimensions(available: list[str | None], required: tuple[str, ...]) -> bool:
    pool = list(available)
    for dimension in required:
        if dimension not in pool:
            return False
        pool.remove(dimension)
    return True


def _structural_formula_overrides(front_payload: dict, route_result) -> list[str]:
    """Add formulas whose missing intermediate is constructed by a specialized engine.

    The constraint graph is dimension-based, so it cannot see intermediate
    quantities that are produced from structured text, such as capacitor energy
    computed from C and U before applying LC energy conservation. These
    overrides stay registry-bound and only fire on broad physical structures.
    """

    text = str(front_payload.get("canonical_question") or "").lower()
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    available = _front_available_dimensions(front_payload)
    formulas: list[str] = []
    if (
        route_result.task_type == "inductor_energy"
        and "energy" in available
        and "capacitance" in available
        and "voltage" in available
        and any(cue in text for cue in ["lc circuit", "oscillation", "ideal lc"])
        and any(cue in f"{text} {target_text}" for cue in ["magnetic field energy", "magnetic energy", "inductor energy"])
    ):
        formulas.append("lc_energy_complement")
    if (
        route_result.task_type == "ohm_law"
        and "rlc" in text
        and any(cue in text for cue in [" cos", " sin", "cos(", "sin("])
        and any(cue in target_text for cue in ["current", "rms", "effective"])
        and "resistance" in available
        and re.search(r"\bl\s*=\s*[^,.?;]+\bh\b", text)
        and re.search(r"\bc\s*=\s*[^,.?;]+\bf\b", text)
    ):
        formulas.append("rlc_current_from_rlcf_voltage")
    return formulas


def build_registry_symbolic_plan(front_payload: dict, route_result, selection: PrincipleSelection) -> dict | None:
    """Build a safe symbolic plan from a minimal registry-owned equation graph.

    The graph bridge is deliberately narrow: it uses only formulas from
    code-owned registries, substitutes only auditable extracted quantities or
    physical constants, and rejects ambiguous repeated dimensions. Unlike the
    original one-equation bridge, this can also solve safe inverse formulas and
    small connected subsets with deterministic intermediate unknowns.
    """

    if route_result.task_type == "unknown":
        return None
    candidates = _candidate_formulas(route_result, selection)
    target_dimensions = _target_dimensions(front_payload, route_result, candidates)
    for target_dimension in target_dimensions:
        for target_symbol in _target_symbols(front_payload, target_dimension):
            max_size = min(3, len(candidates))
            for size in range(1, max_size + 1):
                for combo in combinations(candidates, size):
                    plan = _build_symbolic_plan_for_combo(front_payload, combo, target_symbol, target_dimension)
                    if plan is not None:
                        return plan
    return None


def _candidate_formulas(route_result, selection: PrincipleSelection) -> list[FormulaSpec]:
    selected_ids = list(selection.formula_ids) if selection is not None else []
    route_principles = {
        spec.principle_id
        for spec in FORMULA_REGISTRY.values()
        if spec.task_type == route_result.task_type
    }
    scored: list[tuple[int, str, FormulaSpec]] = []
    for formula_id, spec in FORMULA_REGISTRY.items():
        if formula_id not in FORMULA_IDS:
            continue
        if spec.task_type != route_result.task_type and spec.principle_id not in route_principles:
            continue
        if not _is_safe_equation_expression(spec.expression):
            continue
        score = 0
        if formula_id in selected_ids:
            score -= 30
        if spec.task_type == route_result.task_type:
            score -= 20
        if spec.target_dimension in _ROUTE_TARGET_PRIORITY.get(route_result.task_type, ()):
            score -= 5
        scored.append((score, formula_id, spec))
    return [spec for _, _, spec in sorted(scored, key=lambda item: (item[0], item[1]))]


def _target_dimensions(front_payload: dict, route_result, candidates: list[FormulaSpec]) -> list[str]:
    ordered: list[str] = []

    def add(dimension: str | None) -> None:
        if dimension and dimension not in ordered:
            ordered.append(dimension)

    for quantity in front_payload.get("symbolic_quantities", []):
        add(quantity.get("dimension"))
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    for keyword, dimension in TARGET_DIMENSION_KEYWORDS:
        if keyword in target_text:
            add(dimension)
    for dimension in _ROUTE_TARGET_PRIORITY.get(route_result.task_type, ()):
        add(dimension)
    for spec in candidates:
        if spec.task_type == route_result.task_type:
            add(spec.target_dimension)
    return ordered


def _target_symbols(front_payload: dict, target_dimension: str) -> list[str]:
    symbols: list[str] = []
    for quantity in front_payload.get("symbolic_quantities", []):
        if quantity.get("dimension") == target_dimension:
            symbol = _clean_symbol(str(quantity.get("symbol") or ""))
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    for symbol in DIMENSION_SYMBOLS.get(target_dimension, ()):
        if symbol.isidentifier() and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _build_symbolic_plan_for_combo(front_payload: dict, combo: tuple[FormulaSpec, ...], target_symbol: str, target_dimension: str) -> dict | None:
    if not all(_formula_allowed_by_front(front_payload, spec) for spec in combo):
        return None
    equations: list[str] = []
    formula_ids: list[str] = []
    principle_ids: list[str] = []
    produced_symbols: set[str] = set()
    symbol_dimensions: dict[str, str] = {}
    for spec in combo:
        sanitized = _sanitize_equation(spec.expression)
        if sanitized is None:
            return None
        lhs, _ = sanitized.split("=", 1)
        lhs_symbol = _clean_symbol(lhs)
        if lhs_symbol is None:
            return None
        equations.append(sanitized)
        formula_ids.append(spec.formula_id)
        if spec.principle_id not in principle_ids:
            principle_ids.append(spec.principle_id)
        produced_symbols.add(lhs_symbol)
        symbol_dimensions[lhs_symbol] = spec.target_dimension
    symbols = set().union(*(_equation_symbols(equation) for equation in equations))
    substitutions = _known_symbol_values(front_payload, symbols, target_symbol)
    constant_substitutions = _constant_substitutions(front_payload, symbols)
    if not substitutions and not constant_substitutions:
        return None
    unresolved = symbols - set(substitutions) - set(constant_substitutions) - _SAFE_FUNCTIONS
    unknowns = sorted(unresolved)
    if target_symbol not in unknowns:
        return None
    if not set(unknowns).issubset(produced_symbols | {target_symbol}):
        return None
    if len(equations) < len(unknowns):
        return None
    numeric_equations = [_substitute_symbols(equation, {**substitutions, **constant_substitutions}) for equation in equations]
    targets = [{"symbol": target_symbol, "dimension": target_dimension, "unit": _unit_for_dimension(target_dimension)}]
    for symbol in unknowns:
        if symbol == target_symbol:
            continue
        dimension = symbol_dimensions.get(symbol) or _dimension_for_symbol(symbol) or "dimensionless"
        targets.append({"symbol": symbol, "dimension": dimension, "unit": _unit_for_dimension(dimension)})
    return {
        "symbolic_family": "registry_equation_graph",
        "formula_ids": formula_ids,
        "principle_ids": principle_ids,
        "equations": numeric_equations,
        "targets": targets,
        "timeout_seconds": 3.0,
        "non_negative_target": _should_force_non_negative(front_payload, target_dimension),
        "trace": {
            "stage": "principle_equation_graph",
            "mode": "minimal_registry_subset",
            "formula_ids": formula_ids,
            "substitutions": substitutions,
            "constant_substitutions": constant_substitutions,
            "source_expressions": [spec.expression for spec in combo],
            "unknowns": unknowns,
        },
    }


def _formula_allowed_by_front(front_payload: dict, spec: FormulaSpec) -> bool:
    text = " ".join(
        [
            str(front_payload.get("canonical_question") or ""),
            " ".join(front_payload.get("target_hints") or []),
            " ".join(str(concept) for concept in front_payload.get("concepts") or []),
            " ".join(str((relation or {}).get("qualifier") or "") for relation in front_payload.get("relations") or [] if isinstance(relation, dict)),
            " ".join(str((constraint or {}).get("expression") or "") for constraint in front_payload.get("constraints") or [] if isinstance(constraint, dict)),
        ]
    ).lower()
    formula_id = spec.formula_id
    if formula_id.startswith("topology_") or spec.principle_id == "topology_core":
        topology = front_payload.get("topology_graph") or {}
        canonical = str(topology.get("canonical_form") or "").lower() if isinstance(topology, dict) else ""
        return canonical not in {"", "no_circuit_topology"} or any(cue in text for cue in ["series", "parallel"])
    if formula_id.startswith("rlc_") or spec.principle_id == "rlc_core":
        has_rlc_grounding = any(
            cue in text
            for cue in [
                "rlc",
                "ac circuit",
                "alternating",
                "reactance",
                "impedance",
                "resonance",
                "resonant",
                "resonate",
                "inductor",
                "inductance",
                "capacitor",
                "capacitance",
                "lcω",
                "lcw",
                "u_am",
                "u_mb",
                "out of phase",
                "phase",
            ]
        )
        if not has_rlc_grounding:
            return False
        if "quadrature" in formula_id:
            return any(cue in text for cue in ["quadrature", "out of phase", "90 degree", "90 degrees", "90°", "u_am", "u_mb", "segment", "section"])
    return True


_ROUTE_TARGET_PRIORITY = {
    "capacitor_charge": ("charge",),
    "capacitance": ("capacitance",),
    "capacitor_energy": ("energy",),
    "capacitor_final_voltage": ("voltage",),
    "ohm_law": ("current", "voltage", "resistance"),
    "equivalent_resistance": ("resistance",),
    "electric_current": ("current",),
    "electric_charge_transport": ("charge", "time"),
    "power_energy_time": ("energy", "power", "time"),
    "resistance_material": ("resistance", "resistivity"),
    "electric_power": ("power",),
    "inductor_energy": ("energy", "current"),
    "inductance": ("inductance",),
    "lc_frequency": ("frequency",),
    "lc_period": ("time",),
    "faraday_induction": ("voltage", "magnetic_flux", "time", "current"),
    "rlc_impedance": ("resistance",),
    "power_factor": ("dimensionless",),
    "magnetic_flux": ("magnetic_flux",),
    "solenoid_magnetic_field": ("magnetic_field",),
    "magnetic_field": ("magnetic_field",),
    "turn_density": ("turn_density",),
}


TARGET_DIMENSION_KEYWORDS = (
    ("capacitance", "capacitance"),
    ("capacity", "capacitance"),
    ("area", "area"),
    ("charge", "charge"),
    ("turns", "count"),
    ("number of turns", "count"),
    ("current", "current"),
    ("angular frequency", "angular_frequency"),
    ("omega", "angular_frequency"),
    ("voltage", "voltage"),
    ("potential difference", "voltage"),
    ("resistance", "resistance"),
    ("resistivity", "resistivity"),
    ("impedance", "resistance"),
    ("power factor", "dimensionless"),
    ("random error", "uncertainty"),
    ("absolute error", "uncertainty"),
    ("measurement error", "uncertainty"),
    ("power", "power"),
    ("energy", "energy"),
    ("work", "energy"),
    ("frequency", "frequency"),
    ("period", "time"),
    ("time", "time"),
    ("magnetic flux density", "magnetic_field"),
    ("flux density", "magnetic_field"),
    ("electric field", "electric_field"),
    ("field strength", "electric_field"),
    ("magnetic field", "magnetic_field"),
    ("magnetic flux", "magnetic_flux"),
    ("flux", "magnetic_flux"),
    ("force", "force"),
    ("mass", "mass"),
    ("distance", "length"),
    ("speed", "velocity"),
)


ROUTER_TARGET_DIMENSION_KEYWORDS = (
    (("capacitance", "capacity", "capacitance value", "unknown capacitance", "value of c"), "capacitance"),
    (("area", "plate area", "surface area"), "area"),
    (("charge", "amount of charge", "magnitude of q"), "charge"),
    (("turns", "number of turns"), "count"),
    (("current", "amperage"), "current"),
    (("angular frequency", "omega", "ω"), "angular_frequency"),
    (("voltage", "potential difference", "emf", "electromotive force"), "voltage"),
    (("resistance", "impedance", "reactance"), "resistance"),
    (("resistivity",), "resistivity"),
    (("power factor", "cos phi", "cosφ"), "dimensionless"),
    (("percentage uncertainty", "percent uncertainty", "percentage error", "percent error", "relative error"), "percent"),
    (("random error", "absolute error", "measurement error"), "uncertainty"),
    (("power",), "power"),
    (("energy", "work", "heat"), "energy"),
    (("frequency", "resonant frequency"), "frequency"),
    (("period", "time constant", "how long", "time"), "time"),
    (("electric field", "field strength"), "electric_field"),
    (("magnetic flux density", "flux density"), "magnetic_field"),
    (("magnetic field",), "magnetic_field"),
    (("magnetic flux", "flux"), "magnetic_flux"),
    (("force",), "force"),
    (("mass",), "mass"),
    (("distance", "separation", "radius"), "length"),
    (("speed", "velocity"), "velocity"),
    (("inductance", "inductance value", "unknown inductance", "value of l"), "inductance"),
    (("turn density", "turns per meter"), "turn_density"),
)


_SAFE_FUNCTIONS = {"sqrt", "sin", "cos", "tan", "atan", "pi"}


def _is_safe_equation_expression(expression: str) -> bool:
    sanitized = _sanitize_equation(expression)
    return sanitized is not None and _is_single_equation(sanitized)


def _sanitize_equation(expression: str) -> str | None:
    if not _is_single_equation(expression):
        return None
    if any(cue in expression for cue in ["'", "|", "vector_sum", " by ", " when ", " or ", " at "]):
        return None
    if "abs(" in expression or "abs " in expression:
        return None
    sanitized = expression.replace("^", "**")
    sanitized = re.sub(r"\s*=\s*", "=", sanitized.strip())
    sanitized = re.sub(r"(?<=\d)(?=[A-Za-z_])", "*", sanitized)
    sanitized = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=[A-Za-z_(])", "*", sanitized)
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().=\s]+", sanitized):
        return None
    return sanitized


def _known_symbol_values(front_payload: dict, symbols: set[str], target_symbol: str) -> dict[str, float]:
    known: dict[str, float] = {}
    dims = _by_dimension(front_payload)
    for quantity in front_payload.get("quantities", []):
        symbol = _clean_symbol(str(quantity.get("symbol") or ""))
        if symbol and symbol in symbols and symbol != target_symbol:
            value = _quantity_si_value(quantity)
            if value is not None:
                known[symbol] = value
    for symbol in symbols:
        if symbol in known or symbol == target_symbol or symbol in _SAFE_FUNCTIONS:
            continue
        dimension = _dimension_for_symbol(symbol)
        if not dimension:
            continue
        values = dims.get(dimension) or []
        if len(values) != 1:
            continue
        value = _quantity_si_value(values[0])
        if value is not None:
            known[symbol] = value
    return known


def _constant_substitutions(front_payload: dict, symbols: set[str]) -> dict[str, float]:
    constants: dict[str, float] = {}
    if "epsilon0" in symbols:
        constants["epsilon0"] = 8.8541878128e-12
    if "mu0" in symbols:
        constants["mu0"] = 4 * 3.141592653589793e-7
    if "epsilon_r" in symbols:
        text = str(front_payload.get("canonical_question") or "").lower()
        if "air" in text or "vacuum" in text or "dielectric" not in text:
            constants["epsilon_r"] = 1.0
    return constants


def _substitute_symbols(equation: str, substitutions: dict[str, float]) -> str:
    substituted = equation
    for symbol, value in sorted(substitutions.items(), key=lambda item: len(item[0]), reverse=True):
        substituted = re.sub(rf"\b{re.escape(symbol)}\b", repr(float(value)), substituted)
    return substituted


def _quantity_si_value(quantity: dict) -> float | None:
    info = unit_info(quantity.get("unit") or "")
    if info is None:
        return None
    try:
        return float(quantity["value"]) * info.si_factor
    except (KeyError, TypeError, ValueError):
        return None


def _dimension_for_symbol(symbol: str) -> str | None:
    matches = [dimension for dimension, symbols in DIMENSION_SYMBOLS.items() if symbol in symbols]
    return matches[0] if len(matches) == 1 else None


def _unit_for_dimension(dimension: str) -> str | None:
    return {
        "capacitance": "F",
        "charge": "C",
        "count": "turns",
        "current": "A",
        "electric_field": "V/m",
        "energy": "J",
        "force": "N",
        "frequency": "Hz",
        "inductance": "H",
        "impedance": "Ω",
        "length": "m",
        "magnetic_field": "T",
        "magnetic_flux": "Wb",
        "power": "W",
        "resistance": "Ω",
        "resistivity": "Ω*m",
        "time": "s",
        "turn_density": "turns/m",
        "angular_frequency": "rad/s",
        "area": "m^2",
        "velocity": "m/s",
        "voltage": "V",
        "dimensionless": "-",
    }.get(dimension)


def _should_force_non_negative(front_payload: dict, target_dimension: str) -> bool:
    text = str(front_payload.get("canonical_question") or "").lower()
    if any(cue in text for cue in ["direction", "signed", "polarity", "opposite direction"]):
        return False
    return target_dimension in {
        "capacitance",
        "current",
        "electric_field",
        "energy",
        "force",
        "frequency",
        "inductance",
        "length",
        "magnetic_field",
        "power",
        "resistance",
        "resistivity",
        "time",
        "angular_frequency",
    }


def _by_dimension(front_payload: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for quantity in front_payload.get("quantities", []):
        dimension = quantity.get("dimension")
        if dimension:
            out.setdefault(dimension, []).append(quantity)
    return out


def _first_dimension_quantity(dims: dict[str, list[dict]], dimension: str) -> dict | None:
    values = dims.get(dimension) or []
    return values[0] if values else None


def _is_single_equation(expression: str) -> bool:
    return isinstance(expression, str) and expression.count("=") == 1


def _clean_symbol(text: str) -> str | None:
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*", text)
    return match.group(1) if match else None


def _equation_symbols(expression: str) -> set[str]:
    symbols = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))
    return symbols - {"sqrt", "sin", "cos", "tan", "atan", "abs", "min", "max"}


DIMENSION_SYMBOLS = {
    "area": ("A", "area"),
    "capacitance": ("C",),
    "charge": ("Q", "q"),
    "count": ("N",),
    "current": ("I",),
    "electric_field": ("E",),
    "energy": ("W", "E"),
    "force": ("F",),
    "frequency": ("f",),
    "inductance": ("L",),
    "length": ("r", "d", "l", "x"),
    "mass": ("m",),
    "magnetic_field": ("B",),
    "magnetic_flux": ("Phi", "flux"),
    "number_density": ("n",),
    "power": ("P",),
    "resistance": ("R", "Z", "XL", "XC"),
    "resistivity": ("rho",),
    "angle": ("theta", "phi"),
    "time": ("t", "T"),
    "velocity": ("v", "v_d"),
    "voltage": ("U", "V"),
    "angular_frequency": ("omega",),
}


def _symbol_for_dimension(dimension: str, candidates: set[str]) -> str | None:
    preferred = DIMENSION_SYMBOLS.get(dimension, ())
    for symbol in preferred:
        if symbol in candidates:
            return symbol
    if len(candidates) == 1:
        return next(iter(candidates))
    return None
