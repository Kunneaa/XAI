"""Structured executable solve plans for NSP-Core.

The local LLM may propose one of these plans, but the plan is always treated as
untrusted until ``plan_compiler`` validates and lowers it. Deterministic code can
also build the same shape, which lets the rest of the pipeline operate through a
single plan boundary instead of raw router branches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..engines.spatial_engine import match_geometry_templates
from ..knowledge.constraint_graph import infer_target_dimensions
from ..knowledge.language import has_frequency_transform_cue
from ..knowledge.registries import FORMULA_REGISTRY, formula_execution_branch
from .answer_formats import build_output_format


@dataclass(frozen=True)
class SolvePlanStep:
    step_id: str
    operation: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    output: str | None = None
    depends_on: List[str] = field(default_factory=list)
    public_cot: str | None = None
    formula_id: str | None = None
    principle_id: str | None = None
    geometry_constructor_id: str | None = None
    logic_rule_id: str | None = None
    topology_rule_id: str | None = None

    def to_dict(self) -> dict:
        payload: dict[str, Any] = {
            "step_id": self.step_id,
            "operation": self.operation,
            "inputs": dict(self.inputs),
            "output": self.output,
            "depends_on": list(self.depends_on),
        }
        if self.public_cot:
            payload["public_cot"] = self.public_cot
        for key in (
            "formula_id",
            "principle_id",
            "geometry_constructor_id",
            "logic_rule_id",
            "topology_rule_id",
        ):
            value = getattr(self, key)
            if value:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class StructuredSolvePlan:
    status: str
    task_type: str
    answer_type: str
    targets: List[Dict[str, Any]]
    steps: List[SolvePlanStep]
    assumptions: List[Dict[str, Any]] = field(default_factory=list)
    output_format: Dict[str, Any] = field(default_factory=dict)
    source: str = "deterministic"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "task_type": self.task_type,
            "answer_type": self.answer_type,
            "targets": [dict(target) for target in self.targets],
            "assumptions": [dict(assumption) for assumption in self.assumptions],
            "steps": [step.to_dict() for step in self.steps],
            "output_format": dict(self.output_format),
            "source": self.source,
            "notes": list(self.notes),
        }


def normalize_plan_payload(payload: dict | StructuredSolvePlan | None, source: str = "unknown") -> dict | None:
    """Return a plain plan dict while preserving only known top-level fields."""

    if payload is None:
        return None
    if isinstance(payload, StructuredSolvePlan):
        return payload.to_dict()
    if not isinstance(payload, dict):
        return None
    return {
        "status": payload.get("status"),
        "task_type": payload.get("task_type"),
        "answer_type": payload.get("answer_type"),
        "targets": list(payload.get("targets") or []),
        "assumptions": list(payload.get("assumptions") or []),
        "steps": list(payload.get("steps") or []),
        "output_format": dict(payload.get("output_format") or {}),
        "source": payload.get("source") or source,
        "notes": list(payload.get("notes") or []),
    }


def build_deterministic_solve_plan(front_payload: dict, route_result, graph_selection=None) -> StructuredSolvePlan:
    """Build a registry-backed plan without using an LLM.

    This is the fallback and default path. It keeps legacy deterministic routing
    behavior, but exposes the route as an executable step DAG for verification,
    trace generation, and future LLM-plan replacement.
    """

    task_type = getattr(route_result, "task_type", "unknown")
    answer_type = front_payload.get("answer_type_hint") or getattr(route_result, "answer_type", "unknown")
    targets = _targets(front_payload)
    if task_type == "unknown":
        return StructuredSolvePlan(
            status="unsupported",
            task_type=task_type,
            answer_type=answer_type,
            targets=targets,
            steps=[],
            output_format=build_output_format(front_payload, "unknown", task_type, targets),
            source="deterministic",
            notes=["No registry-connected route was found."],
        )

    selected_formula_ids = _selected_formula_ids(graph_selection, task_type)
    steps: list[SolvePlanStep] = []
    if task_type == "multi_output" or answer_type == "multi_output":
        steps.extend(_multi_output_steps(targets))
    elif _inverse_square_midpoint_field_expression(front_payload):
        formula_id = "point_charge_field_midpoint_inverse_expression"
        spec = FORMULA_REGISTRY.get(formula_id)
        steps.append(
            SolvePlanStep(
                step_id="s1",
                operation="apply_formula",
                formula_id=formula_id,
                principle_id=spec.principle_id if spec else "field_core",
                inputs={"facts": "formal_ir.symbolic_field_geometry"},
                output=_first_target_id(targets, spec.target_dimension if spec else "dimensionless"),
                public_cot="Use inverse-square field-line proportionality to average inverse-square-root field values at the midpoint.",
            )
        )
    elif task_type in {"coulomb_force", "electric_field_point"} and _geometry_step_needed(front_payload, selected_formula_ids):
        steps.extend(_spatial_steps(front_payload, task_type, targets, selected_formula_ids))
    elif task_type == "conceptual" or answer_type in {"conceptual", "yes_no"}:
        steps.append(
            SolvePlanStep(
                step_id="s1",
                operation="apply_logic_rule",
                principle_id="conceptual_core",
                inputs={"facts": "formal_ir"},
                output=_first_target_id(targets, "conceptual_answer"),
                public_cot="Apply a registry logic rule to the accepted semantic facts.",
            )
        )
    elif task_type == "measurement_error":
        formula_id = _measurement_formula_id(front_payload, targets)
        spec = FORMULA_REGISTRY.get(formula_id)
        steps.append(
            SolvePlanStep(
                step_id="s1",
                operation="apply_formula",
                formula_id=formula_id,
                principle_id=spec.principle_id if spec else "measurement_core",
                inputs={"facts": "formal_ir.quantities"},
                output=_first_target_id(targets, spec.target_dimension if spec else "measurement_error"),
                public_cot=(
                    "Compute absolute random error from repeated measurements."
                    if formula_id == "measurement_absolute_error"
                    else "Compute relative measurement uncertainty from explicit measured value and uncertainty."
                ),
            )
        )
    elif _rlc_frequency_transform_formula_id(front_payload, targets):
        formula_id = str(_rlc_frequency_transform_formula_id(front_payload, targets))
        spec = FORMULA_REGISTRY.get(formula_id)
        steps.append(
            SolvePlanStep(
                step_id="s1",
                operation="apply_formula",
                formula_id=formula_id,
                principle_id=spec.principle_id if spec else "rlc_core",
                inputs={"facts": "formal_ir.rlc_reactance_state"},
                output=_first_target_id(targets, spec.target_dimension if spec else "current"),
                public_cot="Transform frequency-dependent RLC reactances, then apply the registry RLC relation.",
            )
        )
    elif task_type == "inductor_energy" and _lc_energy_complement_formula_id(front_payload, targets):
        formula_id = "lc_energy_complement"
        spec = FORMULA_REGISTRY.get(formula_id)
        steps.append(
            SolvePlanStep(
                step_id="s1",
                operation="apply_formula",
                formula_id=formula_id,
                principle_id=spec.principle_id if spec else "lc_core",
                inputs={"facts": "formal_ir.lc_energy_state"},
                output=_first_target_id(targets, spec.target_dimension if spec else "energy"),
                public_cot="Use LC energy conservation to compute the complementary electric or magnetic energy.",
            )
        )
    elif _canonical_topology_formula_id(front_payload, task_type, targets):
        formula_id = _canonical_topology_formula_id(front_payload, task_type, targets)
        spec = FORMULA_REGISTRY.get(str(formula_id))
        steps.append(
            SolvePlanStep(
                step_id="s1",
                operation="apply_formula",
                formula_id=str(formula_id),
                principle_id=spec.principle_id if spec else "topology_core",
                inputs={"topology": "formal_ir.topology_graph"},
                output=_first_target_id(targets, spec.target_dimension if spec else "result"),
                public_cot=f"Apply registry topology formula {formula_id} to the canonical circuit graph.",
            )
        )
    else:
        formula_ids = selected_formula_ids
        if formula_ids:
            for index, formula_id in enumerate(formula_ids, start=1):
                spec = FORMULA_REGISTRY.get(formula_id)
                output_id = _first_target_id(targets, spec.target_dimension if spec else "result")
                if len(formula_ids) > 1:
                    output_id = f"{output_id}:{formula_id}"
                steps.append(
                    SolvePlanStep(
                        step_id=f"s{index}",
                        operation="apply_formula",
                        formula_id=formula_id,
                        principle_id=spec.principle_id if spec else None,
                        inputs={"dimensions": list(spec.required_dimensions) if spec else []},
                        output=output_id,
                        public_cot=f"Apply registry formula {formula_id} using SI-normalized inputs.",
                    )
                )
        else:
            steps.append(
                SolvePlanStep(
                    step_id="s1",
                    operation="solve_equation_subset",
                    inputs={"route_task_type": task_type},
                    output=_first_target_id(targets, "result"),
                    public_cot="Solve a registry-owned equation subset for the requested target.",
                )
            )

    return StructuredSolvePlan(
        status="ok" if steps else "needs_fallback",
        task_type=task_type,
        answer_type=answer_type,
        targets=targets,
        steps=steps,
        assumptions=_assumptions(front_payload),
        output_format=build_output_format(front_payload, answer_type, task_type, targets),
        source="deterministic",
        notes=["Plan was built from deterministic IR, route, and registry graph."],
    )


def plan_summary(plan: dict | StructuredSolvePlan | None) -> dict:
    payload = normalize_plan_payload(plan)
    if payload is None:
        return {"present": False}
    return {
        "present": True,
        "status": payload.get("status"),
        "task_type": payload.get("task_type"),
        "answer_type": payload.get("answer_type"),
        "source": payload.get("source"),
        "step_count": len(payload.get("steps") or []),
        "operations": [step.get("operation") for step in payload.get("steps") or [] if isinstance(step, dict)],
        "formula_ids": [
            step.get("formula_id")
            for step in payload.get("steps") or []
            if isinstance(step, dict) and step.get("formula_id")
        ],
    }


def _targets(front_payload: dict) -> list[dict]:
    targets: list[dict] = []
    answer_type = str(front_payload.get("answer_type_hint") or "unknown")
    for index, goal in enumerate(front_payload.get("goals") or [], start=1):
        target_id = goal.get("goal_id") or f"target:{index}"
        targets.append(
            {
                "id": str(target_id),
                "quantity": goal.get("dimension") or "unknown",
                "symbol": goal.get("symbol"),
                "text": goal.get("text") or goal.get("raw_text") or "",
                "unit": None if answer_type in {"conceptual", "yes_no"} else _unit_for_dimension(goal.get("dimension")),
            }
        )
    if not targets:
        for index, dimension in enumerate(infer_target_dimensions(front_payload), start=1):
            targets.append(
                {
                    "id": f"target:{index}",
                    "quantity": dimension,
                    "symbol": None,
                    "text": dimension,
                    "unit": None if answer_type in {"conceptual", "yes_no"} else _unit_for_dimension(dimension),
                }
            )
    if not targets:
        targets.append({"id": "target:1", "quantity": "unknown", "symbol": None, "text": "", "unit": None})
    return targets


def _assumptions(front_payload: dict) -> list[dict]:
    out = []
    for fact in front_payload.get("implicit_facts") or []:
        out.append(
            {
                "assumption_id": fact.get("rule_id") or fact.get("fact_id"),
                "trigger_span": fact.get("trigger_text") or fact.get("source"),
                "source": "implicit_kb",
            }
        )
    return out


def _selected_formula_ids(graph_selection, task_type: str) -> list[str]:
    formula_ids = [
        formula_id
        for formula_id in list(getattr(graph_selection, "formula_ids", []) or [])
        if FORMULA_REGISTRY.get(formula_id) and FORMULA_REGISTRY[formula_id].task_type == task_type
    ]
    if formula_ids:
        return formula_ids
    return [
        formula_id
        for formula_id, spec in FORMULA_REGISTRY.items()
        if spec.task_type == task_type and _is_direct_scalar_formula(spec.expression)
    ][:1]


def _measurement_formula_id(front_payload: dict, targets: list[dict]) -> str:
    text = " ".join(
        [str(front_payload.get("canonical_question") or "")]
        + [str(target.get("text") or "") for target in targets]
    ).lower()
    target_quantities = {str(target.get("quantity") or "") for target in targets}
    if "uncertainty" in target_quantities or any(
        cue in text for cue in ["random error", "absolute error", "average absolute error", "mean absolute error"]
    ):
        return "measurement_absolute_error"
    return "measurement_error_direct"


def _geometry_step_needed(front_payload: dict, selected_formula_ids: list[str] | None = None) -> bool:
    if selected_formula_ids and _is_spatial_formula(selected_formula_ids[0]):
        return True
    if match_geometry_templates(front_payload):
        return True
    if _direction_only_two_charge_geometry(front_payload):
        return True
    text = str(front_payload.get("canonical_question") or "").lower()
    if "equidistant" in text and (
        any(cue in text for cue in ["charges", "point charges"]) or _two_named_points_context(text)
    ):
        return True
    counts = {}
    for quantity in front_payload.get("quantities") or []:
        dimension = quantity.get("dimension")
        counts[dimension] = counts.get(dimension, 0) + 1
    structural_counts = _dimension_counts(front_payload)
    return max(counts.get("charge", 0), structural_counts.get("charge", 0)) >= 3 and max(
        counts.get("length", 0), structural_counts.get("length", 0)
    ) >= 2


def _spatial_steps(front_payload: dict, task_type: str, targets: list[dict], selected_formula_ids: list[str] | None = None) -> list[SolvePlanStep]:
    selected_formula_id = _selected_spatial_formula_id(front_payload, task_type, selected_formula_ids or [])
    template_id = _spatial_template_id(front_payload, selected_formula_id)
    first_output = "geom"
    steps = [
        SolvePlanStep(
            step_id="s1",
            operation="construct_geometry",
            geometry_constructor_id=template_id,
            inputs={"facts": "formal_ir.geometry"},
            output=first_output,
            public_cot="Construct deterministic geometry from accepted geometry facts.",
        )
    ]
    if task_type == "coulomb_force":
        formula_id = selected_formula_id or _symmetry_zero_formula_id(front_payload, "force") or (
            "coulomb_force_direction_superposition" if _direction_only_two_charge_geometry(front_payload) else "coulomb_force_triangle_sides"
        )
        spec = FORMULA_REGISTRY.get(formula_id)
        steps.append(
            SolvePlanStep(
                step_id="s2",
                operation=_spatial_operation_for_formula(formula_id),
                formula_id=formula_id,
                principle_id=spec.principle_id if spec else None,
                inputs={"geometry": first_output, "charges": "formal_ir.charges"},
                output=_first_target_id(targets, "net_force"),
                depends_on=["s1"],
                public_cot="Resolve Coulomb-force directions and combine them by superposition."
                if formula_id == "coulomb_force_direction_superposition"
                else "Compute vector Coulomb force contributions and combine them by superposition.",
            )
        )
    else:
        formula_id = selected_formula_id or _symmetry_zero_formula_id(front_payload, "electric_field") or (
            "electric_field_symbolic_superposition"
            if _symbolic_vector_field_case(front_payload)
            else "electric_field_two_charge_triangle_sides"
        )
        spec = FORMULA_REGISTRY.get(formula_id)
        steps.append(
            SolvePlanStep(
                step_id="s2",
                operation="resolve_vector_components",
                formula_id=formula_id,
                principle_id=spec.principle_id if spec else None,
                inputs={"geometry": first_output, "charges": "formal_ir.charges"},
                output=_first_target_id(targets, "net_field"),
                depends_on=["s1"],
                public_cot="Resolve electric-field contributions into components and sum them.",
            )
        )
    return steps


def _multi_output_steps(targets: list[dict]) -> list[SolvePlanStep]:
    steps = []
    for index, target in enumerate(targets, start=1):
        steps.append(
            SolvePlanStep(
                step_id=f"s{index}",
                operation="solve_equation_subset",
                inputs={"target": target["id"]},
                output=target["id"],
                depends_on=[] if index == 1 else [f"s{index-1}"],
                public_cot=f"Solve verified target branch {target['id']} with registry equations.",
            )
        )
    steps.append(
        SolvePlanStep(
            step_id=f"s{len(steps)+1}",
            operation="format_target",
            inputs={"ordered_targets": [target["id"] for target in targets]},
            output="final_answer",
            depends_on=[step.step_id for step in steps],
            public_cot="Format verified target branches in requested order.",
        )
    )
    return steps


def _symbolic_vector_field_case(front_payload: dict) -> bool:
    text = str(front_payload.get("canonical_question") or "").lower()
    has_field_target = any((goal.get("dimension") == "electric_field") for goal in front_payload.get("goals") or [])
    if not has_field_target or front_payload.get("answer_type_hint") != "symbolic":
        return False
    if _has_structured_geometry(front_payload):
        return True
    return bool(re.search(r"\b(?:triangle|square|rectangle|midpoint|bisector|altitude|hypotenuse|center|centre)\b", text))


def _direction_only_two_charge_geometry(front_payload: dict) -> bool:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "direction" not in text or not _direction_probe_geometry_cue(text):
        return False
    counts = _dimension_counts(front_payload)
    return counts.get("charge", 0) >= 2 and counts.get("length", 0) >= 3


def _direction_probe_geometry_cue(lowered_question: str) -> bool:
    return bool(
        re.search(r"\b(?:test|probe|trial|target)\s+charge\b", lowered_question)
        or re.search(r"\bpoint\s+whose\s+distances?\b", lowered_question)
        or re.search(r"\bdistances?\s+to\s+(?:the\s+)?(?:two|three|\d+)\s+charges?\b", lowered_question)
        or re.search(
            r"\b(?:force|field)\s+(?:acting\s+)?(?:on|at|toward|towards)\s+(?:the\s+)?(?:test|probe|trial|target)?\s*charge\b",
            lowered_question,
        )
    )


def _inverse_square_midpoint_field_expression(front_payload: dict) -> bool:
    text = str(front_payload.get("canonical_question") or "").lower()
    if front_payload.get("answer_type_hint") != "symbolic":
        return False
    has_field_target = any((goal.get("dimension") == "electric_field") for goal in front_payload.get("goals") or [])
    has_midpoint = "midpoint" in text or _has_geometry_cue(front_payload, "midpoint")
    has_field_cue = "field" in text or has_field_target
    if not has_field_target or not has_midpoint or not has_field_cue:
        return False
    symbolic_fields = [
        quantity
        for quantity in front_payload.get("symbolic_quantities") or []
        if quantity.get("dimension") == "electric_field" or str(quantity.get("symbol") or "").lower().startswith("e")
    ]
    symbolic_relation_text = " ".join(
        str(relation.get("raw_text") or relation.get("lhs") or "") + " " + str(relation.get("rhs") or "")
        for relation in front_payload.get("symbolic_relations") or []
        if isinstance(relation, dict)
    ).lower()
    inverse_square_cue = any(cue in text or cue in symbolic_relation_text for cue in ["inverse-square", "inverse square", "1/sqrt"])
    sqrt_relation_cue = "sqrt" in symbolic_relation_text and len(symbolic_fields) >= 2
    endpoint_cue = bool(
        len(symbolic_fields) >= 3
        or re.search(r"\b(?:endpoints?|two\s+points?|points?\s+[a-z]\s+and\s+[a-z]|field\s+line)\b", text)
    )
    return (inverse_square_cue or sqrt_relation_cue) and endpoint_cue


def _symmetry_zero_formula_id(front_payload: dict, target_dimension: str) -> str | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    center_cue = _has_geometry_cue(front_payload, "center") or any(cue in text for cue in ["center", "centre", "centroid", "intersection point", "diagonal"])
    midpoint_cue = _has_geometry_cue(front_payload, "midpoint") or "midpoint" in text
    shape_cue = _has_geometry_cue(front_payload, "square", "equilateral_triangle") or any(cue in text for cue in ["square", "equilateral triangle", "regular triangle"])
    equal_cue = bool(
        re.search(r"\b(?:identical|equal|same)\s+(?:positive\s+|negative\s+)?charges\b", text)
        or re.search(r"\b(?:all|three|four)\s+(?:identical|equal)\b", text)
        or re.search(r"\bequal\s+magnitude\b|\bsame\s+magnitude\b|\bsame\s+sign\b", text)
    )
    if not ((center_cue and shape_cue and equal_cue) or (midpoint_cue and equal_cue)):
        return None
    if target_dimension == "force":
        return "symmetric_zero_force"
    if target_dimension == "electric_field":
        return "electric_field_symmetric_zero"
    return None


def _rlc_frequency_transform_formula_id(front_payload: dict, targets: list[dict]) -> str | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not (
        any(cue in text for cue in ["rlc", "reactance", "ac circuit", "impedance"])
        or re.search(r"\b(?:x\s*[lc]|x[lc])\s*=", text)
    ):
        return None
    if not _frequency_transform_cue(text):
        return None
    target_quantities = {str(target.get("quantity") or "") for target in targets}
    target_text = " ".join(str(target.get("text") or "") for target in targets).lower()
    if _frequency_scale_factor_query(text, target_text):
        return "rlc_resonance_frequency_multiplier"
    if "reactance" in target_text or re.search(r"\b(?:x_l|xl|z_l|zl)\b", target_text):
        return "rlc_resonance_reactance_from_current_ratio"
    if "current" in target_quantities or "current" in target_text:
        return "rlc_current_from_rlcf_voltage"
    if "voltage" in target_quantities and any(cue in target_text for cue in ["resistor", "across r"]):
        return "rlc_resonance_resistor_voltage"
    if "resistance" in target_quantities or "impedance" in target_text:
        return "rlc_impedance"
    if "angle" in target_quantities or "phase" in target_text:
        return "rlc_phase_angle"
    return None


def _frequency_scale_factor_query(text: str, target_text: str) -> bool:
    combined = f"{text} {target_text}"
    return bool(
        any(cue in combined for cue in ["multiple", "multiplier", "factor", "scale", "ratio"])
        or re.search(r"\bby\s+what\s+factor\b", combined)
        or re.search(r"\bhow\s+many\s+times\b", combined)
        or any(cue in combined for cue in ["to achieve resonance", "to obtain resonance", "to resonate"])
    )


def _lc_energy_complement_formula_id(front_payload: dict, targets: list[dict]) -> str | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    target_text = " ".join(str(target.get("text") or "") for target in targets).lower()
    if "lc" not in text or "energy" not in text:
        return None
    has_energy_components = any(cue in text for cue in ["w_c", "wc", "w_l", "wl"])
    has_energy_components = has_energy_components or (
        any(cue in text for cue in ["electric energy", "electric field energy", "capacitor energy"])
        and any(cue in text for cue in ["magnetic energy", "magnetic field energy", "inductor energy"])
    )
    if not has_energy_components:
        return None
    if not any(cue in target_text for cue in ["electric", "magnetic", "w_c", "wc", "w_l", "wl", "energy"]):
        return None
    if any(cue in text for cue in ["cos", "sin", "t =", "t=", "period", "quarter", "maximum", "minimum", "zero"]):
        return "lc_energy_complement"
    return None


def _unoccupied_square_vertex_field_query(front_payload: dict) -> bool:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not (_has_geometry_cue(front_payload, "square") or "square" in text):
        return False
    vertex_cue = _unoccupied_square_vertex_cue(text)
    if not vertex_cue:
        return False
    counts = _dimension_counts(front_payload)
    return counts.get("charge", 0) >= 3


def _two_named_points_context(text: str) -> bool:
    return bool(re.search(r"\b(?:points?|vertices?)\s+[a-z]\s+and\s+[a-z]\b", text))


def _unoccupied_square_vertex_cue(text: str) -> bool:
    return bool(
        any(
            cue in text
            for cue in [
                "fourth vertex",
                "remaining vertex",
                "empty vertex",
                "unoccupied vertex",
                "missing vertex",
                "vertex without a charge",
                "vertex with no charge",
                "remaining corner",
                "empty corner",
                "unoccupied corner",
                "corner without a charge",
                "corner with no charge",
            ]
        )
        or re.search(r"\b(?:field|force)\s+at\s+the\s+(?:other|last)\s+(?:vertex|corner)\b", text)
        or re.search(r"\b(?:other|last)\s+(?:vertex|corner)\s+(?:of|in)\s+the\s+square\b", text)
    )


def _selected_spatial_formula_id(front_payload: dict, task_type: str, selected_formula_ids: list[str]) -> str | None:
    spatial_selected = [formula_id for formula_id in selected_formula_ids if _is_spatial_formula(formula_id)]
    if task_type == "coulomb_force":
        for preferred in (
            "coulomb_right_isosceles_identical_vertex",
            "coulomb_force_triangle_sides",
            "coulomb_force_direction_superposition",
            "symmetric_zero_force",
        ):
            if preferred in spatial_selected:
                return preferred
        return _symmetry_zero_formula_id(front_payload, "force") or (
            "coulomb_force_direction_superposition" if _direction_only_two_charge_geometry(front_payload) else None
        )
    if task_type == "electric_field_point":
        symmetry = _symmetry_zero_formula_id(front_payload, "electric_field")
        if symmetry:
            return symmetry
        text = str(front_payload.get("canonical_question") or "").lower()
        if _unoccupied_square_vertex_field_query(front_payload):
            return "electric_field_square_three_equal_vertex"
        if "equidistant" in text and any(cue in text for cue in ["separated", "distance between", "charges", "point charges"]):
            return "electric_field_two_charge_triangle_sides"
        for preferred in (
            "electric_field_zero_line_two_charges",
            "electric_field_two_charge_triangle_sides",
            "electric_field_equilateral_vertex",
            "electric_field_two_charge_superposition",
            "point_charge_field_midpoint_from_two_fields",
            "electric_field_square_three_equal_vertex",
            "electric_field_square_center_cancel_charge",
            "electric_field_symmetric_zero",
        ):
            if preferred in spatial_selected:
                return preferred
        if spatial_selected and front_payload.get("answer_type_hint") == "symbolic":
            return spatial_selected[0]
        return (
            "electric_field_symbolic_superposition" if _symbolic_vector_field_case(front_payload) else None
        )
    return None


def _spatial_template_id(front_payload: dict, formula_id: str | None) -> str:
    matches = match_geometry_templates(front_payload)
    if matches:
        return matches[0].template_id
    text = str(front_payload.get("canonical_question") or "").lower()
    if formula_id in {"coulomb_force_triangle_sides", "electric_field_two_charge_triangle_sides"}:
        if _has_geometry_cue(front_payload, "equilateral_triangle") or re.search(r"\b(?:equilateral|regular)\s+triangle\b", text):
            return "equilateral_triangle_vertex"
        return "triangle_sides"
    if formula_id in {"coulomb_force_direction_superposition", "electric_field_two_charge_superposition"}:
        return "triangle_sides"
    if formula_id in {"symmetric_zero_force", "electric_field_symmetric_zero"}:
        if _has_geometry_cue(front_payload, "square") or "square" in text:
            return "square_vertex_field"
        if _has_geometry_cue(front_payload, "equilateral_triangle") or re.search(r"\b(?:equilateral|regular)\s+triangle\b", text):
            return "equilateral_triangle_vertex"
    return "triangle_sides"


def _spatial_operation_for_formula(formula_id: str) -> str:
    if formula_id == "coulomb_force_triangle_sides":
        return "compute_pairwise_force"
    return "resolve_vector_components"


def _is_spatial_formula(formula_id: str) -> bool:
    return formula_id in FORMULA_REGISTRY and formula_execution_branch(formula_id) == "spatial_vector"


def _dimension_counts(front_payload: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for section in ("quantities", "symbolic_quantities"):
        for item in front_payload.get(section) or []:
            dimension = item.get("dimension")
            if dimension:
                counts[str(dimension)] = counts.get(str(dimension), 0) + 1
    structures = front_payload.get("canonical_structures") or {}
    component_counts = structures.get("component_counts") or {}
    for dimension, count in component_counts.items():
        if isinstance(count, int):
            counts[str(dimension)] = max(counts.get(str(dimension), 0), count)
    return counts


def _relation_qualifiers(front_payload: dict) -> set[str]:
    return {
        str(relation.get("qualifier") or "").lower()
        for relation in front_payload.get("relations") or []
        if isinstance(relation, dict) and relation.get("relation_type") == "geometry" and relation.get("qualifier")
    }


def _geometry_structures(front_payload: dict) -> dict:
    structures = front_payload.get("canonical_structures") or {}
    return structures.get("geometry") if isinstance(structures, dict) and isinstance(structures.get("geometry"), dict) else {}


def _geometry_cues(front_payload: dict) -> set[str]:
    cues = set(_relation_qualifiers(front_payload))
    geometry = _geometry_structures(front_payload)
    triangles = geometry.get("triangles") or []
    if triangles:
        cues.add("triangle")
    if any(isinstance(triangle, dict) and (triangle.get("right_angle_at") or triangle.get("canonical_right_angle_at")) for triangle in triangles):
        cues.add("right_triangle")
    if geometry.get("squares"):
        cues.add("square")
    return cues


def _has_geometry_cue(front_payload: dict, *names: str) -> bool:
    cues = _geometry_cues(front_payload)
    return any(name in cues for name in names)


def _has_structured_geometry(front_payload: dict) -> bool:
    if _relation_qualifiers(front_payload):
        return True
    geometry = _geometry_structures(front_payload)
    return bool(geometry.get("triangles") or geometry.get("squares"))


def _frequency_transform_cue(text: str) -> bool:
    return has_frequency_transform_cue(text)


def _first_target_id(targets: list[dict], fallback: str) -> str:
    if targets:
        return str(targets[0].get("id") or fallback)
    return fallback


def _unit_for_dimension(dimension: str | None) -> str | None:
    return {
        "capacitance": "F",
        "charge": "C",
        "current": "A",
        "dimensionless": "-",
        "electric_field": "V/m",
        "energy": "J",
        "force": "N",
        "frequency": "Hz",
        "inductance": "H",
        "impedance": "Ω",
        "length": "m",
        "magnetic_field": "T",
        "magnetic_flux": "Wb",
        "percent": "%",
        "power": "W",
        "resistance": "Ω",
        "time": "s",
        "voltage": "V",
    }.get(str(dimension or ""))


def _is_direct_scalar_formula(expression: str) -> bool:
    lowered = expression.lower()
    return "=" in expression and not any(cue in lowered for cue in ["deterministic", "vector", "|", "sum("])


def _canonical_topology_formula_id(front_payload: dict, task_type: str, targets: list[dict]) -> str | None:
    if task_type not in {"equivalent_resistance", "capacitance", "ohm_law"}:
        return None
    text = str(front_payload.get("canonical_question") or "").lower()
    if task_type == "ohm_law" and any(cue in text for cue in ["rlc", "resonance", "reactance", "impedance"]):
        return None
    topology = front_payload.get("topology_graph") or {}
    canonical = topology.get("canonical_form")
    if canonical not in {"series_topology", "parallel_topology"} or topology.get("ambiguity"):
        return None
    relation = "series" if canonical == "series_topology" else "parallel"
    target_quantities = {str(target.get("quantity") or "") for target in targets}
    if task_type == "equivalent_resistance" or "resistance" in target_quantities:
        candidate = f"{relation}_resistance_equivalent"
        return candidate if candidate in FORMULA_REGISTRY else None
    if task_type == "capacitance" or "capacitance" in target_quantities:
        candidate = f"{relation}_capacitance_equivalent"
        return candidate if candidate in FORMULA_REGISTRY else None
    if task_type == "ohm_law":
        if "current" in target_quantities:
            candidate = f"topology_ohm_current_{relation}_resistance"
            return candidate if candidate in FORMULA_REGISTRY else None
        if "voltage" in target_quantities:
            candidate = f"topology_ohm_voltage_{relation}_resistance"
            return candidate if candidate in FORMULA_REGISTRY else None
    return None
