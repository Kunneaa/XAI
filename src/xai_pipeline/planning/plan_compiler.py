"""Validate and lower structured solve plans into executable dispatch hints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .answer_formats import output_format_issues
from .solve_plan import normalize_plan_payload, plan_summary
from ..engines.logic_engine import allowed_implicit_rule_ids
from ..knowledge.registries import (
    ANSWER_TYPES,
    FORMULA_IDS,
    FORMULA_REGISTRY,
    GEOMETRY_TEMPLATE_IDS,
    PLAN_OPERATION_IDS,
    PRINCIPLE_IDS,
    PROPOSAL_STATUSES,
    TASK_TYPES,
    formula_execution_branch,
)


DEFAULT_ENGINE_ORDER = ["logic", "fast_formula", "spatial", "algebraic"]
SPATIAL_ENGINE_ORDER = ["spatial", "logic", "fast_formula", "algebraic"]
MULTI_OUTPUT_ENGINE_ORDER = ["multi_output", "logic", "fast_formula", "spatial", "algebraic"]
ALGEBRAIC_ENGINE_ORDER = ["algebraic", "fast_formula", "logic", "spatial"]
FAST_FORMULA_FIRST_IDS = frozenset(
    {
        "rlc_current_from_rlcf_voltage",
    }
)

FORBIDDEN_PLAN_OUTPUTS = [
    "numeric_answers",
    "new_formulas",
    "new_constants",
    "new_units",
    "coordinates",
    "python_code",
    "free_form_cot",
]

FORBIDDEN_PAYLOAD_KEYS = {
    "answer",
    "final_answer",
    "numeric_answer",
    "coordinates",
    "coordinate",
    "python",
    "code",
}

FORBIDDEN_TOP_LEVEL_KEYS = {"formula", "constant", "unit"}

PUBLIC_COT_FORBIDDEN_SUBSTRINGS = (
    "\n",
    "```",
    "__import__",
    "lambda ",
    "final answer",
    "answer is",
    "therefore",
    "hence",
    "coordinate",
    "python",
    "code",
)
PUBLIC_NOTE_FORBIDDEN_SUBSTRINGS = PUBLIC_COT_FORBIDDEN_SUBSTRINGS + (
    "chain-of-thought",
    "chain of thought",
)


@dataclass(frozen=True)
class PlanValidation:
    ok: bool
    issues: List[str]
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "issues": list(self.issues), "audit": dict(self.audit)}


@dataclass(frozen=True)
class CompiledPlan:
    ok: bool
    plan: Dict[str, Any]
    selected_formula_ids: List[str]
    preferred_engine_order: List[str]
    issues: List[str]
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "plan": dict(self.plan),
            "selected_formula_ids": list(self.selected_formula_ids),
            "preferred_engine_order": list(self.preferred_engine_order),
            "issues": list(self.issues),
            "trace": dict(self.trace),
        }


def build_plan_error_packet(compiled_plan: CompiledPlan, front_payload: dict, route_result, graph_selection=None) -> dict:
    """Create the strict packet sent to the LLM for one re-plan attempt."""

    return {
        "stage": "plan_compiler",
        "error_code": "structured_solve_plan_invalid",
        "issues": list(compiled_plan.issues),
        "plan_summary": plan_summary(compiled_plan.plan),
        "route": route_result.to_dict() if hasattr(route_result, "to_dict") else {},
        "target_hints": list((front_payload or {}).get("target_hints") or [])[:6],
        "goals": list((front_payload or {}).get("goals") or [])[:6],
        "known_quantity_count": len((front_payload or {}).get("quantities") or []),
        "candidate_formula_ids": list(getattr(graph_selection, "formula_ids", []) or [])[:12],
        "allowed_operations": sorted(PLAN_OPERATION_IDS),
        "allowed_edits": [
            "targets",
            "assumptions",
            "steps",
            "step_dependencies",
            "step_inputs",
            "step_outputs",
            "registry_ids",
            "public_cot_labels",
        ],
        "forbidden_outputs": list(FORBIDDEN_PLAN_OUTPUTS),
    }


def compile_solve_plan(
    plan_payload: dict | None,
    front_payload: dict,
    route_result,
    graph_selection=None,
) -> CompiledPlan:
    """Validate a plan and convert it to deterministic dispatch metadata."""

    plan = normalize_plan_payload(plan_payload)
    if plan is None:
        return CompiledPlan(
            ok=False,
            plan={},
            selected_formula_ids=[],
            preferred_engine_order=list(DEFAULT_ENGINE_ORDER),
            issues=["plan_missing"],
            trace={"stage": "plan_compiler", "summary": {"present": False}},
        )
    validation = validate_structured_solve_plan(plan, front_payload, route_result)
    formula_ids = _selected_formula_ids(plan, graph_selection)
    engine_order = _preferred_engine_order(plan, front_payload, route_result)
    issues = list(validation.issues)
    if plan.get("status") != "ok":
        issues.append(f"plan_status:{plan.get('status')}")
    return CompiledPlan(
        ok=validation.ok and not issues,
        plan=plan,
        selected_formula_ids=formula_ids,
        preferred_engine_order=engine_order,
        issues=issues,
        trace={
            "stage": "plan_compiler",
            "validation": validation.to_dict(),
            "summary": plan_summary(plan),
            "engine_order": engine_order,
            "selected_formula_ids": formula_ids,
            "graph_formula_ids": list(getattr(graph_selection, "formula_ids", []) or []),
            "policy": "registry_backed_steps_only_no_raw_cot_execution",
        },
    )


def validate_structured_solve_plan(plan_payload: dict | None, front_payload: dict | None = None, route_result=None) -> PlanValidation:
    """Validate a public executable plan proposal.

    This validator is intentionally about structure and governance. It does not
    solve. It rejects unknown operations, unknown registry IDs, cycles, numeric
    answer leakage, and untriggered assumptions.
    """

    plan = normalize_plan_payload(plan_payload)
    if plan is None:
        return PlanValidation(False, ["plan_not_object"], {"stage": "structured_solve_plan_validator"})

    issues: list[str] = []
    for field_name in ("status", "task_type", "answer_type", "targets", "steps"):
        if field_name not in plan or plan.get(field_name) is None:
            issues.append(f"missing_field:{field_name}")
    if plan.get("status") not in PROPOSAL_STATUSES:
        issues.append("unknown_status")
    if plan.get("task_type") not in TASK_TYPES:
        issues.append("unknown_task_type")
    if plan.get("answer_type") not in ANSWER_TYPES:
        issues.append("unknown_answer_type")
    if route_result is not None and plan.get("task_type") != getattr(route_result, "task_type", plan.get("task_type")):
        if (
            getattr(route_result, "task_type", "unknown") != "unknown"
            and plan.get("task_type") != "unknown"
            and not _llm_route_override_is_valid(plan)
        ):
            issues.append(f"plan_route_task_mismatch:{plan.get('task_type')}:{getattr(route_result, 'task_type', None)}")

    targets = plan.get("targets")
    if not isinstance(targets, list) or not targets:
        issues.append("empty_targets")
    else:
        for target in targets:
            if not isinstance(target, dict) or not target.get("id"):
                issues.append("invalid_target")

    steps = plan.get("steps")
    if not isinstance(steps, list):
        issues.append("steps_not_list")
        steps = []
    elif plan.get("status") == "ok" and not steps:
        issues.append("empty_steps")
    notes = plan.get("notes") or []
    if notes and not isinstance(notes, list):
        issues.append("notes_not_list")
        notes = []
    for index, note in enumerate(notes[:8]):
        note_issue = _public_note_issue(index, note)
        if note_issue:
            issues.append(note_issue)

    step_ids: set[str] = set()
    outputs: set[str] = set()
    allowed_implicit = set(allowed_implicit_rule_ids())
    triggered_implicit = {fact.get("rule_id") for fact in (front_payload or {}).get("implicit_facts", [])}

    for assumption in plan.get("assumptions") or []:
        if not isinstance(assumption, dict):
            issues.append("invalid_assumption")
            continue
        assumption_id = assumption.get("assumption_id")
        if assumption_id in allowed_implicit and assumption_id not in triggered_implicit:
            issues.append(f"implicit_rule_not_triggered:{assumption_id}")
        if assumption_id and assumption_id not in allowed_implicit and not str(assumption_id).startswith("universal_constant"):
            issues.append(f"unknown_assumption_id:{assumption_id}")

    for step in steps:
        if not isinstance(step, dict):
            issues.append("invalid_step")
            continue
        if _contains_forbidden_payload(step):
            issues.append(f"forbidden_payload_in_step:{step.get('step_id')}")
        step_id = step.get("step_id")
        if not step_id or not isinstance(step_id, str):
            issues.append("invalid_step_id")
            continue
        public_cot_issue = _public_cot_issue(step_id, step.get("public_cot"))
        if public_cot_issue:
            issues.append(public_cot_issue)
        if step_id in step_ids:
            issues.append(f"duplicate_step_id:{step_id}")
        step_ids.add(step_id)
        operation = step.get("operation")
        if operation not in PLAN_OPERATION_IDS:
            issues.append(f"unknown_operation:{operation}")
        formula_id = step.get("formula_id")
        if formula_id and formula_id not in FORMULA_IDS:
            issues.append(f"unknown_formula_id:{formula_id}")
        principle_id = step.get("principle_id")
        if principle_id and principle_id not in PRINCIPLE_IDS:
            issues.append(f"unknown_principle_id:{principle_id}")
        if formula_id in FORMULA_IDS:
            formula = FORMULA_REGISTRY.get(formula_id)
            if formula and plan.get("task_type") not in {"unknown", formula.task_type} and not _formula_task_compatible(
                formula_id, str(plan.get("task_type") or "")
            ):
                issues.append(f"formula_task_mismatch:{formula_id}:{formula.task_type}:{plan.get('task_type')}")
        if formula_id and principle_id:
            formula = FORMULA_REGISTRY.get(formula_id)
            if formula and formula.principle_id != principle_id:
                issues.append(f"formula_principle_mismatch:{formula_id}:{principle_id}")
        geometry_id = step.get("geometry_constructor_id")
        if geometry_id and geometry_id not in GEOMETRY_TEMPLATE_IDS:
            issues.append(f"unknown_geometry_template_id:{geometry_id}")
        if operation == "construct_geometry" and not geometry_id:
            issues.append(f"construct_geometry_missing_template:{step_id}")
        if operation in {"apply_formula", "compute_pairwise_force", "resolve_vector_components"} and not formula_id:
            issues.append(f"formula_operation_missing_formula_id:{step_id}")
        if operation == "apply_logic_rule" and not (principle_id or step.get("logic_rule_id")):
            issues.append(f"logic_step_missing_rule_or_principle:{step_id}")
        depends_on = step.get("depends_on") or []
        if not isinstance(depends_on, list):
            issues.append(f"depends_on_not_list:{step_id}")
        output = step.get("output")
        if output:
            if output in outputs:
                issues.append(f"duplicate_step_output:{output}")
            outputs.add(str(output))

    issues.extend(_dependency_issues(steps))
    if _contains_forbidden_payload(plan, top_level=True):
        issues.append("forbidden_top_level_payload")
    issues.extend(output_format_issues(plan))

    audit = {
        "stage": "structured_solve_plan_validator",
        "step_count": len(steps),
        "operation_ids": sorted({step.get("operation") for step in steps if isinstance(step, dict) and step.get("operation")}),
        "formula_ids": [step.get("formula_id") for step in steps if isinstance(step, dict) and step.get("formula_id")],
        "principle_ids": [step.get("principle_id") for step in steps if isinstance(step, dict) and step.get("principle_id")],
        "geometry_template_ids": [
            step.get("geometry_constructor_id") for step in steps if isinstance(step, dict) and step.get("geometry_constructor_id")
        ],
        "note_count": len(notes),
        "assumption_ids": [
            assumption.get("assumption_id") for assumption in plan.get("assumptions") or [] if isinstance(assumption, dict) and assumption.get("assumption_id")
        ],
        "output_format_kind": (plan.get("output_format") or {}).get("format_kind") if isinstance(plan.get("output_format"), dict) else None,
        "forbidden_payload_policy": "no_numeric_answers_no_code_no_coordinates_no_free_form_cot",
    }
    return PlanValidation(not issues, issues, audit)


def _formula_task_compatible(formula_id: str, plan_task_type: str) -> bool:
    """Allow registry-family formulas to serve neighboring route labels.

    Routes are coarse semantic intents, while executable formulas are normalized
    registry operators. RLC frequency transforms are a good example: a question
    may be routed as a reactance/current/resonance task, but the deterministic
    operator belongs to the shared ``rlc_core`` formula family.
    """

    formula = FORMULA_REGISTRY.get(formula_id)
    if formula is None:
        return False
    if formula.principle_id == "rlc_core" and plan_task_type in {
        "ohm_law",
        "electric_power",
        "inductive_reactance",
        "capacitive_reactance",
        "rlc_impedance",
        "power_factor",
    }:
        return True
    if formula.principle_id == "magnetic_core" and plan_task_type in {
        "magnetic_flux",
        "magnetic_field",
        "solenoid_magnetic_field",
        "inductor_energy",
        "turn_density",
    }:
        return True
    return False


def _llm_route_override_is_valid(plan: dict) -> bool:
    """Allow a local-LLM plan to override coarse routing only via valid formulas."""

    if plan.get("source") not in {"local_llm", "local_llm_repair"}:
        return False
    task_type = str(plan.get("task_type") or "")
    formula_ids = [
        str(step.get("formula_id"))
        for step in plan.get("steps") or []
        if isinstance(step, dict) and step.get("formula_id")
    ]
    if not formula_ids:
        return False
    for formula_id in formula_ids:
        formula = FORMULA_REGISTRY.get(formula_id)
        if formula is None:
            return False
        if formula.task_type != task_type and not _formula_task_compatible(formula_id, task_type):
            return False
    return True


_NUMERIC_UNIT_LEAK_RE = re.compile(
    r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*(?:N/C|V/m|ohm|Ω|N|A|V|J|F|C|W|Hz|H|T|Wb|m|s|%)\b",
    re.IGNORECASE,
)


def _public_cot_issue(step_id: str, value: Any) -> str | None:
    """Validate the public CoT/action label attached to one plan step."""

    if value is None:
        return f"missing_public_cot:{step_id}"
    if not isinstance(value, str):
        return f"invalid_public_cot:{step_id}:not_string"
    text = value.strip()
    if not text:
        return f"invalid_public_cot:{step_id}:empty"
    if len(text) > 220:
        return f"invalid_public_cot:{step_id}:too_long"
    lowered = text.lower()
    if any(cue in lowered for cue in PUBLIC_COT_FORBIDDEN_SUBSTRINGS):
        return f"invalid_public_cot:{step_id}:forbidden_text"
    if "=" in text:
        return f"invalid_public_cot:{step_id}:equation_text"
    if _NUMERIC_UNIT_LEAK_RE.search(text):
        return f"invalid_public_cot:{step_id}:numeric_result_leak"
    return None


def _public_note_issue(index: int, value: Any) -> str | None:
    """Validate optional model notes as metadata, not reasoning or answers."""

    if not isinstance(value, str):
        return f"invalid_note:{index}:not_string"
    text = value.strip()
    if not text:
        return f"invalid_note:{index}:empty"
    if len(text) > 180:
        return f"invalid_note:{index}:too_long"
    lowered = text.lower()
    if any(cue in lowered for cue in PUBLIC_NOTE_FORBIDDEN_SUBSTRINGS):
        return f"invalid_note:{index}:forbidden_text"
    if "=" in text:
        return f"invalid_note:{index}:equation_text"
    if _NUMERIC_UNIT_LEAK_RE.search(text):
        return f"invalid_note:{index}:numeric_result_leak"
    return None


def plan_prefers_spatial(compiled_plan: CompiledPlan | None) -> bool:
    if compiled_plan is None:
        return False
    return "spatial" in compiled_plan.preferred_engine_order[:1]


def _selected_formula_ids(plan: dict, graph_selection=None) -> list[str]:
    ordered: list[str] = []
    for step in plan.get("steps") or []:
        if isinstance(step, dict) and step.get("formula_id") in FORMULA_IDS and step["formula_id"] not in ordered:
            ordered.append(step["formula_id"])
    if plan.get("source") in {"local_llm", "local_llm_repair"}:
        return ordered
    for formula_id in list(getattr(graph_selection, "formula_ids", []) or []):
        if formula_id in FORMULA_IDS and formula_id not in ordered:
            ordered.append(formula_id)
    return ordered


def _preferred_engine_order(plan: dict, front_payload: dict, route_result) -> list[str]:
    operations = {step.get("operation") for step in plan.get("steps") or [] if isinstance(step, dict)}
    formula_ids = [step.get("formula_id") for step in plan.get("steps") or [] if isinstance(step, dict) and step.get("formula_id")]
    task_type = plan.get("task_type") if plan.get("source") in {"local_llm", "local_llm_repair"} else getattr(route_result, "task_type", plan.get("task_type"))
    if task_type == "multi_output" or plan.get("answer_type") == "multi_output":
        return list(MULTI_OUTPUT_ENGINE_ORDER)
    if any(formula_id in FORMULA_IDS and formula_execution_branch(str(formula_id)) == "spatial_vector" for formula_id in formula_ids):
        return list(SPATIAL_ENGINE_ORDER)
    if any(str(formula_id) in FAST_FORMULA_FIRST_IDS for formula_id in formula_ids):
        return list(DEFAULT_ENGINE_ORDER)
    if any(formula_id in FORMULA_IDS and formula_execution_branch(str(formula_id)) == "algebraic_system" for formula_id in formula_ids):
        return list(ALGEBRAIC_ENGINE_ORDER)
    if operations & {"construct_geometry", "compute_pairwise_force", "resolve_vector_components", "vector_sum"}:
        return list(SPATIAL_ENGINE_ORDER)
    if operations & {"apply_logic_rule", "check_condition"} or task_type == "conceptual":
        return list(DEFAULT_ENGINE_ORDER)
    if operations == {"solve_equation_subset"}:
        return list(ALGEBRAIC_ENGINE_ORDER)
    return list(DEFAULT_ENGINE_ORDER)


def _dependency_issues(steps: list[Any]) -> list[str]:
    issues: list[str] = []
    graph: dict[str, list[str]] = {}
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("step_id"), str):
            continue
        depends_on = step.get("depends_on") or []
        if not isinstance(depends_on, list):
            continue
        graph[step["step_id"]] = [str(item) for item in depends_on]
    known = set(graph)
    for step_id, deps in graph.items():
        for dep in deps:
            if dep not in known:
                issues.append(f"unknown_step_dependency:{step_id}:{dep}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            issues.append(f"cyclic_step_dependency:{node}")
            return
        visiting.add(node)
        for dep in graph.get(node, []):
            if dep in graph:
                visit(dep)
        visiting.remove(node)
        visited.add(node)

    for step_id in list(graph):
        visit(step_id)
    return issues


def _contains_forbidden_payload(payload: Any, top_level: bool = False) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_PAYLOAD_KEYS:
                return True
            if top_level and lowered in FORBIDDEN_TOP_LEVEL_KEYS:
                return True
            if _contains_forbidden_payload(value):
                return True
    elif isinstance(payload, list):
        return any(_contains_forbidden_payload(item) for item in payload)
    elif isinstance(payload, str):
        lowered = payload.lower()
        if "final answer" in lowered or "```" in lowered or "__import__" in lowered:
            return True
    return False
