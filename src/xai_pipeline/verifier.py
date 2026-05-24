"""Deterministic verification for solver outputs and plans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from .confidence import apply_confidence_caps
from .implicit_kb import allowed_implicit_rule_ids
from .planner_schema import validate_planner_schema
from .registries import (
    ANSWER_TYPES,
    FORMULA_IDS,
    FORMULA_REGISTRY,
    GEOMETRY_TEMPLATE_IDS,
    PLANNER_STATUSES,
    PRINCIPLE_IDS,
    SOLVE_STRATEGIES,
    TASK_TYPES,
)


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    confidence: float
    issues: List[str]

    def to_dict(self):
        return {"ok": self.ok, "confidence": self.confidence, "issues": list(self.issues)}


def verify_solver(front_payload: dict, route_result, solver_result, registry_validation=None, unit_conversion=None) -> VerificationResult:
    issues: List[str] = []
    if registry_validation is not None and not registry_validation.ok:
        issues.extend(f"registry:{issue}" for issue in registry_validation.issues)
    if unit_conversion is not None and not unit_conversion.ok:
        issues.extend(f"unit_conversion:{issue}" for issue in unit_conversion.issues)
    if not solver_result.solved:
        issues.append("solver_not_solved")
        return VerificationResult(False, 0.0, issues)
    if solver_result.formula_id not in FORMULA_IDS:
        issues.append("unknown_formula_id")
    formula = FORMULA_REGISTRY.get(solver_result.formula_id or "")
    if formula and solver_result.formula_id != "multi_output_direct" and solver_result.unit != formula.target_unit:
        issues.append("target_unit_mismatch")
    answer_type = front_payload.get("answer_type_hint")
    is_conceptual_result = isinstance(solver_result.value, str) or solver_result.formula_id in {"conceptual_direct", "yes_no_direct"}
    if is_conceptual_result:
        if not isinstance(solver_result.value, str) or not solver_result.answer:
            issues.append("conceptual_answer_missing")
        if solver_result.principle_id not in PRINCIPLE_IDS:
            issues.append("conceptual_answer_without_principle")
        if answer_type == "yes_no" and solver_result.answer not in {"Yes", "No"}:
            issues.append("yes_no_answer_not_normalized")
    elif solver_result.value is None:
        issues.append("non_finite_value")
    elif isinstance(solver_result.value, list):
        for item in solver_result.value:
            if not math.isfinite(float(item.get("value", float("nan")))):
                issues.append("non_finite_value")
                break
    elif not math.isfinite(float(solver_result.value)):
        issues.append("non_finite_value")
    _verify_vector_trace_consistency(solver_result, issues)
    if not solver_result.answer:
        issues.append("answer_missing_unit")
    elif is_conceptual_result:
        pass
    elif isinstance(solver_result.value, list):
        for unit in [item.get("unit") for item in solver_result.value if item.get("unit") and item.get("unit") != "-"]:
            if unit not in solver_result.answer:
                issues.append("answer_missing_unit")
                break
    elif solver_result.unit not in solver_result.answer:
        issues.append("answer_missing_unit")
    if route_result.task_type == "unknown":
        issues.append("unknown_route")
    confidence = min(route_result.confidence, solver_result.confidence)
    cap_reasons = []
    if solver_result.trace.get("geometry", {}).get("recoverable"):
        cap_reasons.append("geometry_assumption")
    if issues:
        cap_reasons.append("verifier_failed")
    confidence, _ = apply_confidence_caps(confidence, cap_reasons)
    return VerificationResult(not issues, confidence, issues)


def _verify_vector_trace_consistency(solver_result, issues: List[str]) -> None:
    geometry_engine = solver_result.trace.get("geometry_engine") if hasattr(solver_result, "trace") else None
    if not isinstance(geometry_engine, dict):
        return
    vector = geometry_engine.get("vector") or geometry_engine.get("components")
    value = geometry_engine.get("value")
    if value is None and isinstance(vector, dict):
        value = vector.get("magnitude")
    if not isinstance(vector, dict) or value is None:
        return
    try:
        x = float(vector.get("x", 0.0))
        y = float(vector.get("y", 0.0))
        magnitude = float(vector.get("magnitude", math.hypot(x, y)))
        reported = float(value)
        solver_value = float(solver_result.value) if solver_result.value is not None and not isinstance(solver_result.value, (str, list)) else reported
    except (TypeError, ValueError):
        issues.append("vector_trace_not_numeric")
        return
    if not math.isclose(math.hypot(x, y), magnitude, rel_tol=1e-6, abs_tol=1e-9):
        issues.append("vector_component_magnitude_mismatch")
    if not math.isclose(reported, solver_value, rel_tol=1e-6, abs_tol=1e-9):
        issues.append("vector_trace_solver_value_mismatch")


def validate_plan(plan: dict, front_payload: dict) -> VerificationResult:
    issues: List[str] = []
    schema_result = validate_planner_schema(plan)
    if not schema_result.ok:
        issues.extend(f"schema:{issue}" for issue in schema_result.issues)
    allowed_implicit_ids = set(allowed_implicit_rule_ids())
    if plan.get("status") not in PLANNER_STATUSES:
        issues.append("unknown_status")
    if plan.get("task_type") not in TASK_TYPES:
        issues.append("unknown_task_type")
    if plan.get("answer_type") not in ANSWER_TYPES:
        issues.append("unknown_answer_type")
    if plan.get("solve_strategy") not in SOLVE_STRATEGIES:
        issues.append("unknown_solve_strategy")
    if not isinstance(plan.get("targets"), list) or not plan.get("targets"):
        issues.append("empty_targets")
    for formula_id in plan.get("formula_ids", []):
        if formula_id not in FORMULA_IDS:
            issues.append(f"unknown_formula_id:{formula_id}")
    for principle_id in plan.get("principle_ids", []):
        if principle_id not in PRINCIPLE_IDS:
            issues.append(f"unknown_principle_id:{principle_id}")
    for template_id in plan.get("geometry_template_ids", []):
        if template_id not in GEOMETRY_TEMPLATE_IDS:
            issues.append(f"unknown_geometry_template_id:{template_id}")
    for rule_id in plan.get("implicit_rule_ids", []):
        if rule_id not in allowed_implicit_ids:
            issues.append(f"unknown_implicit_rule_id:{rule_id}")
        elif front_payload and rule_id not in {fact.get("rule_id") for fact in front_payload.get("implicit_facts", [])}:
            issues.append(f"implicit_rule_not_triggered:{rule_id}")
    for target in plan.get("targets", []):
        if not isinstance(target, dict) or not target.get("symbol"):
            issues.append("invalid_target")
    if plan.get("formula_ids") and plan.get("task_type") in TASK_TYPES:
        task_type = plan.get("task_type")
        available_dimensions = [quantity.get("dimension") for quantity in (front_payload or {}).get("quantities", [])]
        for formula_id in plan.get("formula_ids", []):
            formula = FORMULA_REGISTRY.get(formula_id)
            if formula is not None and formula.task_type != task_type:
                issues.append(f"formula_task_mismatch:{formula_id}:{task_type}")
            if formula is not None and formula.required_dimensions:
                missing = _missing_required_dimensions(available_dimensions, formula.required_dimensions)
                if missing:
                    issues.append(f"formula_missing_required_dimensions:{formula_id}:{','.join(missing)}")
    if plan.get("numeric_answer") is not None:
        issues.append("planner_supplied_numeric_answer")
    if plan.get("answer_type") == "numeric" and plan.get("conceptual_answer"):
        issues.append("numeric_task_with_conceptual_answer")
    if plan.get("answer_type") in {"conceptual", "yes_no"} and plan.get("conceptual_answer") and not plan.get("principle_ids"):
        issues.append("conceptual_answer_without_principle")
    return VerificationResult(not issues, 0.8 if not issues else 0.0, issues)


def _missing_required_dimensions(available_dimensions: list[str | None], required_dimensions: tuple[str, ...]) -> list[str]:
    pool = list(available_dimensions)
    missing: list[str] = []
    for dimension in required_dimensions:
        if dimension not in pool:
            missing.append(dimension)
        else:
            pool.remove(dimension)
    return missing
