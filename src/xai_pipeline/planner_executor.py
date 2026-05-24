"""Execute a validated Qwen plan through deterministic code only."""

from __future__ import annotations

from .executor import execute_deterministic
from .router import RouteResult


def execute_validated_plan(front_payload: dict, planner_result, unit_conversion_result):
    """Use Qwen's validated route proposal, then solve deterministically.

    Qwen never supplies equations or numeric answers here. Its plan can only
    select whitelisted IDs that have already passed ``validate_plan``.
    """

    plan = planner_result.plan if planner_result is not None else None
    validation = planner_result.validation if planner_result is not None else {}
    if not isinstance(plan, dict):
        return None
    if not validation.get("ok"):
        return None
    if plan.get("status") != "ok":
        return None
    if plan.get("numeric_answer") is not None:
        return None
    route_result = RouteResult(
        task_type=plan["task_type"],
        answer_type=plan["answer_type"],
        confidence=min(float(plan.get("confidence", 0.0)), 0.82),
        reasons=["validated_qwen_plan_route"],
    )
    solver_result = execute_deterministic(front_payload, route_result, unit_conversion_result)
    solver_result.trace["planner_execution"] = {
        "stage": "planner_executor",
        "used_validated_plan": True,
        "formula_ids": list(plan.get("formula_ids", [])),
        "principle_ids": list(plan.get("principle_ids", [])),
        "geometry_template_ids": list(plan.get("geometry_template_ids", [])),
    }
    return route_result, solver_result
