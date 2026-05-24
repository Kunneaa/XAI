"""Deterministic executor dispatcher."""

from __future__ import annotations

from .executor_modes import select_executor_mode
from .geometry import geometry_recoverability
from .principles import build_registry_symbolic_plan, select_minimal_equation_subset
from .registries import FORMULA_REGISTRY
from .simultaneous_solver import solve_simultaneous_targets
from .solver import SolverResult, solve_fast
from .symbolic_executor import execute_symbolic_expression


def execute_deterministic(front_payload: dict, route_result, unit_conversion_result):
    executor_mode = select_executor_mode(front_payload, route_result)
    principle_selection = select_minimal_equation_subset(front_payload, route_result)
    geometry_trace = geometry_recoverability(front_payload)
    solver_result = _dispatch_by_mode(front_payload, route_result, executor_mode, principle_selection)
    solver_result.trace["executor_mode"] = executor_mode
    solver_result.trace["executor_stage"] = "deterministic_executor"
    solver_result.trace["unit_conversion_ok"] = unit_conversion_result.ok
    solver_result.trace["principle_selection"] = principle_selection.to_dict()
    solver_result.trace["geometry"] = geometry_trace
    return solver_result


def _dispatch_by_mode(front_payload: dict, route_result, executor_mode: dict, principle_selection=None):
    mode = executor_mode.get("mode")
    if mode == "multi_output":
        simultaneous_plan = front_payload.get("simultaneous_plan")
        if simultaneous_plan:
            simultaneous = solve_simultaneous_targets(simultaneous_plan, front_payload.get("quantities", []))
            if simultaneous.get("ok"):
                return _simultaneous_to_solver_result(simultaneous, route_result)
            fallback = _solve_fast_then_principle_graph(front_payload, route_result, principle_selection)
            fallback.trace["simultaneous_solver"] = simultaneous
            return fallback
    if mode == "symbolic_expression":
        symbolic_plan = front_payload.get("symbolic_plan")
        if symbolic_plan:
            symbolic = execute_symbolic_expression(symbolic_plan)
            if symbolic.get("ok") and _symbolic_plan_is_route_owned(symbolic_plan, route_result):
                return _symbolic_to_solver_result(symbolic, symbolic_plan, route_result)
            fallback = _solve_fast_then_principle_graph(front_payload, route_result, principle_selection)
            fallback.trace["symbolic_expression_executor"] = symbolic
            return fallback
    return _solve_fast_then_principle_graph(front_payload, route_result, principle_selection)


def _solve_fast_then_principle_graph(front_payload: dict, route_result, principle_selection=None):
    fast = solve_fast(front_payload, route_result)
    if fast.solved:
        return fast
    symbolic_plan = build_registry_symbolic_plan(front_payload, route_result, principle_selection) if principle_selection is not None else None
    if symbolic_plan is None:
        return fast
    symbolic = execute_symbolic_expression(symbolic_plan)
    if symbolic.get("ok") and _symbolic_plan_is_route_owned(symbolic_plan, route_result):
        result = _symbolic_to_solver_result(symbolic, symbolic_plan, route_result)
        result.trace["principle_graph_fallback"] = {
            "used": True,
            "previous_solver_reason": fast.trace.get("reason"),
            "plan_trace": symbolic_plan.get("trace"),
        }
        return result
    fast.trace["principle_graph_fallback"] = {
        "used": False,
        "plan_trace": symbolic_plan.get("trace"),
        "symbolic_result": symbolic,
    }
    return fast


def _simultaneous_to_solver_result(simultaneous: dict, route_result) -> SolverResult:
    outputs = [
        {
            "name": item.get("symbol"),
            "value": float(item.get("value")),
            "unit": item.get("unit") or "-",
            "dimension": item.get("dimension", "dimensionless"),
        }
        for item in simultaneous.get("partial_results", [])
    ]
    answer = "; ".join(f"{item['name']}={item['value']:.6g} {item['unit']}".strip() for item in outputs)
    return SolverResult(
        solved=True,
        answer=answer,
        value=outputs,
        unit=";".join(item["unit"] for item in outputs),
        formula_id="multi_output_direct",
        principle_id="conceptual_core",
        premises=["A whitelisted simultaneous equation family solved the coupled targets."],
        trace={
            "stage": "simultaneous_executor",
            "expression": "whitelisted simultaneous family",
            "target_dimension": "multi_output",
            "simultaneous_solver": simultaneous,
        },
        confidence=min(0.7, route_result.confidence),
    )


def _symbolic_plan_is_route_owned(symbolic_plan: dict, route_result) -> bool:
    formula_ids = symbolic_plan.get("formula_ids") or []
    if not formula_ids:
        return False
    formula = FORMULA_REGISTRY.get(formula_ids[0])
    return formula is not None and formula.task_type == route_result.task_type


def _symbolic_to_solver_result(symbolic: dict, symbolic_plan: dict, route_result) -> SolverResult:
    value = symbolic.get("value")
    formula = FORMULA_REGISTRY[symbolic_plan["formula_ids"][0]]
    unit = symbolic.get("unit") or formula.target_unit
    answer = symbolic.get("answer") or (f"{value} {unit}".strip() if value is not None else "")
    if unit != "-" and unit not in answer:
        answer = f"{answer} {unit}".strip()
    return SolverResult(
        solved=True,
        answer=answer,
        value=value,
        unit=unit,
        formula_id=formula.formula_id,
        principle_id=formula.principle_id,
        premises=["A whitelisted symbolic executor solved the registry-owned equation family."],
        trace={
            "stage": "symbolic_expression_executor",
            "expression": "registry-owned symbolic family",
            "target_dimension": symbolic.get("target_dimension", formula.target_dimension),
            "symbolic_expression_executor": symbolic,
        },
        confidence=min(0.72, route_result.confidence),
    )
