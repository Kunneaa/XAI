"""Bounded CAS-lite execution for registry-owned small equation systems."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any

from .equation_engine import SolverResult, _format, _safe_eval, _sanitize_equation, _symbols_in_expression
from ..knowledge.registries import FORMULA_REGISTRY


NONNEGATIVE_DIMENSIONS = {
    "area",
    "capacitance",
    "electric_field",
    "energy",
    "force",
    "frequency",
    "impedance",
    "inductance",
    "length",
    "magnetic_field",
    "magnetic_flux",
    "power",
    "resistance",
    "resistivity",
    "time",
    "turn_density",
    "angular_frequency",
}


def solve_algebraic_plan(plan: dict | None, route_result) -> SolverResult:
    """Solve a small registry equation graph without accepting free formulas.

    The plan must come from ``constraint_graph.build_registry_symbolic_plan``.
    Equations are already registry-owned and have known quantities substituted.
    This executor first performs topological explicit-assignment propagation,
    then falls back to a bounded one-variable numeric root search. It is not a
    general CAS and fails closed when the system is underdetermined or ambiguous.
    """

    if not plan:
        return _unsolved("no_algebraic_plan", getattr(route_result, "task_type", "unknown"))
    if plan.get("symbolic_family") != "registry_equation_graph":
        return _unsolved("unsupported_algebraic_family", getattr(route_result, "task_type", "unknown"), {"plan": plan})

    equations = [eq for eq in plan.get("equations", []) if isinstance(eq, str)]
    targets = [target for target in plan.get("targets", []) if isinstance(target, dict) and target.get("symbol")]
    if not equations or not targets:
        return _unsolved("algebraic_plan_missing_equations_or_targets", getattr(route_result, "task_type", "unknown"), {"plan": plan})

    target = targets[0]
    target_symbol = str(target["symbol"])
    target_dimension = target.get("dimension") or "dimensionless"
    target_unit = target.get("unit") or "-"
    sanitized = [_sanitize_equation(eq) for eq in equations]
    if any(eq is None for eq in sanitized):
        return _unsolved("algebraic_plan_has_unsafe_equation", getattr(route_result, "task_type", "unknown"), {"equations": equations})
    solved_values, propagation_trace = _propagate_explicit_assignments([str(eq) for eq in sanitized])
    if target_symbol not in solved_values:
        root = _solve_single_unknown([str(eq) for eq in sanitized], target_symbol, str(target_dimension), bool(plan.get("non_negative_target")))
        if not root["ok"]:
            return _unsolved(root["reason"], getattr(route_result, "task_type", "unknown"), {"plan_trace": plan.get("trace"), "algebraic": root})
        solved_values[target_symbol] = root["value"]
        propagation_trace.append(root["trace"])

    value = float(solved_values[target_symbol])
    domain_issue = _domain_issue(value, str(target_dimension), bool(plan.get("non_negative_target")))
    if domain_issue:
        return _unsolved(domain_issue, getattr(route_result, "task_type", "unknown"), {"target_symbol": target_symbol, "value": value})

    residuals = _residuals([str(eq) for eq in sanitized], solved_values)
    bad_residuals = [item for item in residuals if not item["ok"]]
    if bad_residuals:
        return _unsolved("algebraic_residual_check_failed", getattr(route_result, "task_type", "unknown"), {"residuals": residuals})

    formula_ids = list(plan.get("formula_ids") or [])
    target_formula_id = _target_formula_id(formula_ids, str(target_dimension))
    spec = FORMULA_REGISTRY.get(target_formula_id or "")
    principle_ids = list(plan.get("principle_ids") or [])
    answer = _format(value, str(target_unit))
    return SolverResult(
        solved=True,
        answer=answer,
        value=value,
        unit=str(target_unit),
        formula_id=target_formula_id,
        principle_id=(spec.principle_id if spec else (principle_ids[0] if principle_ids else None)),
        premises=[FORMULA_REGISTRY[fid].premise for fid in formula_ids if fid in FORMULA_REGISTRY],
        trace={
            "stage": "algebraic_constraint_engine",
            "formula_id": target_formula_id,
            "formula_ids": formula_ids,
            "principle_ids": principle_ids,
            "expression": spec.expression if spec else "; ".join(equations),
            "target_dimension": target_dimension,
            "target_symbol": target_symbol,
            "equations": equations,
            "solution": dict(solved_values),
            "value": value,
            "unit": target_unit,
            "residuals": residuals,
            "propagation_trace": propagation_trace,
            "plan_trace": plan.get("trace", {}),
            "binding_audit": {"policy": "registry_equation_graph", "equation_count": len(equations)},
        },
        confidence=min(0.82, float(getattr(route_result, "confidence", 0.7))),
    )


def _propagate_explicit_assignments(equations: list[str]) -> tuple[dict[str, float], list[dict]]:
    values: dict[str, float] = {}
    trace: list[dict] = []
    pending = list(equations)
    progressed = True
    while pending and progressed:
        progressed = False
        next_pending = []
        for equation in pending:
            lhs, rhs = equation.split("=", 1)
            lhs_symbol = lhs.strip()
            rhs_symbols = [symbol for symbol in _symbols_in_expression(rhs) if symbol not in _SAFE_NAMES]
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs_symbol):
                next_pending.append(equation)
                continue
            if any(symbol not in values for symbol in rhs_symbols):
                next_pending.append(equation)
                continue
            try:
                value = _safe_eval(rhs, values)
            except Exception:
                next_pending.append(equation)
                continue
            values[lhs_symbol] = float(value)
            trace.append({"mode": "explicit_assignment", "equation": equation, "symbol": lhs_symbol, "value": float(value)})
            progressed = True
        pending = next_pending
    return values, trace


def _solve_single_unknown(equations: list[str], target_symbol: str, target_dimension: str, force_nonnegative: bool) -> dict:
    if len(equations) != 1:
        return {"ok": False, "reason": "algebraic_system_not_explicit_or_single_unknown"}
    equation = equations[0]
    symbols = {symbol for symbol in _symbols_in_expression(equation) if symbol not in _SAFE_NAMES}
    if symbols != {target_symbol}:
        return {"ok": False, "reason": "algebraic_single_unknown_shape_unsupported", "symbols": sorted(symbols)}

    def residual(x: float) -> float:
        lhs, rhs = equation.split("=", 1)
        return _safe_eval(lhs, {target_symbol: x}) - _safe_eval(rhs, {target_symbol: x})

    lower_bound = 0.0 if force_nonnegative or target_dimension in NONNEGATIVE_DIMENSIONS else -1.0e9
    samples = _positive_samples() if lower_bound == 0.0 else _signed_samples()
    previous_x = None
    previous_y = None
    best = {"x": None, "abs": float("inf")}
    for x in samples:
        if x < lower_bound:
            continue
        try:
            y = residual(x)
        except Exception:
            continue
        if math.isfinite(y) and abs(y) < best["abs"]:
            best = {"x": x, "abs": abs(y)}
        if math.isfinite(y) and abs(y) <= 1e-9 * max(1.0, abs(x)):
            return {"ok": True, "value": x, "trace": {"mode": "sampled_root", "equation": equation, "residual": y}}
        if previous_x is not None and previous_y is not None and math.isfinite(y) and previous_y * y < 0:
            root = _bisect(residual, previous_x, x)
            return {"ok": True, "value": root, "trace": {"mode": "bisection_root", "equation": equation, "interval": [previous_x, x], "residual": residual(root)}}
        previous_x, previous_y = x, y
    if best["x"] is not None and best["abs"] <= 1e-7:
        return {"ok": True, "value": float(best["x"]), "trace": {"mode": "best_sample_root", "equation": equation, "residual": best["abs"]}}
    return {"ok": False, "reason": "algebraic_single_unknown_no_verified_root", "best_residual": best["abs"]}


def _bisect(fn, a: float, b: float, iterations: int = 80) -> float:
    fa = fn(a)
    for _ in range(iterations):
        mid = (a + b) / 2.0
        fm = fn(mid)
        if abs(fm) <= 1e-12:
            return mid
        if fa * fm <= 0:
            b = mid
        else:
            a = mid
            fa = fm
    return (a + b) / 2.0


def _residuals(equations: list[str], values: dict[str, float]) -> list[dict]:
    out = []
    for equation in equations:
        lhs, rhs = equation.split("=", 1)
        try:
            lhs_value = _safe_eval(lhs, values)
            rhs_value = _safe_eval(rhs, values)
            residual = lhs_value - rhs_value
            tolerance = 1e-7 * max(1.0, abs(lhs_value), abs(rhs_value))
            out.append({"equation": equation, "residual": residual, "tolerance": tolerance, "ok": abs(residual) <= tolerance})
        except Exception as exc:
            out.append({"equation": equation, "ok": False, "issue": f"residual_eval_failed:{type(exc).__name__}"})
    return out


def _target_formula_id(formula_ids: list[str], target_dimension: str) -> str | None:
    for formula_id in reversed(formula_ids):
        spec = FORMULA_REGISTRY.get(formula_id)
        if spec and spec.target_dimension == target_dimension:
            return formula_id
    return formula_ids[-1] if formula_ids else None


def _domain_issue(value: float, dimension: str, force_nonnegative: bool) -> str | None:
    if not math.isfinite(value):
        return "algebraic_non_finite_solution"
    if (force_nonnegative or dimension in NONNEGATIVE_DIMENSIONS) and value < -1e-12:
        return f"algebraic_negative_value_for_nonnegative_dimension:{dimension}"
    return None


def _positive_samples() -> list[float]:
    values = [0.0]
    for exponent in range(-15, 16):
        values.append(10.0**exponent)
    return values


def _signed_samples() -> list[float]:
    positives = _positive_samples()[1:]
    return [-value for value in reversed(positives)] + [0.0] + positives


def _unsolved(reason: str, task_type: str, extra: dict[str, Any] | None = None) -> SolverResult:
    trace = {"stage": "algebraic_constraint_engine", "reason": reason, "task_type": task_type}
    if extra:
        trace.update(extra)
    return SolverResult(False, "", None, None, None, None, [], trace, 0.0)


_SAFE_NAMES = {"sqrt", "sin", "cos", "tan", "atan", "exp", "log", "abs", "pi"}
