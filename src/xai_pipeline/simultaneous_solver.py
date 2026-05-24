"""Whitelisted simultaneous multi-target solver."""

from __future__ import annotations

import re

from .worker_pool import get_default_sympy_pool


ALLOWED_SIMULTANEOUS_FAMILIES = {"linear_system"}


def solve_simultaneous_targets(plan: dict, givens: list[dict]) -> dict:
    targets = plan.get("targets", []) if isinstance(plan, dict) else []
    family = plan.get("simultaneous_family") if isinstance(plan, dict) else None
    if family not in ALLOWED_SIMULTANEOUS_FAMILIES:
        return _fail([f"simultaneous_family_not_whitelisted:{family}"], targets, givens)
    equations = list(plan.get("equations") or [])
    target_symbols = [str(target.get("symbol") if isinstance(target, dict) else target) for target in targets]
    if not equations or not target_symbols:
        return _fail(["missing_equations_or_targets"], targets, givens)
    safety_issues = _validate_equation_safety(equations, target_symbols)
    if safety_issues:
        return _fail(safety_issues, targets, givens)
    result = get_default_sympy_pool().solve(equations=equations, targets=target_symbols, timeout_seconds=float(plan.get("timeout_seconds", 3.0)))
    if not result.ok:
        return {
            "ok": False,
            "partial_results": [],
            "issues": result.issues,
            "trace": _trace(targets, givens, family, {"worker": result.trace}),
        }
    solutions = result.value or []
    if len(solutions) != 1:
        return {
            "ok": False,
            "partial_results": [],
            "issues": ["ambiguous_solution" if len(solutions) > 1 else "no_solution"],
            "trace": _trace(targets, givens, family, {"worker": result.trace, "solution_count": len(solutions)}),
        }
    solution = solutions[0]
    ordered = []
    for target in targets:
        symbol = str(target.get("symbol") if isinstance(target, dict) else target)
        if symbol not in solution:
            return {
                "ok": False,
                "partial_results": ordered,
                "issues": [f"target_missing_from_solution:{symbol}"],
                "trace": _trace(targets, givens, family, {"worker": result.trace}),
            }
        ordered.append(
            {
                "symbol": symbol,
                "value": solution[symbol],
                "unit": target.get("unit") if isinstance(target, dict) else None,
                "dimension": target.get("dimension", "dimensionless") if isinstance(target, dict) else "dimensionless",
            }
        )
    return {
        "ok": True,
        "partial_results": ordered,
        "issues": [],
        "trace": _trace(targets, givens, family, {"worker": result.trace}),
    }


def _fail(issues: list[str], targets: list, givens: list[dict]) -> dict:
    return {
        "ok": False,
        "partial_results": [],
        "issues": issues,
        "trace": _trace(targets, givens, None, {}),
    }


def _trace(targets: list, givens: list[dict], family: str | None, extra: dict) -> dict:
    trace = {
        "stage": "simultaneous_solver",
        "family": family,
        "target_count": len(targets),
        "given_count": len(givens),
    }
    trace.update(extra)
    return trace


def _validate_equation_safety(equations: list[str], targets: list[str]) -> list[str]:
    issues: list[str] = []
    allowed_symbols = set(targets) | {"sqrt", "sin", "cos", "tan", "pi"}
    if len(equations) > 4 or len(targets) > 4:
        issues.append("simultaneous_system_too_large")
    for index, equation in enumerate(equations):
        if not isinstance(equation, str) or equation.count("=") != 1:
            issues.append(f"invalid_equation_shape:{index}")
            continue
        if not re.fullmatch(r"[A-Za-z0-9_+\-*/().=^\s]+", equation):
            issues.append(f"unsafe_equation_tokens:{index}")
            continue
        names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", equation))
        unknown = sorted(name for name in names if name not in allowed_symbols)
        if unknown:
            issues.append(f"unknown_equation_symbol:{index}:{','.join(unknown)}")
    for symbol in targets:
        if not symbol.isidentifier():
            issues.append(f"invalid_target_symbol:{symbol}")
    return issues
