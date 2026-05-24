"""Registry-owned symbolic expression executor."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .root_filter import filter_roots
from .worker_pool import get_default_sympy_pool
from .numerical_solver import solve_numerically_bounded


ALLOWED_SYMBOLIC_FAMILIES = {
    "linear_single_equation",
    "quadratic_single_equation",
    "registry_single_equation",
    "registry_equation_graph",
}


@dataclass(frozen=True)
class SymbolicExecutionResult:
    ok: bool
    value: object
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {"ok": self.ok, "value": self.value, "issues": list(self.issues), "trace": dict(self.trace)}


def execute_symbolic_expression(plan: dict) -> dict:
    """Solve only schema-provided registry-safe symbolic equations.

    Accepted plan shape:
    {
      "symbolic_family": "linear_single_equation",
      "equations": ["x+1=2"],
      "targets": [{"symbol": "x", "dimension": "dimensionless"}]
    }
    """

    if not isinstance(plan, dict):
        return _fail(["symbolic_plan_not_object"], plan)
    family = plan.get("symbolic_family")
    if family not in ALLOWED_SYMBOLIC_FAMILIES:
        return _fail([f"symbolic_family_not_whitelisted:{family}"], plan)
    equations = list(plan.get("equations") or [])
    targets = list(plan.get("targets") or [])
    if not equations or not targets:
        return _fail(["missing_symbolic_equations_or_targets"], plan)
    target_symbol = str(targets[0].get("symbol") if isinstance(targets[0], dict) else targets[0])
    target_dimension = str(targets[0].get("dimension", "dimensionless")) if isinstance(targets[0], dict) else "dimensionless"
    solve_targets = [str(target.get("symbol") if isinstance(target, dict) else target) for target in targets]
    worker = get_default_sympy_pool()
    result = worker.solve(equations=equations, targets=solve_targets, timeout_seconds=float(plan.get("timeout_seconds", 3.0)))
    if not result.ok:
        numerical = _try_numerical_fallback(plan, result.issues)
        if numerical is not None and numerical.get("ok"):
            return numerical
        return {
            "ok": False,
            "answer": None,
            "issues": result.issues + ([] if numerical is None else numerical.get("issues", [])),
            "trace": {"stage": "symbolic_expression_executor", "family": family, "worker": result.trace, "numerical_fallback": numerical},
        }
    roots = []
    for solution in result.value or []:
        if target_symbol in solution:
            try:
                roots.append(float(solution[target_symbol]))
            except (TypeError, ValueError):
                pass
    filtered = filter_roots(
        roots,
        target_dimension=target_dimension,
        elapsed_time=bool(plan.get("elapsed_time", False)),
        non_negative=bool(plan.get("non_negative_target", False)),
    )
    if filtered["issues"]:
        numerical = _try_numerical_fallback(plan, filtered["issues"])
        if numerical is not None and numerical.get("ok"):
            return numerical
        return {
            "ok": False,
            "answer": None,
            "issues": filtered["issues"] + ([] if numerical is None else numerical.get("issues", [])),
            "trace": {"stage": "symbolic_expression_executor", "family": family, "worker": result.trace, "root_filter": filtered, "numerical_fallback": numerical},
        }
    value = filtered["valid_roots"][0]
    return {
        "ok": True,
        "answer": str(value),
        "value": value,
        "unit": targets[0].get("unit") if isinstance(targets[0], dict) else None,
        "target_dimension": target_dimension,
        "issues": [],
        "trace": {"stage": "symbolic_expression_executor", "family": family, "worker": result.trace, "root_filter": filtered},
    }


def _fail(issues: list[str], plan) -> dict:
    return {
        "ok": False,
        "answer": None,
        "issues": issues,
        "trace": {
            "stage": "symbolic_expression_executor",
            "formula_ids": list(plan.get("formula_ids", [])) if isinstance(plan, dict) else [],
        },
    }


def _try_numerical_fallback(plan: dict, symbolic_issues: list[str]) -> dict | None:
    family_id = plan.get("numerical_family_id")
    bounds = plan.get("bounds")
    if not family_id:
        auto = _try_registry_scalar_root_fallback(plan, symbolic_issues)
        if auto is not None:
            return auto
        return {
            "ok": False,
            "answer": None,
            "issues": ["numerical_fallback_not_applicable"],
            "trace": {"stage": "numerical_fallback", "symbolic_issues": symbolic_issues, "reason": "missing_numerical_family_id"},
        }
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return {
            "ok": False,
            "answer": None,
            "issues": [*symbolic_issues, "numerical_fallback_missing_bounds"],
            "trace": {"stage": "symbolic_expression_executor", "symbolic_issues": symbolic_issues},
        }
    numerical = solve_numerically_bounded(
        family_id=str(family_id),
        bounds=(float(bounds[0]), float(bounds[1])),
        timeout_seconds=float(plan.get("timeout_seconds", 3.0)),
        parameters=dict(plan.get("parameters") or {}),
    )
    if not numerical.ok or numerical.value is None:
        return {
            "ok": False,
            "answer": None,
            "issues": [*symbolic_issues, *numerical.issues],
            "trace": {"stage": "symbolic_expression_executor", "symbolic_issues": symbolic_issues, "numerical_fallback": numerical.to_dict()},
        }
    targets = list(plan.get("targets") or [])
    target = targets[0] if targets and isinstance(targets[0], dict) else {}
    return {
        "ok": True,
        "answer": str(numerical.value),
        "value": numerical.value,
        "unit": target.get("unit"),
        "target_dimension": target.get("dimension", "dimensionless"),
        "issues": [],
        "trace": {
            "stage": "symbolic_expression_executor",
            "symbolic_issues": symbolic_issues,
            "numerical_fallback": numerical.to_dict(),
            "confidence_cap": "numerical_fallback_used",
        },
    }


def _try_registry_scalar_root_fallback(plan: dict, symbolic_issues: list[str]) -> dict | None:
    """Fallback for one-unknown registry equations when SymPy is unavailable.

    This does not evaluate arbitrary model equations: the caller supplies
    sanitized registry-owned equations with all non-target symbols already
    substituted. The root is then substitution-verified by the numerical
    solver's residual check.
    """

    if plan.get("symbolic_family") not in {"registry_single_equation", "registry_equation_graph"}:
        return None
    equations = list(plan.get("equations") or [])
    targets = list(plan.get("targets") or [])
    if len(equations) != 1 or len(targets) != 1 or not isinstance(targets[0], dict):
        return None
    target = targets[0]
    symbol = str(target.get("symbol") or "")
    if not symbol or not symbol.isidentifier():
        return None
    function = _safe_equation_function(equations[0], symbol)
    if function is None:
        return None
    bounds = _default_scalar_bounds(str(target.get("dimension", "dimensionless")), bool(plan.get("non_negative_target", False)))
    numerical = solve_numerically_bounded(
        family_id="scalar_root",
        bounds=bounds,
        timeout_seconds=float(plan.get("timeout_seconds", 3.0)),
        function=function,
        parameters={},
    )
    if not numerical.ok or numerical.value is None:
        return {
            "ok": False,
            "answer": None,
            "issues": [*symbolic_issues, *numerical.issues],
            "trace": {"stage": "symbolic_expression_executor", "symbolic_issues": symbolic_issues, "auto_numerical_fallback": numerical.to_dict()},
        }
    return {
        "ok": True,
        "answer": str(numerical.value),
        "value": numerical.value,
        "unit": target.get("unit"),
        "target_dimension": target.get("dimension", "dimensionless"),
        "issues": [],
        "trace": {
            "stage": "symbolic_expression_executor",
            "symbolic_issues": symbolic_issues,
            "auto_numerical_fallback": numerical.to_dict(),
            "confidence_cap": "numerical_fallback_used",
        },
    }


def _safe_equation_function(equation: str, symbol: str):
    if "=" not in equation:
        return None
    lhs, rhs = equation.split("=", 1)
    expression = f"({lhs})-({rhs})"
    names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))
    allowed = {symbol, "sqrt", "sin", "cos", "tan", "pi"}
    if not names <= allowed:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().=\s]+", equation):
        return None
    code = compile(expression.replace("^", "**"), "<registry_scalar_root>", "eval")

    def fn(value: float) -> float:
        local = {
            symbol: float(value),
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
        }
        return float(eval(code, {"__builtins__": {}}, local))

    return fn


def _default_scalar_bounds(target_dimension: str, non_negative: bool) -> tuple[float, float]:
    if target_dimension in {"capacitance", "current", "electric_field", "energy", "force", "frequency", "inductance", "length", "magnetic_field", "power", "resistance", "time"} or non_negative:
        return (0.0, 1e12)
    return (-1e12, 1e12)
