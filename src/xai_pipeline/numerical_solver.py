"""Bounded numerical fallback boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass


NUMERICAL_FAMILIES = {
    "scalar_root": "generic bounded scalar root with explicit callable",
    "rc_charge_fraction": "solve 1-exp(-t/tau)=fraction",
    "rc_discharge_fraction": "solve exp(-t/tau)=fraction",
}


@dataclass(frozen=True)
class NumericalSolveResult:
    ok: bool
    value: float | None
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {"ok": self.ok, "value": self.value, "issues": list(self.issues), "trace": dict(self.trace)}


def solve_numerically_bounded(
    *,
    family_id: str,
    bounds: tuple[float, float] | None,
    timeout_seconds: float = 3.0,
    function=None,
    parameters: dict | None = None,
) -> NumericalSolveResult:
    if family_id not in NUMERICAL_FAMILIES:
        return NumericalSolveResult(
            False,
            None,
            [f"numerical_family_not_whitelisted:{family_id}"],
            {"stage": "numerical_fallback", "family_id": family_id, "timeout_seconds": timeout_seconds},
        )
    if bounds is None:
        return NumericalSolveResult(
            False,
            None,
            ["missing_bounds"],
            {"stage": "numerical_fallback", "family_id": family_id, "timeout_seconds": timeout_seconds},
        )
    lo, hi = bounds
    if not (math.isfinite(float(lo)) and math.isfinite(float(hi))) or not lo < hi:
        return NumericalSolveResult(False, None, ["invalid_bounds"], {"stage": "numerical_fallback", "family_id": family_id, "bounds": bounds})
    fn = function or _family_function(family_id, parameters or {})
    if fn is None:
        return NumericalSolveResult(False, None, ["missing_numerical_function"], {"stage": "numerical_fallback", "family_id": family_id, "bounds": bounds})
    try:
        from scipy.optimize import brentq

        value = float(brentq(fn, float(lo), float(hi), maxiter=128, xtol=1e-12))
        residual = float(fn(value))
        if not math.isfinite(value) or abs(residual) > 1e-6:
            return NumericalSolveResult(False, None, ["substitution_verification_failed"], {"stage": "numerical_fallback", "family_id": family_id, "residual": residual})
        return NumericalSolveResult(
            True,
            value,
            [],
            {"stage": "numerical_fallback", "family_id": family_id, "bounds": bounds, "residual": residual, "confidence_cap": 0.75},
        )
    except Exception as exc:
        return NumericalSolveResult(False, None, [f"numerical_error:{type(exc).__name__}"], {"stage": "numerical_fallback", "family_id": family_id, "bounds": bounds})


def _family_function(family_id: str, parameters: dict):
    if family_id == "rc_charge_fraction":
        tau = float(parameters.get("tau", 0.0))
        fraction = float(parameters.get("fraction", 0.0))
        if tau <= 0 or not 0 < fraction < 1:
            return None
        return lambda t: 1.0 - math.exp(-t / tau) - fraction
    if family_id == "rc_discharge_fraction":
        tau = float(parameters.get("tau", 0.0))
        fraction = float(parameters.get("fraction", 0.0))
        if tau <= 0 or not 0 < fraction < 1:
            return None
        return lambda t: math.exp(-t / tau) - fraction
    return None
