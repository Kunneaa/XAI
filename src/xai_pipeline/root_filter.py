"""Target-aware root filtering boundary."""

from __future__ import annotations


NON_NEGATIVE_DIMENSIONS = {"mass", "resistance", "capacitance", "length", "frequency", "time", "energy", "inductance"}
SIGNED_DIMENSIONS = {"charge", "current", "voltage", "velocity", "acceleration", "force"}


def filter_roots(roots: list[float], *, target_dimension: str, elapsed_time: bool = False, non_negative: bool = False) -> dict:
    valid: list[float] = []
    for root in roots:
        value = float(root)
        if (non_negative or target_dimension in NON_NEGATIVE_DIMENSIONS) and value < 0:
            continue
        if target_dimension == "time" and elapsed_time and value < 0:
            continue
        valid.append(value)
    issues: list[str] = []
    if not valid:
        issues.append("no_valid_roots")
    elif len(valid) > 1:
        issues.append("ambiguous_solution")
    return {
        "stage": "root_filter",
        "target_dimension": target_dimension,
        "non_negative": non_negative,
        "input_roots": list(roots),
        "valid_roots": valid,
        "issues": issues,
    }
