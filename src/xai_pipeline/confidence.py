"""Confidence cap policy from the core contract."""

from __future__ import annotations


CAPS = {
    "verifier_failed": 0.45,
    "qwen_json_repair": 0.80,
    "unconstrained_json_fallback": 0.82,
    "weak_conceptual_principle": 0.75,
    "geometry_assumption": 0.70,
    "numerical_fallback": 0.75,
    "weak_numerical_bounds": 0.65,
    "partial_multi_target": 0.70,
}


def apply_confidence_caps(base_confidence: float, cap_reasons: list[str]) -> tuple[float, list[dict]]:
    confidence = float(base_confidence)
    applied: list[dict] = []
    for reason in cap_reasons:
        cap = CAPS.get(reason)
        if cap is None:
            continue
        if confidence > cap:
            confidence = cap
        applied.append({"reason": reason, "cap": cap})
    return confidence, applied
