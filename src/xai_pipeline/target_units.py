"""Target-unit policy and SI-to-target conversion helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .units import UNIT_REGISTRY, unit_info


PREFERRED_UNIT_BY_DIMENSION = {
    "capacitance": "F",
    "charge": "C",
    "current": "A",
    "electric_field": "V/m",
    "energy": "J",
    "force": "N",
    "frequency": "Hz",
    "inductance": "H",
    "length": "m",
    "magnetic_field": "T",
    "magnetic_flux": "Wb",
    "power": "W",
    "resistance": "Ω",
    "time": "s",
    "voltage": "V",
    "angle": "deg",
    "dimensionless": "-",
    "percent": "%",
}


@dataclass(frozen=True)
class TargetConversion:
    ok: bool
    value: float | None
    unit: str | None
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {"ok": self.ok, "value": self.value, "unit": self.unit, "issues": list(self.issues), "trace": dict(self.trace)}


def choose_target_unit(target_dimension: str, requested_unit: str | None = None) -> str:
    if requested_unit and unit_info(requested_unit) is not None:
        return unit_info(requested_unit).canonical
    return PREFERRED_UNIT_BY_DIMENSION.get(target_dimension, "-")


def convert_si_to_target(value_si: float, target_dimension: str, target_unit: str | None = None) -> TargetConversion:
    unit = choose_target_unit(target_dimension, target_unit)
    info = unit_info(unit)
    if info is None:
        return TargetConversion(False, None, unit, [f"unknown_target_unit:{unit}"], {"stage": "target_unit_converter"})
    if info.dimension != target_dimension and not (target_dimension == "dimensionless" and info.dimension in {"dimensionless", "percent"}):
        return TargetConversion(False, None, unit, [f"target_dimension_mismatch:{target_dimension}:{info.dimension}"], {"stage": "target_unit_converter"})
    return TargetConversion(
        True,
        float(value_si) / info.si_factor,
        info.canonical,
        [],
        {"stage": "target_unit_converter", "target_dimension": target_dimension, "target_unit": info.canonical},
    )


def detect_requested_target_unit(front_payload: dict, target_dimension: str) -> str | None:
    """Detect explicit output-unit requests without copying input units."""

    text = " ".join(
        [
            " ".join(front_payload.get("target_hints", [])),
            str(front_payload.get("canonical_question") or ""),
        ]
    )
    candidates: list[str] = []
    patterns = [
        r"\bunit\s*:\s*([A-Za-zμΩ/%°^0-9*./]+)",
        r"\banswer\s+in\s+([A-Za-zμΩ/%°^0-9*./]+)",
        r"\bresult\s+in\s+([A-Za-zμΩ/%°^0-9*./]+)",
        r"\bexpress(?:ed)?\s+in\s+([A-Za-zμΩ/%°^0-9*./]+)",
        r"\bin\s+([A-Za-zμΩ/%°^0-9*./]+)\s*(?:\.|\?|$)",
    ]
    for pattern in patterns:
        candidates.extend(match.group(1).strip("()[] ,.;:?") for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    for candidate in candidates:
        info = unit_info(candidate)
        if info is None:
            continue
        if info.dimension == target_dimension or (target_dimension == "dimensionless" and info.dimension in {"dimensionless", "percent"}):
            return info.canonical
    return None
