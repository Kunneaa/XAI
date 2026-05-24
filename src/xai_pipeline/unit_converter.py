"""Deterministic SI unit conversion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

from .units import unit_info
from .target_units import convert_si_to_target, detect_requested_target_unit


SI_UNIT_BY_DIMENSION = {
    "area": "m^2",
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
    "turn_density": "turns/m",
    "voltage": "V",
}


@dataclass(frozen=True)
class UnitConversionResult:
    ok: bool
    quantities_si: List[dict]
    issues: List[str]
    trace: dict

    def to_dict(self):
        return {
            "ok": self.ok,
            "quantities_si": [dict(item) for item in self.quantities_si],
            "issues": list(self.issues),
            "trace": dict(self.trace),
        }


def convert_front_quantities_to_si(front_payload: dict) -> UnitConversionResult:
    converted: List[dict] = []
    issues: List[str] = []
    for index, quantity in enumerate(front_payload.get("quantities", [])):
        info = unit_info(quantity.get("unit") or "")
        if info is None:
            issues.append(f"unknown_unit:{index}:{quantity.get('unit')}")
            continue
        si_value = float(quantity["value"]) * info.si_factor
        converted.append(
            {
                "raw_text": quantity.get("raw_text"),
                "symbol": quantity.get("symbol"),
                "dimension": info.dimension,
                "original_value": quantity.get("value"),
                "original_unit": quantity.get("unit"),
                "si_value": si_value,
                "si_unit": _si_unit(info.dimension, info.canonical),
            }
        )
    trace = {
        "stage": "unit_converter",
        "input_quantity_count": len(front_payload.get("quantities", [])),
        "converted_quantity_count": len(converted),
    }
    return UnitConversionResult(not issues, converted, issues, trace)


def _si_unit(dimension: str, fallback: Optional[str]) -> str:
    return SI_UNIT_BY_DIMENSION.get(dimension, fallback or "")


def convert_solver_result_to_target(solver_result, requested_unit: str | None = None):
    if not getattr(solver_result, "solved", False) or solver_result.value is None or isinstance(solver_result.value, list):
        return None
    return convert_si_to_target(float(solver_result.value), solver_result.trace.get("target_dimension") or "", requested_unit)


def apply_requested_target_unit(front_payload: dict, solver_result):
    """Convert a verified SI/default solver result for final display when asked."""

    if not getattr(solver_result, "solved", False) or solver_result.value is None or isinstance(solver_result.value, str):
        return solver_result, {"stage": "target_unit_converter", "applied": False, "reason": "unsupported_solver_value_shape"}
    if isinstance(solver_result.value, list):
        return _apply_requested_target_unit_multi(front_payload, solver_result)
    target_dimension = solver_result.trace.get("target_dimension") or ""
    requested_unit = detect_requested_target_unit(front_payload, target_dimension)
    if not requested_unit or requested_unit == solver_result.unit:
        return solver_result, {"stage": "target_unit_converter", "applied": False, "target_dimension": target_dimension, "requested_unit": requested_unit}
    converted = convert_si_to_target(float(solver_result.value), target_dimension, requested_unit)
    if not converted.ok or converted.value is None or converted.unit is None:
        return solver_result, {"stage": "target_unit_converter", "applied": False, "issues": converted.issues, "requested_unit": requested_unit}
    trace = dict(solver_result.trace)
    trace["target_unit_conversion"] = {
        "stage": "target_unit_converter",
        "applied": True,
        "original_value": solver_result.value,
        "original_unit": solver_result.unit,
        "target_value": converted.value,
        "target_unit": converted.unit,
    }
    return (
        replace(
            solver_result,
            answer=_format_target(converted.value, converted.unit),
            value=converted.value,
            unit=converted.unit,
            trace=trace,
        ),
        trace["target_unit_conversion"],
    )


def _format_target(value: float, unit: str) -> str:
    number = f"{value:.6g}"
    return f"{number} {unit}".strip()


def _apply_requested_target_unit_multi(front_payload: dict, solver_result):
    converted_items = []
    traces = []
    applied = False
    for item in solver_result.value:
        next_item = dict(item)
        dimension = item.get("dimension")
        requested_unit = detect_requested_target_unit(front_payload, dimension or "")
        if dimension and requested_unit and requested_unit != item.get("unit"):
            converted = convert_si_to_target(float(item["value"]), dimension, requested_unit)
            traces.append(converted.to_dict())
            if converted.ok and converted.value is not None and converted.unit is not None:
                next_item["value"] = converted.value
                next_item["unit"] = converted.unit
                applied = True
        converted_items.append(next_item)
    if not applied:
        return solver_result, {"stage": "target_unit_converter", "applied": False, "reason": "no_multi_output_target_unit_match", "items_checked": len(converted_items)}
    answer = "; ".join(f"{item.get('name')}={_format_target(float(item['value']), item.get('unit') or '-')}" for item in converted_items)
    trace = dict(solver_result.trace)
    trace["target_unit_conversion"] = {
        "stage": "target_unit_converter",
        "applied": True,
        "mode": "multi_output",
        "items": converted_items,
        "conversion_traces": traces,
    }
    return (
        replace(
            solver_result,
            answer=answer,
            value=converted_items,
            unit=";".join(str(item.get("unit") or "-") for item in converted_items),
            trace=trace,
        ),
        trace["target_unit_conversion"],
    )
