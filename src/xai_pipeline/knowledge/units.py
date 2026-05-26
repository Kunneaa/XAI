"""Project-owned unit registry and SI conversion helpers.

The equation engine receives plain SI numbers. This module owns unit
canonicalization, dimension checks, hidden-SI defaults, and target-unit
conversion so no model or free-form text can decide unit semantics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class UnitInfo:
    canonical: str
    dimension: str
    si_factor: float


@dataclass(frozen=True)
class TargetConversion:
    ok: bool
    value: float | None
    unit: str | None
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "value": self.value,
            "unit": self.unit,
            "issues": list(self.issues),
            "trace": dict(self.trace),
        }


UNIT_REGISTRY: Dict[str, UnitInfo] = {
    "-": UnitInfo("-", "dimensionless", 1.0),
    "%": UnitInfo("%", "percent", 0.01),
    "times": UnitInfo("times", "dimensionless", 1.0),
    "deg": UnitInfo("deg", "angle", 0.017453292519943295),
    "degree": UnitInfo("deg", "angle", 0.017453292519943295),
    "degrees": UnitInfo("deg", "angle", 0.017453292519943295),
    "rad": UnitInfo("rad", "angle", 1.0),
    "N*m^2/C^2": UnitInfo("N*m^2/C^2", "coulomb_constant_unit", 1.0),
    "N*m2/C2": UnitInfo("N*m^2/C^2", "coulomb_constant_unit", 1.0),
    "N.m2/C2": UnitInfo("N*m^2/C^2", "coulomb_constant_unit", 1.0),
    "°C": UnitInfo("°C", "temperature", 1.0),
    "atm": UnitInfo("atm", "pressure", 101325.0),
    "ml": UnitInfo("ml", "volume", 1e-6),
    "mL": UnitInfo("ml", "volume", 1e-6),
    "turn": UnitInfo("turn", "count", 1.0),
    "turns": UnitInfo("turns", "count", 1.0),
    "turns/m": UnitInfo("turns/m", "turn_density", 1.0),
    "rad/s": UnitInfo("rad/s", "angular_frequency", 1.0),
    "Ω*m": UnitInfo("Ω*m", "resistivity", 1.0),
    "Ωm": UnitInfo("Ω*m", "resistivity", 1.0),
    "ohm*m": UnitInfo("Ω*m", "resistivity", 1.0),
    "ohm m": UnitInfo("Ω*m", "resistivity", 1.0),
    "m": UnitInfo("m", "length", 1.0),
    "m²": UnitInfo("m^2", "area", 1.0),
    "m^2": UnitInfo("m^2", "area", 1.0),
    "m2": UnitInfo("m^2", "area", 1.0),
    "cm²": UnitInfo("cm^2", "area", 1e-4),
    "cm^2": UnitInfo("cm^2", "area", 1e-4),
    "cm2": UnitInfo("cm^2", "area", 1e-4),
    "m³": UnitInfo("m^3", "volume", 1.0),
    "m^3": UnitInfo("m^3", "volume", 1.0),
    "m3": UnitInfo("m^3", "volume", 1.0),
    "cm³": UnitInfo("cm^3", "volume", 1e-6),
    "cm^3": UnitInfo("cm^3", "volume", 1e-6),
    "cm3": UnitInfo("cm^3", "volume", 1e-6),
    "cm": UnitInfo("cm", "length", 1e-2),
    "mm": UnitInfo("mm", "length", 1e-3),
    "km": UnitInfo("km", "length", 1e3),
    "s": UnitInfo("s", "time", 1.0),
    "ms": UnitInfo("ms", "time", 1e-3),
    "kg": UnitInfo("kg", "mass", 1.0),
    "g": UnitInfo("g", "mass", 1e-3),
    "kg/m³": UnitInfo("kg/m^3", "density", 1.0),
    "kg/m^3": UnitInfo("kg/m^3", "density", 1.0),
    "kg/m3": UnitInfo("kg/m^3", "density", 1.0),
    "g/cm³": UnitInfo("g/cm^3", "density", 1000.0),
    "g/cm^3": UnitInfo("g/cm^3", "density", 1000.0),
    "g/cm3": UnitInfo("g/cm^3", "density", 1000.0),
    "J/m³": UnitInfo("J/m^3", "energy_density", 1.0),
    "J/m^3": UnitInfo("J/m^3", "energy_density", 1.0),
    "J/m3": UnitInfo("J/m^3", "energy_density", 1.0),
    "m^-3": UnitInfo("m^-3", "number_density", 1.0),
    "m-3": UnitInfo("m^-3", "number_density", 1.0),
    "cm^-3": UnitInfo("cm^-3", "number_density", 1e6),
    "cm-3": UnitInfo("cm^-3", "number_density", 1e6),
    "C": UnitInfo("C", "charge", 1.0),
    "C/m^2": UnitInfo("C/m^2", "surface_charge_density", 1.0),
    "C/m": UnitInfo("C/m", "linear_charge_density", 1.0),
    "mC": UnitInfo("mC", "charge", 1e-3),
    "μC": UnitInfo("μC", "charge", 1e-6),
    "μC/m^2": UnitInfo("μC/m^2", "surface_charge_density", 1e-6),
    "uC/m^2": UnitInfo("μC/m^2", "surface_charge_density", 1e-6),
    "μC/m": UnitInfo("μC/m", "linear_charge_density", 1e-6),
    "uC/m": UnitInfo("μC/m", "linear_charge_density", 1e-6),
    "nC/m": UnitInfo("nC/m", "linear_charge_density", 1e-9),
    "uC": UnitInfo("μC", "charge", 1e-6),
    "nC": UnitInfo("nC", "charge", 1e-9),
    "pC": UnitInfo("pC", "charge", 1e-12),
    "V": UnitInfo("V", "voltage", 1.0),
    "mV": UnitInfo("mV", "voltage", 1e-3),
    "kV": UnitInfo("kV", "voltage", 1e3),
    "A": UnitInfo("A", "current", 1.0),
    "mA": UnitInfo("mA", "current", 1e-3),
    "μA": UnitInfo("μA", "current", 1e-6),
    "uA": UnitInfo("μA", "current", 1e-6),
    "Ω": UnitInfo("Ω", "resistance", 1.0),
    "ohm": UnitInfo("Ω", "resistance", 1.0),
    "ohms": UnitInfo("Ω", "resistance", 1.0),
    "kΩ": UnitInfo("kΩ", "resistance", 1e3),
    "kohm": UnitInfo("kΩ", "resistance", 1e3),
    "F": UnitInfo("F", "capacitance", 1.0),
    "mF": UnitInfo("mF", "capacitance", 1e-3),
    "μF": UnitInfo("μF", "capacitance", 1e-6),
    "uF": UnitInfo("μF", "capacitance", 1e-6),
    "nF": UnitInfo("nF", "capacitance", 1e-9),
    "pF": UnitInfo("pF", "capacitance", 1e-12),
    "H": UnitInfo("H", "inductance", 1.0),
    "mH": UnitInfo("mH", "inductance", 1e-3),
    "μH": UnitInfo("μH", "inductance", 1e-6),
    "uH": UnitInfo("μH", "inductance", 1e-6),
    "Hz": UnitInfo("Hz", "frequency", 1.0),
    "kHz": UnitInfo("kHz", "frequency", 1e3),
    "N": UnitInfo("N", "force", 1.0),
    "mN": UnitInfo("mN", "force", 1e-3),
    "kN": UnitInfo("kN", "force", 1e3),
    "J": UnitInfo("J", "energy", 1.0),
    "mJ": UnitInfo("mJ", "energy", 1e-3),
    "μJ": UnitInfo("μJ", "energy", 1e-6),
    "uJ": UnitInfo("μJ", "energy", 1e-6),
    "nJ": UnitInfo("nJ", "energy", 1e-9),
    "W": UnitInfo("W", "power", 1.0),
    "mW": UnitInfo("mW", "power", 1e-3),
    "Pa": UnitInfo("Pa", "pressure", 1.0),
    "kPa": UnitInfo("kPa", "pressure", 1e3),
    "V/m": UnitInfo("V/m", "electric_field", 1.0),
    "N/C": UnitInfo("N/C", "electric_field", 1.0),
    "T": UnitInfo("T", "magnetic_field", 1.0),
    "Wb": UnitInfo("Wb", "magnetic_flux", 1.0),
    "m/s": UnitInfo("m/s", "velocity", 1.0),
    "km/s": UnitInfo("km/s", "velocity", 1e3),
    "m/s²": UnitInfo("m/s^2", "acceleration", 1.0),
    "m/s^2": UnitInfo("m/s^2", "acceleration", 1.0),
    "m/s2": UnitInfo("m/s^2", "acceleration", 1.0),
    "N/m": UnitInfo("N/m", "spring_constant", 1.0),
    "J/kg": UnitInfo("J/kg", "specific_energy", 1.0),
    "J/(kg°C)": UnitInfo("J/(kg°C)", "specific_heat", 1.0),
    "J/kg°C": UnitInfo("J/(kg°C)", "specific_heat", 1.0),
}


UNIT_ALIASES = {
    "µ": "μ",
    "ω": "Ω",
    "Ohm": "Ω",
    "Ohms": "Ω",
    "ohm": "Ω",
    "ohms": "Ω",
    "Volt": "V",
    "Volts": "V",
    "volt": "V",
    "volts": "V",
    "Amp": "A",
    "Amps": "A",
    "amp": "A",
    "amps": "A",
    "ampere": "A",
    "amperes": "A",
    "Joule": "J",
    "Joules": "J",
    "joule": "J",
    "joules": "J",
    "Watt": "W",
    "Watts": "W",
    "watt": "W",
    "watts": "W",
    "Farad": "F",
    "Farads": "F",
    "farad": "F",
    "farads": "F",
    "Coulomb": "C",
    "Coulombs": "C",
    "coulomb": "C",
    "coulombs": "C",
    "°": "deg",
}


def normalize_unit(raw_unit: str) -> Optional[str]:
    unit = (raw_unit or "").strip().strip("()[]").replace("µ", "μ")
    unit = unit.replace("N.m^2/C^2", "N*m^2/C^2")
    unit = unit.replace("N×m²/C²", "N*m^2/C^2")
    unit = unit.replace("²", "^2").replace("³", "^3")
    unit = unit.replace("⁻", "-")
    unit = unit.replace(" / ", "/")
    unit = UNIT_ALIASES.get(unit, unit)
    return UNIT_REGISTRY[unit].canonical if unit in UNIT_REGISTRY else None


def unit_info(unit: str) -> Optional[UnitInfo]:
    canonical = normalize_unit(unit)
    if canonical is None:
        return None
    return UNIT_REGISTRY.get(canonical)


PREFERRED_UNIT_BY_DIMENSION = {
    "angle": "deg",
    "area": "m^2",
    "capacitance": "F",
    "charge": "C",
    "count": "turns",
    "current": "A",
    "dimensionless": "-",
    "electric_field": "V/m",
    "energy": "J",
    "energy_density": "J/m^3",
    "force": "N",
    "frequency": "Hz",
    "inductance": "H",
    "length": "m",
    "magnetic_field": "T",
    "magnetic_flux": "Wb",
    "percent": "%",
    "power": "W",
    "resistance": "Ω",
    "resistivity": "Ω*m",
    "time": "s",
    "turn_density": "turns/m",
    "angular_frequency": "rad/s",
    "voltage": "V",
}


def choose_target_unit(target_dimension: str, requested_unit: str | None = None) -> str:
    if requested_unit and unit_info(requested_unit) is not None:
        return unit_info(requested_unit).canonical  # type: ignore[union-attr]
    return PREFERRED_UNIT_BY_DIMENSION.get(target_dimension, "-")


def convert_si_to_target(value_si: float, target_dimension: str, target_unit: str | None = None) -> TargetConversion:
    unit = choose_target_unit(target_dimension, target_unit)
    info = unit_info(unit)
    if info is None:
        return TargetConversion(False, None, unit, [f"unknown_target_unit:{unit}"], {"stage": "units.target_conversion"})
    dimension_ok = info.dimension == target_dimension or (
        target_dimension == "dimensionless" and info.dimension in {"dimensionless", "percent"}
    )
    if not dimension_ok:
        return TargetConversion(
            False,
            None,
            unit,
            [f"target_dimension_mismatch:{target_dimension}:{info.dimension}"],
            {"stage": "units.target_conversion"},
        )
    return TargetConversion(
        True,
        float(value_si) / info.si_factor,
        info.canonical,
        [],
        {"stage": "units.target_conversion", "target_dimension": target_dimension, "target_unit": info.canonical},
    )


def detect_requested_target_unit(front_payload: dict, target_dimension: str) -> str | None:
    """Detect explicit output-unit requests without copying answer examples."""

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
