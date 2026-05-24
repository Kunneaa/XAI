"""Project-owned unit registry for normalization.

The full solver will do SI conversion after planning. This module only owns
unit canonicalization and dimensions so earlier stages can reject unknown units
instead of delegating that decision to an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class UnitInfo:
    canonical: str
    dimension: str
    si_factor: float


UNIT_REGISTRY: Dict[str, UnitInfo] = {
    "-": UnitInfo("-", "dimensionless", 1.0),
    "%": UnitInfo("%", "percent", 0.01),
    "times": UnitInfo("times", "dimensionless", 1.0),
    "deg": UnitInfo("deg", "angle", 1.0),
    "degree": UnitInfo("deg", "angle", 1.0),
    "degrees": UnitInfo("deg", "angle", 1.0),
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
