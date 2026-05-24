"""Principle-family equation selection boundary."""

from __future__ import annotations

import re
from itertools import combinations
from dataclasses import dataclass
from typing import List

from .registries import FORMULA_REGISTRY, PRINCIPLE_IDS, FormulaSpec
from .units import unit_info


@dataclass(frozen=True)
class PrincipleSelection:
    ok: bool
    formula_ids: List[str]
    issues: List[str]
    trace: dict

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "formula_ids": list(self.formula_ids),
            "issues": list(self.issues),
            "trace": dict(self.trace),
        }


def select_minimal_equation_subset(front_payload: dict, route_result, target_dimension: str | None = None) -> PrincipleSelection:
    """Select registry formulas that are dimension-connected to current inputs.

    This is intentionally conservative. It does not invoke SymPy; it only
    prepares the small connected subset that a future symbolic worker may use.
    """

    if route_result.task_type == "unknown":
        return PrincipleSelection(False, [], ["unknown_route"], {"stage": "principle_selector"})
    available = [q.get("dimension") for q in front_payload.get("quantities", [])]
    selected: List[str] = []
    for formula_id, spec in FORMULA_REGISTRY.items():
        if spec.task_type != route_result.task_type:
            continue
        if target_dimension and spec.target_dimension != target_dimension:
            continue
        if _has_required_dimensions(available, spec.required_dimensions):
            selected.append(formula_id)
    issues = [] if selected else ["no_connected_formula_subset"]
    principle_ids = sorted({FORMULA_REGISTRY[formula_id].principle_id for formula_id in selected})
    unknown_principles = [principle_id for principle_id in principle_ids if principle_id not in PRINCIPLE_IDS]
    issues.extend(f"unknown_principle:{principle_id}" for principle_id in unknown_principles)
    return PrincipleSelection(
        ok=not issues,
        formula_ids=selected,
        issues=issues,
        trace={
            "stage": "principle_selector",
            "route_task_type": route_result.task_type,
            "available_dimensions": available,
            "target_dimension": target_dimension,
            "selected_count": len(selected),
        },
    )


def _has_required_dimensions(available: list[str | None], required: tuple[str, ...]) -> bool:
    pool = list(available)
    for dimension in required:
        if dimension not in pool:
            return False
        pool.remove(dimension)
    return True


def build_registry_symbolic_plan(front_payload: dict, route_result, selection: PrincipleSelection) -> dict | None:
    """Build a safe symbolic plan from a minimal registry-owned equation graph.

    The graph bridge is deliberately narrow: it uses only formulas from
    code-owned registries, substitutes only auditable extracted quantities or
    physical constants, and rejects ambiguous repeated dimensions. Unlike the
    original one-equation bridge, this can also solve safe inverse formulas and
    small connected subsets with deterministic intermediate unknowns.
    """

    if route_result.task_type == "unknown":
        return None
    candidates = _candidate_formulas(route_result, selection)
    target_dimensions = _target_dimensions(front_payload, route_result, candidates)
    for target_dimension in target_dimensions:
        for target_symbol in _target_symbols(front_payload, target_dimension):
            max_size = min(3, len(candidates))
            for size in range(1, max_size + 1):
                for combo in combinations(candidates, size):
                    plan = _build_symbolic_plan_for_combo(front_payload, combo, target_symbol, target_dimension)
                    if plan is not None:
                        return plan
    return None


def _candidate_formulas(route_result, selection: PrincipleSelection) -> list[FormulaSpec]:
    selected_ids = list(selection.formula_ids) if selection is not None else []
    route_principles = {
        spec.principle_id
        for spec in FORMULA_REGISTRY.values()
        if spec.task_type == route_result.task_type
    }
    scored: list[tuple[int, str, FormulaSpec]] = []
    for formula_id, spec in FORMULA_REGISTRY.items():
        if spec.task_type != route_result.task_type and spec.principle_id not in route_principles:
            continue
        if not _is_safe_equation_expression(spec.expression):
            continue
        score = 0
        if formula_id in selected_ids:
            score -= 30
        if spec.task_type == route_result.task_type:
            score -= 20
        if spec.target_dimension in _ROUTE_TARGET_PRIORITY.get(route_result.task_type, ()):
            score -= 5
        scored.append((score, formula_id, spec))
    return [spec for _, _, spec in sorted(scored, key=lambda item: (item[0], item[1]))]


def _target_dimensions(front_payload: dict, route_result, candidates: list[FormulaSpec]) -> list[str]:
    ordered: list[str] = []

    def add(dimension: str | None) -> None:
        if dimension and dimension not in ordered:
            ordered.append(dimension)

    for quantity in front_payload.get("symbolic_quantities", []):
        add(quantity.get("dimension"))
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    for keyword, dimension in TARGET_DIMENSION_KEYWORDS:
        if keyword in target_text:
            add(dimension)
    for dimension in _ROUTE_TARGET_PRIORITY.get(route_result.task_type, ()):
        add(dimension)
    for spec in candidates:
        if spec.task_type == route_result.task_type:
            add(spec.target_dimension)
    return ordered


def _target_symbols(front_payload: dict, target_dimension: str) -> list[str]:
    symbols: list[str] = []
    for quantity in front_payload.get("symbolic_quantities", []):
        if quantity.get("dimension") == target_dimension:
            symbol = _clean_symbol(str(quantity.get("symbol") or ""))
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    for symbol in DIMENSION_SYMBOLS.get(target_dimension, ()):
        if symbol.isidentifier() and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _build_symbolic_plan_for_combo(front_payload: dict, combo: tuple[FormulaSpec, ...], target_symbol: str, target_dimension: str) -> dict | None:
    equations: list[str] = []
    formula_ids: list[str] = []
    principle_ids: list[str] = []
    produced_symbols: set[str] = set()
    symbol_dimensions: dict[str, str] = {}
    for spec in combo:
        sanitized = _sanitize_equation(spec.expression)
        if sanitized is None:
            return None
        lhs, _ = sanitized.split("=", 1)
        lhs_symbol = _clean_symbol(lhs)
        if lhs_symbol is None:
            return None
        equations.append(sanitized)
        formula_ids.append(spec.formula_id)
        if spec.principle_id not in principle_ids:
            principle_ids.append(spec.principle_id)
        produced_symbols.add(lhs_symbol)
        symbol_dimensions[lhs_symbol] = spec.target_dimension
    symbols = set().union(*(_equation_symbols(equation) for equation in equations))
    substitutions = _known_symbol_values(front_payload, symbols, target_symbol)
    constant_substitutions = _constant_substitutions(front_payload, symbols)
    if not substitutions and not constant_substitutions:
        return None
    unresolved = symbols - set(substitutions) - set(constant_substitutions) - _SAFE_FUNCTIONS
    unknowns = sorted(unresolved)
    if target_symbol not in unknowns:
        return None
    if not set(unknowns).issubset(produced_symbols | {target_symbol}):
        return None
    if len(equations) < len(unknowns):
        return None
    numeric_equations = [_substitute_symbols(equation, {**substitutions, **constant_substitutions}) for equation in equations]
    targets = [{"symbol": target_symbol, "dimension": target_dimension, "unit": _unit_for_dimension(target_dimension)}]
    for symbol in unknowns:
        if symbol == target_symbol:
            continue
        dimension = symbol_dimensions.get(symbol) or _dimension_for_symbol(symbol) or "dimensionless"
        targets.append({"symbol": symbol, "dimension": dimension, "unit": _unit_for_dimension(dimension)})
    return {
        "symbolic_family": "registry_equation_graph",
        "formula_ids": formula_ids,
        "principle_ids": principle_ids,
        "equations": numeric_equations,
        "targets": targets,
        "timeout_seconds": 3.0,
        "non_negative_target": _should_force_non_negative(front_payload, target_dimension),
        "trace": {
            "stage": "principle_equation_graph",
            "mode": "minimal_registry_subset",
            "formula_ids": formula_ids,
            "substitutions": substitutions,
            "constant_substitutions": constant_substitutions,
            "source_expressions": [spec.expression for spec in combo],
            "unknowns": unknowns,
        },
    }


_ROUTE_TARGET_PRIORITY = {
    "capacitor_charge": ("charge",),
    "capacitance": ("capacitance",),
    "capacitor_energy": ("energy",),
    "capacitor_final_voltage": ("voltage",),
    "ohm_law": ("current", "voltage", "resistance"),
    "electric_power": ("power",),
    "inductor_energy": ("energy", "current"),
    "inductance": ("inductance",),
    "lc_frequency": ("frequency",),
    "lc_period": ("time",),
    "rlc_impedance": ("resistance",),
    "power_factor": ("dimensionless",),
    "magnetic_flux": ("magnetic_flux",),
    "solenoid_magnetic_field": ("magnetic_field",),
    "turn_density": ("turn_density",),
}


TARGET_DIMENSION_KEYWORDS = (
    ("capacitance", "capacitance"),
    ("capacity", "capacitance"),
    ("charge", "charge"),
    ("current", "current"),
    ("voltage", "voltage"),
    ("potential difference", "voltage"),
    ("resistance", "resistance"),
    ("impedance", "resistance"),
    ("power factor", "dimensionless"),
    ("power", "power"),
    ("energy", "energy"),
    ("work", "energy"),
    ("frequency", "frequency"),
    ("period", "time"),
    ("time", "time"),
    ("electric field", "electric_field"),
    ("field strength", "electric_field"),
    ("magnetic field", "magnetic_field"),
    ("magnetic flux", "magnetic_flux"),
    ("flux", "magnetic_flux"),
    ("force", "force"),
    ("distance", "length"),
    ("speed", "velocity"),
)


_SAFE_FUNCTIONS = {"sqrt", "sin", "cos", "tan", "pi"}


def _is_safe_equation_expression(expression: str) -> bool:
    sanitized = _sanitize_equation(expression)
    return sanitized is not None and _is_single_equation(sanitized)


def _sanitize_equation(expression: str) -> str | None:
    if not _is_single_equation(expression):
        return None
    if any(cue in expression for cue in ["'", "|", "vector_sum", " by ", " when ", " or ", " at "]):
        return None
    if "abs(" in expression or "abs " in expression:
        return None
    sanitized = expression.replace("^", "**")
    sanitized = re.sub(r"\s*=\s*", "=", sanitized.strip())
    sanitized = re.sub(r"(?<=\d)(?=[A-Za-z_])", "*", sanitized)
    sanitized = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=[A-Za-z_(])", "*", sanitized)
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().=\s]+", sanitized):
        return None
    return sanitized


def _known_symbol_values(front_payload: dict, symbols: set[str], target_symbol: str) -> dict[str, float]:
    known: dict[str, float] = {}
    dims = _by_dimension(front_payload)
    for quantity in front_payload.get("quantities", []):
        symbol = _clean_symbol(str(quantity.get("symbol") or ""))
        if symbol and symbol in symbols and symbol != target_symbol:
            value = _quantity_si_value(quantity)
            if value is not None:
                known[symbol] = value
    for symbol in symbols:
        if symbol in known or symbol == target_symbol or symbol in _SAFE_FUNCTIONS:
            continue
        dimension = _dimension_for_symbol(symbol)
        if not dimension:
            continue
        values = dims.get(dimension) or []
        if len(values) != 1:
            continue
        value = _quantity_si_value(values[0])
        if value is not None:
            known[symbol] = value
    return known


def _constant_substitutions(front_payload: dict, symbols: set[str]) -> dict[str, float]:
    constants: dict[str, float] = {}
    if "epsilon0" in symbols:
        constants["epsilon0"] = 8.8541878128e-12
    if "mu0" in symbols:
        constants["mu0"] = 4 * 3.141592653589793e-7
    if "epsilon_r" in symbols:
        text = str(front_payload.get("canonical_question") or "").lower()
        if "air" in text or "vacuum" in text or "dielectric" not in text:
            constants["epsilon_r"] = 1.0
    return constants


def _substitute_symbols(equation: str, substitutions: dict[str, float]) -> str:
    substituted = equation
    for symbol, value in sorted(substitutions.items(), key=lambda item: len(item[0]), reverse=True):
        substituted = re.sub(rf"\b{re.escape(symbol)}\b", repr(float(value)), substituted)
    return substituted


def _quantity_si_value(quantity: dict) -> float | None:
    info = unit_info(quantity.get("unit") or "")
    if info is None:
        return None
    try:
        return float(quantity["value"]) * info.si_factor
    except (KeyError, TypeError, ValueError):
        return None


def _dimension_for_symbol(symbol: str) -> str | None:
    matches = [dimension for dimension, symbols in DIMENSION_SYMBOLS.items() if symbol in symbols]
    return matches[0] if len(matches) == 1 else None


def _unit_for_dimension(dimension: str) -> str | None:
    return {
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
        "velocity": "m/s",
        "voltage": "V",
        "dimensionless": "-",
    }.get(dimension)


def _should_force_non_negative(front_payload: dict, target_dimension: str) -> bool:
    text = str(front_payload.get("canonical_question") or "").lower()
    if any(cue in text for cue in ["direction", "signed", "polarity", "opposite direction"]):
        return False
    return target_dimension in {
        "capacitance",
        "current",
        "electric_field",
        "energy",
        "force",
        "frequency",
        "inductance",
        "length",
        "magnetic_field",
        "power",
        "resistance",
        "time",
    }


def _by_dimension(front_payload: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for quantity in front_payload.get("quantities", []):
        dimension = quantity.get("dimension")
        if dimension:
            out.setdefault(dimension, []).append(quantity)
    return out


def _first_dimension_quantity(dims: dict[str, list[dict]], dimension: str) -> dict | None:
    values = dims.get(dimension) or []
    return values[0] if values else None


def _is_single_equation(expression: str) -> bool:
    return isinstance(expression, str) and expression.count("=") == 1


def _clean_symbol(text: str) -> str | None:
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*", text)
    return match.group(1) if match else None


def _equation_symbols(expression: str) -> set[str]:
    symbols = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))
    return symbols - {"sqrt", "sin", "cos", "tan", "abs", "min", "max"}


DIMENSION_SYMBOLS = {
    "area": ("A", "area"),
    "capacitance": ("C",),
    "charge": ("Q", "q"),
    "count": ("N",),
    "current": ("I",),
    "electric_field": ("E",),
    "energy": ("W", "E"),
    "force": ("F",),
    "frequency": ("f",),
    "inductance": ("L",),
    "length": ("r", "d", "l", "x"),
    "mass": ("m",),
    "magnetic_field": ("B",),
    "magnetic_flux": ("Phi", "flux"),
    "number_density": ("n",),
    "power": ("P",),
    "resistance": ("R", "Z", "XL", "XC"),
    "angle": ("theta", "phi"),
    "time": ("t", "T"),
    "velocity": ("v", "v_d"),
    "voltage": ("U", "V"),
}


def _symbol_for_dimension(dimension: str, candidates: set[str]) -> str | None:
    preferred = DIMENSION_SYMBOLS.get(dimension, ())
    for symbol in preferred:
        if symbol in candidates:
            return symbol
    if len(candidates) == 1:
        return next(iter(candidates))
    return None
