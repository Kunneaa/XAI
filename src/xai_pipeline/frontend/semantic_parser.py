"""Deterministic question normalization and raw quantity extraction."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Iterable, List, Optional, Tuple

from .semantic_ir import (
    Constraint,
    Entity,
    Event,
    Goal,
    NormalizedQuestion,
    NumericConstant,
    Quantity,
    Relation,
    State,
    SymbolicQuantity,
    SymbolicRelation,
    TopologyGraph,
)
from ..knowledge.language import has_frequency_transform_cue
from ..knowledge.units import normalize_unit, unit_info


SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
    }
)

UNIT_PATTERN = (
    r"N\*m\^2/C\^2|N\.m\^2/C\^2|N\*m2/C2|N\.m2/C2|N×m²/C²|V/m|N/C|μC/m\^2|µC/m\^2|uC/m\^2|μC/m²|µC/m²|uC/m²|μC/m2|µC/m2|uC/m2|μC/m|µC/m|uC/m|nC/m|C/m\^2|C/m²|C/m2|C/m|km/s|m/s\^2|m/s²|m/s2|m/s|"
    r"kg/m\^3|kg/m³|kg/m3|g/cm\^3|g/cm³|g/cm3|m\^-3|m-3|cm\^-3|cm-3|J/\(kg°C\)|J/kg°C|"
    r"Ω\*m|Ωm|ohm\*m|ohm m|kΩ|kohm|Ω|ohms?|"
    r"μF|µF|uF|mF|nF|pF|F|"
    r"μC|µC|uC|mC|nC|pC|C|"
    r"μA|µA|uA|mA|A|"
    r"μH|µH|uH|mH|H|"
    r"kHz|Hz|"
    r"mN|kN|N|"
    r"mJ|μJ|µJ|uJ|nJ|J|"
    r"mW|W|"
    r"kV|mV|V|"
    r"cm\^2|cm²|cm2|m\^2|m²|m2|cm\^3|cm³|cm3|m\^3|m³|m3|"
    r"km|cm|mm|m|"
    r"turns/m|rad/s|turns|turn|"
    r"ms|s|kg|g|Pa|kPa|T|Wb|atm|ml|mL|%|°C|°|deg|degree|degrees|rad"
)
UNIT_CAPTURE = rf"\(?(?:{UNIT_PATTERN})\)?"

SYMBOL_CHARS = r"A-Za-zωΩφΦμμεεθΘρΡ′'"
SYMBOL_PATTERN = rf"[{SYMBOL_CHARS}][{SYMBOL_CHARS}0-9_]*(?:\([^)]+\))?"
PLAIN_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
SCI_NUMBER = (
    rf"{PLAIN_NUMBER}(?:\s*(?:x|X|×|\*)\s*10\s*\^?\s*[-+]?\d+|"
    rf"[eE][-+]?\d+)?|[-+]?10\s*\^?\s*[-+]?\d+"
)
NUMBER_PATTERN = rf"(?:{SCI_NUMBER})"

UNIT_END = r"(?=$|[^A-Za-z0-9_/%])"

ASSIGNMENT_RE = re.compile(
    rf"\b(?P<symbol>{SYMBOL_PATTERN})\s*=\s*(?P<number>{NUMBER_PATTERN})\s*(?P<unit>{UNIT_CAPTURE}){UNIT_END}",
    flags=re.IGNORECASE,
)

VALUE_UNIT_RE = re.compile(
    rf"(?<![A-Za-z0-9_=])(?P<number>{NUMBER_PATTERN})\s*(?P<unit>{UNIT_CAPTURE}){UNIT_END}",
    flags=re.IGNORECASE,
)

UNCERTAINTY_PAIR_RE = re.compile(
    rf"(?<![A-Za-z0-9_=])(?P<value>{NUMBER_PATTERN})\s*(?:±|\+/-)\s*(?P<uncertainty>{NUMBER_PATTERN})\s*(?P<unit>{UNIT_CAPTURE}){UNIT_END}",
    flags=re.IGNORECASE,
)

TURN_COUNT_BEFORE_NUMBER_RE = re.compile(
    rf"\b(?P<label>primary|secondary|input|output)?\s*(?:turns?|number of turns)\s*(?:=|is|are|:)?\s*(?P<number>{NUMBER_PATTERN})(?=$|[^A-Za-z0-9_])",
    flags=re.IGNORECASE,
)

RELATION_RE = re.compile(
    rf"\b(?P<lhs>{SYMBOL_PATTERN})\s*=\s*(?P<rhs>[^,.;?]+)",
    flags=re.IGNORECASE,
)

SYMBOL_UNIT_RE = re.compile(
    rf"\b(?P<symbol>[{SYMBOL_CHARS}][{SYMBOL_CHARS}0-9_]*)\s*\((?P<unit>{UNIT_PATTERN})\)",
    flags=re.IGNORECASE,
)

QUOTED_SYMBOL_RE = re.compile(rf"['\"](?P<symbol>[{SYMBOL_CHARS}][{SYMBOL_CHARS}0-9_]*)['\"]")

PHRASE_SYMBOL_RE = re.compile(
    r"\b(?:side length|distance|radius|area|charge|capacitance|voltage|current|resistance|impedance|reactance|inductance|energy|field|flux)\s+"
    rf"(?P<symbol>[{SYMBOL_CHARS}][{SYMBOL_CHARS}0-9_]*)\b",
    flags=re.IGNORECASE,
)

BARE_SYMBOL_RE = re.compile(rf"\b(?P<symbol>[qQ][{SYMBOL_CHARS}0-9_]*|[{SYMBOL_CHARS.lower()}])\b")
FIELD_SYMBOL_RE = re.compile(r"\b(?P<symbol>E[_ ]?[A-Za-z][A-Za-z0-9_]*)\b")

UNITLESS_ASSIGNMENT_RE = re.compile(
    rf"\b(?P<symbol>{SYMBOL_PATTERN})\s*=\s*(?P<number>{NUMBER_PATTERN})(?=$|[^A-Za-z0-9_μΩ/%°])",
    flags=re.IGNORECASE,
)

CONCEPT_PATTERNS = {
    "lc_circuit": re.compile(r"\bLC circuit\b", re.IGNORECASE),
    "ideal_lc_circuit": re.compile(r"\bideal LC circuit\b", re.IGNORECASE),
    "rlc_circuit": re.compile(r"\b(series RLC|RLC)\b", re.IGNORECASE),
    "resonance": re.compile(r"\bresonan\w*\b", re.IGNORECASE),
    "solenoid": re.compile(r"\bsolenoid\b", re.IGNORECASE),
    "inductance": re.compile(r"\binductance\b", re.IGNORECASE),
    "induced_emf": re.compile(r"\b(induced electromotive force|EMF)\b", re.IGNORECASE),
    "power_factor": re.compile(r"\bpower factor|cosφ|cos phi\b", re.IGNORECASE),
    "transformer": re.compile(r"\btransformer\b", re.IGNORECASE),
    "drift_current": re.compile(r"\b(drift velocity|carrier density|number density)\b", re.IGNORECASE),
    "wheatstone_bridge": re.compile(r"\bWheatstone\b", re.IGNORECASE),
    "electric_field_energy": re.compile(r"\belectric field energy\b", re.IGNORECASE),
    "magnetic_field_energy": re.compile(r"\bmagnetic field energy\b", re.IGNORECASE),
    "total_energy": re.compile(r"\btotal energy\b", re.IGNORECASE),
    "si_unit": re.compile(r"\b(SI unit|unit of|what is the unit)\b", re.IGNORECASE),
    "graph_shape": re.compile(r"\b(shape of the graph|graph representing)\b", re.IGNORECASE),
    "proportionality": re.compile(r"\b(directly proportional|depend linearly|depends? on)\b", re.IGNORECASE),
    "qualitative_change": re.compile(r"\b(changes?|increases?|decreases?|reduced|scaled|multiplied|brighter|halve[ds]?|double[ds]?|triple[ds]?|quadruple[ds]?|factor|times|compare)\b", re.IGNORECASE),
    "measurement_uncertainty": re.compile(r"\b(uncertainty|percentage error|percent error|relative error|relative uncertainty|measurement error|random error|absolute error|least count)\b", re.IGNORECASE),
    "parallel_circuit": re.compile(r"\bparallel\b(?!\s*[- ]?\s*plates?\b)", re.IGNORECASE),
    "brightness": re.compile(r"\b(bright|brighter|bulbs?|lamp)\b", re.IGNORECASE),
    "uniform_electric_field": re.compile(r"\buniform electric field\b", re.IGNORECASE),
    "magnetic_flux": re.compile(r"\bmagnetic flux\b|\bflux\b", re.IGNORECASE),
    "reactance": re.compile(r"\breactance\b|\bX_L\b|\bX_C\b|\bXL\b|\bXC\b", re.IGNORECASE),
    "impedance": re.compile(r"\bimpedance\b|\bZ\b", re.IGNORECASE),
}

TARGET_PATTERNS = (
    re.compile(r"\b(?:calculate|find|determine|compute)\s+(?P<target>[^.?]+)", re.IGNORECASE),
    re.compile(r"\bwhat is\s+(?P<target>[^.?]+)", re.IGNORECASE),
    re.compile(r"\bwhat (?:will|would|should|can)\s+be\s+(?P<target>[^.?]+)", re.IGNORECASE),
    re.compile(r"\bwhat\s+value\s+(?:must|should|can|would|will)\s+(?P<target>[^.?]+)", re.IGNORECASE),
    re.compile(r"\bwhat\s+value\s+(?:of|for)\s+(?P<target>[^.?]+)", re.IGNORECASE),
    re.compile(
        r"\bwhat\s+(?P<target>(?:voltage|potential difference|current|resistance|impedance|reactance|"
        r"capacitance|inductance|charge|power|energy|frequency|period|force|field|angle|distance|length)"
        r"\s+[^.?]+)",
        re.IGNORECASE,
    ),
    re.compile(r"\bwhat\s+(?P<target>charge\s+[^.?]+)", re.IGNORECASE),
    re.compile(r"\bwhere is\s+(?P<target>[^.?]+)", re.IGNORECASE),
    re.compile(r"\bhow (?:do|does|will|would|is|are)\s+(?P<target>[^.?]+)", re.IGNORECASE),
    re.compile(r"\bdescribe\s+(?:how\s+)?(?P<target>[^.?]+)", re.IGNORECASE),
)

TARGET_REQUEST_CUE = re.compile(
    r"\b(?:calculate|find|determine|compute|derive|express|evaluate|what|where|how|describe)\b",
    re.IGNORECASE,
)

ANSWER_TYPE_IDS = {"numeric", "symbolic", "conceptual", "yes_no", "multi_output", "unknown"}

ENTITY_PATTERNS = (
    ("resistor", "resistor", re.compile(r"\bresistors?\s*(?P<symbol>R\d*|R_[A-Za-z0-9]+)?\b", re.IGNORECASE)),
    ("capacitor", "capacitor", re.compile(r"\bcapacitors?\s*(?P<symbol>C\d*|C_[A-Za-z0-9]+)?\b", re.IGNORECASE)),
    ("inductor", "inductor", re.compile(r"\b(?:inductor|coil)s?\s*(?P<symbol>L\d*|L_[A-Za-z0-9]+)?\b", re.IGNORECASE)),
    ("charge", "charge", re.compile(r"\b(?:charge|point charge)\s*(?P<symbol>[qQ][A-Za-z0-9_′']*)?(?=$|[^A-Za-z0-9_′'])", re.IGNORECASE)),
    ("battery", "source", re.compile(r"\b(?:battery|source|voltage source)\b", re.IGNORECASE)),
    ("solenoid", "solenoid", re.compile(r"\bsolenoid\b", re.IGNORECASE)),
    ("branch", "branch", re.compile(r"\bbranch(?:es)?\b", re.IGNORECASE)),
    ("node", "node", re.compile(r"\bnode\s*(?P<symbol>[A-Z]\d*)?\b", re.IGNORECASE)),
    ("point", "point", re.compile(r"\bpoint\s+(?P<symbol>[A-Z])\b", re.IGNORECASE)),
)

STATE_PATTERNS = (
    ("initial", re.compile(r"\b(?:initially|initial|at first|before)\b", re.IGNORECASE)),
    ("final", re.compile(r"\b(?:finally|final|after|then|becomes?)\b", re.IGNORECASE)),
    ("resonance", re.compile(r"\b(?:at\s+)?resonan\w*\b", re.IGNORECASE)),
    ("connected_to_source", re.compile(r"\bconnected\s+to\s+(?:a\s+)?(?:battery|source)\b", re.IGNORECASE)),
    ("disconnected_from_source", re.compile(r"\bdisconnected\s+from\s+(?:the\s+)?(?:battery|source)\b|\bbattery\s+removed\b", re.IGNORECASE)),
    ("fully_charged", re.compile(r"\bfully charged\b", re.IGNORECASE)),
    ("changed_frequency", re.compile(r"\bfrequency\s+(?:is\s+)?(?:changed|scaled|multiplied|double[ds]?|halve[ds]?|triple[ds]?|quadruple[ds]?|increased|decreased|reduced)\b", re.IGNORECASE)),
    ("conditional", re.compile(r"\bwhen\b[^,.?;]*", re.IGNORECASE)),
)

EVENT_PATTERNS = (
    ("battery_removed", re.compile(r"\bbattery\s+(?:is\s+)?removed\b|\bremove(?:d)?\s+the\s+battery\b", re.IGNORECASE)),
    ("disconnect", re.compile(r"\bdisconnected?\b", re.IGNORECASE)),
    ("connect", re.compile(r"\bconnected?\b", re.IGNORECASE)),
    ("frequency_changed", re.compile(r"\bfrequency\s+(?:is\s+)?(?:changed|scaled|multiplied|double[ds]?|halve[ds]?|triple[ds]?|quadruple[ds]?|increased|decreased|reduced)\b", re.IGNORECASE)),
    ("comes_to_rest", re.compile(r"\b(?:comes?\s+to\s+rest|stops?|until\s+it\s+stops)\b", re.IGNORECASE)),
    ("dielectric_inserted", re.compile(r"\bdielectric\s+(?:is\s+)?inserted\b|\binsert(?:ed)?\s+(?:a\s+)?dielectric\b", re.IGNORECASE)),
)

GOAL_DIMENSION_KEYWORDS = (
    (("angle", "phase angle"), "angle"),
    (("dielectric constant", "relative permittivity"), "dimensionless"),
    (("magnetic flux density", "flux density"), "magnetic_field"),
    (("electric field", "field strength"), "electric_field"),
    (("magnetic field",), "magnetic_field"),
    (("force",), "force"),
    (("capacitance", "capacity"), "capacitance"),
    (("charge",), "charge"),
    (("current",), "current"),
    (("voltage", "potential difference", "emf"), "voltage"),
    (("resistance", "impedance", "reactance"), "resistance"),
    (("power factor",), "dimensionless"),
    (("percentage uncertainty", "percent uncertainty", "percentage error", "percent error", "relative error"), "percent"),
    (("random error", "absolute error", "average absolute error", "measurement error"), "uncertainty"),
    (("power",), "power"),
    (("energy", "work", "heat"), "energy"),
    (("frequency",), "frequency"),
    (("period", "time"), "time"),
    (("magnetic flux", "flux"), "magnetic_flux"),
    (("distance", "separation", "radius"), "length"),
    (("inductance",), "inductance"),
)


def canonicalize_question(question: str) -> str:
    """Return a deterministic canonical text for cache and parsing."""

    raw_text = str(question or "")
    raw_text = re.sub(
        r"10([⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda match: "10^" + match.group(1).translate(SUPERSCRIPT_TRANSLATION),
        raw_text,
    )
    text = raw_text.translate(SUPERSCRIPT_TRANSLATION)
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "µ": "μ",
        "−": "-",
        "–": "-",
        "—": "-",
        "×": "×",
        "·": "*",
        "π": "pi",
        "Ω": "Ω",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("^^", "^")
    text = re.sub(r"\b([A-Za-z]+)\s*/\s*([A-Za-z]+)\b", r"\1/\2", text)
    text = re.sub(r"(?i)\b10([+-]\d+)", r"10^\1", text)
    text = re.sub(r"(?i)\b10\s*\^\s*([+-]?\d+)", r"10^\1", text)
    text = re.sub(r"(?i)\b10\s+([+-]\d+)", r"10^\1", text)
    text = re.sub(r"(?i)\b(\d+)\.10\^([-+]?\d+)", r"\1×10^\2", text)
    text = re.sub(r"\s*=\s*", " = ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_number(raw: str) -> Optional[float]:
    token = str(raw).strip().replace(" ", "")
    token = token.replace("×", "x").replace("X", "x")
    if "x10" in token:
        base, exponent = token.split("x10", 1)
        exponent = exponent.lstrip("^")
        try:
            return float(base) * (10 ** int(exponent))
        except ValueError:
            return None
    ten_power = re.fullmatch(r"([-+]?)10\^([-+]?\d+)", token)
    if ten_power:
        sign = -1.0 if ten_power.group(1) == "-" else 1.0
        return sign * (10 ** int(ten_power.group(2)))
    try:
        return float(token)
    except ValueError:
        return None


def _context(text: str, span: Tuple[int, int], window: int = 36) -> str:
    start = max(0, span[0] - window)
    end = min(len(text), span[1] + window)
    return text[start:end].strip()


def _build_quantity(
    text: str,
    match: re.Match[str],
    symbol: Optional[str],
    seen_spans: set[Tuple[int, int]],
) -> Optional[Quantity]:
    span = match.span()
    if span in seen_spans:
        return None
    value = parse_number(match.group("number"))
    raw_unit = match.group("unit")
    unit = normalize_unit(raw_unit)
    if value is None or unit is None:
        return None
    info = unit_info(unit)
    seen_spans.add(span)
    confidence = 0.98 if symbol else 0.88
    return Quantity(
        raw_text=match.group(0),
        value=value,
        unit=unit,
        raw_unit=raw_unit,
        symbol=_clean_symbol(symbol),
        dimension=info.dimension if info else None,
        span=span,
        context=_context(text, span),
        confidence=confidence,
    )


def _clean_symbol(symbol: Optional[str]) -> Optional[str]:
    if not symbol:
        return None
    return symbol.strip().strip(",.;:")


def extract_quantities(canonical_question: str) -> List[Quantity]:
    quantities: List[Quantity] = []
    seen_spans: set[Tuple[int, int]] = set()

    for match in UNCERTAINTY_PAIR_RE.finditer(canonical_question):
        unit = normalize_unit(match.group("unit"))
        info = unit_info(unit) if unit else None
        if info is None:
            continue
        value = parse_number(match.group("value"))
        uncertainty = parse_number(match.group("uncertainty"))
        if value is None or uncertainty is None:
            continue
        value_span = match.span("value")
        uncertainty_span = (match.start("uncertainty"), match.end("unit"))
        seen_spans.add(value_span)
        seen_spans.add(uncertainty_span)
        quantities.append(
            Quantity(
                raw_text=match.group("value"),
                value=value,
                unit=unit,
                raw_unit=match.group("unit"),
                symbol=None,
                dimension=info.dimension,
                span=value_span,
                context=_context(canonical_question, match.span()),
                confidence=0.88,
            )
        )
        quantities.append(
            Quantity(
                raw_text=f"{match.group('uncertainty')} {unit}",
                value=uncertainty,
                unit=unit,
                raw_unit=match.group("unit"),
                symbol=None,
                dimension=info.dimension,
                span=uncertainty_span,
                context=_context(canonical_question, match.span()),
                confidence=0.94,
            )
        )

    for match in ASSIGNMENT_RE.finditer(canonical_question):
        quantity = _build_quantity(canonical_question, match, match.group("symbol"), seen_spans)
        if quantity:
            quantities.append(quantity)

    for match in TURN_COUNT_BEFORE_NUMBER_RE.finditer(canonical_question):
        span = match.span()
        if _inside_existing_span(span, seen_spans):
            continue
        value = parse_number(match.group("number"))
        if value is None:
            continue
        label = (match.group("label") or "").lower()
        symbol = "N1" if label in {"primary", "input"} else "N2" if label in {"secondary", "output"} else None
        unit = normalize_unit("turns")
        info = unit_info(unit) if unit else None
        seen_spans.add(span)
        quantities.append(
            Quantity(
                raw_text=match.group(0),
                value=value,
                unit=unit or "turns",
                raw_unit="turns",
                symbol=symbol,
                dimension=info.dimension if info else "count",
                span=span,
                context=_context(canonical_question, span),
                confidence=0.9 if symbol else 0.84,
            )
        )

    for match in VALUE_UNIT_RE.finditer(canonical_question):
        if _inside_existing_span(match.span(), seen_spans):
            continue
        quantity = _build_quantity(canonical_question, match, None, seen_spans)
        if quantity:
            quantities.append(quantity)

    return sorted(quantities, key=lambda q: q.span or (0, 0))


def expand_chained_numeric_equalities(canonical_question: str, quantities: List[Quantity]) -> List[Quantity]:
    """Clone explicit numeric facts across chains such as ``q1 = q2 = 5 μC``.

    The extractor's assignment regex naturally binds the right-most symbol in
    a chained equality. This pass makes the equivalent left-side symbols
    executable facts too, without inventing any value not already present in
    the text. It is structural normalization, not dataset-pattern routing.
    """

    by_symbol = {
        str(quantity.symbol).lower(): quantity
        for quantity in quantities
        if quantity.symbol and quantity.dimension and unit_info(quantity.unit) is not None
    }
    additions: list[Quantity] = []
    chain_re = re.compile(
        rf"\b(?P<chain>{SYMBOL_PATTERN}(?:\s*=\s*{SYMBOL_PATTERN})+)\s*=\s*"
        rf"(?P<number>{NUMBER_PATTERN})\s*(?P<unit>{UNIT_CAPTURE}){UNIT_END}",
        flags=re.IGNORECASE,
    )
    for match in chain_re.finditer(canonical_question):
        value = parse_number(match.group("number"))
        unit = normalize_unit(match.group("unit"))
        info = unit_info(unit) if unit else None
        if value is None or unit is None or info is None:
            continue
        symbols = [_clean_symbol(part) for part in re.split(r"\s*=\s*", match.group("chain"))]
        symbols = [symbol for symbol in symbols if symbol]
        if len(symbols) < 2:
            continue
        for symbol in symbols:
            key = symbol.lower()
            existing = by_symbol.get(key)
            if existing and existing.unit == unit and abs(float(existing.value) - value) <= max(1e-15, abs(value) * 1e-12):
                continue
            cloned = Quantity(
                raw_text=match.group(0),
                value=value,
                unit=unit,
                raw_unit=match.group("unit"),
                symbol=symbol,
                dimension=info.dimension,
                span=match.span(),
                context=_context(canonical_question, match.span()),
                confidence=0.94,
            )
            additions.append(cloned)
            by_symbol[key] = cloned
    opposite_re = re.compile(
        rf"\b(?P<lhs>{SYMBOL_PATTERN})\s*=\s*-\s*(?P<rhs>{SYMBOL_PATTERN})\s*=\s*"
        rf"(?P<number>{NUMBER_PATTERN})\s*(?P<unit>{UNIT_CAPTURE}){UNIT_END}",
        flags=re.IGNORECASE,
    )
    for match in opposite_re.finditer(canonical_question):
        value = parse_number(match.group("number"))
        unit = normalize_unit(match.group("unit"))
        info = unit_info(unit) if unit else None
        if value is None or unit is None or info is None:
            continue
        for symbol, signed_value in ((match.group("lhs"), -value), (match.group("rhs"), value)):
            clean = _clean_symbol(symbol)
            if not clean:
                continue
            key = clean.lower()
            existing = by_symbol.get(key)
            if existing and existing.unit == unit and abs(float(existing.value) - signed_value) <= max(1e-15, abs(signed_value) * 1e-12):
                continue
            cloned = Quantity(
                raw_text=match.group(0),
                value=signed_value,
                unit=unit,
                raw_unit=match.group("unit"),
                symbol=clean,
                dimension=info.dimension,
                span=match.span(),
                context=_context(canonical_question, match.span()),
                confidence=0.9,
            )
            additions.append(cloned)
            by_symbol[key] = cloned
    if not additions:
        return quantities
    return sorted([*quantities, *additions], key=lambda q: q.span or (0, 0))


def expand_grouped_numeric_equalities(canonical_question: str, quantities: List[Quantity]) -> List[Quantity]:
    """Clone grouped numeric facts such as ``qA and qB, both equal to 3 C``.

    This covers a common structural phrasing where the numeric value is written
    once for a set of like quantities. The pass only duplicates an explicit
    value/unit already present in the question and only for symbols in the
    grouped phrase.
    """

    existing = {
        str(quantity.symbol).lower()
        for quantity in quantities
        if quantity.symbol and quantity.dimension and unit_info(quantity.unit) is not None
    }
    additions: list[Quantity] = []
    group_re = re.compile(
        rf"\b(?P<symbols>{SYMBOL_PATTERN}(?:\s*,\s*{SYMBOL_PATTERN})*(?:\s*,?\s+and\s+{SYMBOL_PATTERN}))"
        rf"\s*,?\s*(?:are\s+)?(?:both\s+|all\s+)?(?:equal\s+to|each\s+equal\s+to|of\s+magnitude)\s*"
        rf"(?P<number>{NUMBER_PATTERN})\s*(?P<unit>{UNIT_CAPTURE}){UNIT_END}",
        flags=re.IGNORECASE,
    )
    for match in group_re.finditer(canonical_question):
        value = parse_number(match.group("number"))
        unit = normalize_unit(match.group("unit"))
        info = unit_info(unit) if unit else None
        if value is None or unit is None or info is None:
            continue
        symbols = [
            _clean_symbol(symbol)
            for symbol in re.split(r"\s*,\s*|\s+and\s+", match.group("symbols"), flags=re.IGNORECASE)
        ]
        for symbol in [symbol for symbol in symbols if symbol]:
            key = symbol.lower()
            if key in existing:
                continue
            additions.append(
                Quantity(
                    raw_text=match.group(0),
                    value=value,
                    unit=unit,
                    raw_unit=match.group("unit"),
                    symbol=symbol,
                    dimension=info.dimension,
                    span=match.span(),
                    context=_context(canonical_question, match.span()),
                    confidence=0.92,
                )
            )
            existing.add(key)
    if not additions:
        return quantities
    return sorted([*quantities, *additions], key=lambda q: q.span or (0, 0))


def extract_symbolic_relations(canonical_question: str, numeric_quantities: List[Quantity]) -> List[SymbolicRelation]:
    numeric_symbols = {q.symbol for q in numeric_quantities if q.symbol}
    relations: List[SymbolicRelation] = []
    seen: set[Tuple[int, int]] = set()
    for match in RELATION_RE.finditer(canonical_question):
        span = match.span()
        lhs = _clean_symbol(match.group("lhs"))
        rhs = _trim_relation_rhs(match.group("rhs").strip())
        if not lhs or not rhs:
            continue
        if lhs in numeric_symbols:
            continue
        if _is_plain_numeric_rhs(rhs) and not _is_compound_numeric_assignment_symbol(lhs):
            continue
        if span in seen:
            continue
        seen.add(span)
        raw_text = f"{lhs} = {rhs}"
        relation_start = match.start()
        relation_span = (relation_start, relation_start + len(raw_text))
        relations.append(
            SymbolicRelation(
                raw_text=raw_text,
                lhs=lhs,
                rhs=rhs,
                span=relation_span,
                context=_context(canonical_question, relation_span),
            )
        )
    return relations


def extract_numeric_constants(canonical_question: str, numeric_quantities: List[Quantity]) -> List[NumericConstant]:
    numeric_spans = {q.span for q in numeric_quantities if q.span is not None}
    constants: List[NumericConstant] = []
    seen: set[Tuple[int, int]] = set()
    for match in UNITLESS_ASSIGNMENT_RE.finditer(canonical_question):
        span = match.span()
        if _inside_existing_span(span, numeric_spans):
            continue
        symbol = _clean_symbol(match.group("symbol"))
        value = parse_number(match.group("number"))
        inferred_dimension = _infer_symbol_dimension(symbol or "", canonical_question) if symbol else None
        if not symbol or value is None:
            continue
        if not _is_symbol_candidate(symbol) and inferred_dimension is None:
            continue
        if _is_compound_numeric_assignment_symbol(symbol):
            continue
        if span in seen:
            continue
        seen.add(span)
        constants.append(
            NumericConstant(
                raw_text=match.group(0),
                symbol=symbol,
                value=value,
                dimension=inferred_dimension,
                span=span,
                context=_context(canonical_question, span),
            )
        )
    return constants


def _is_compound_numeric_assignment_symbol(symbol: str) -> bool:
    """Reject equation-like left sides such as LCω2 = 1 as hidden-unit quantities."""

    compact = re.sub(r"[_\d\s]+", "", symbol or "")
    lowered = compact.lower()
    if not compact:
        return True
    if any(marker in lowered for marker in ["ω", "omega"]):
        return len(compact) > 1
    allowed_compound_symbols = {"xl", "xc", "emf", "epsr", "epsilonr"}
    if lowered in allowed_compound_symbols:
        return False
    uppercase_count = sum(1 for char in compact if char.isupper())
    if uppercase_count >= 2:
        return True
    if re.fullmatch(r"[rlc]{2,}", lowered):
        return True
    return False


def extract_symbolic_quantities(
    canonical_question: str,
    numeric_quantities: List[Quantity],
    symbolic_relations: List[SymbolicRelation],
) -> List[SymbolicQuantity]:
    numeric_symbols = {q.symbol for q in numeric_quantities if q.symbol}
    symbolic: List[SymbolicQuantity] = []
    seen_symbols: set[str] = set()

    def add(raw_text: str, symbol: str, span: Tuple[int, int], raw_unit: Optional[str] = None, confidence: float = 0.75):
        clean = _clean_symbol(symbol)
        if not clean or clean in numeric_symbols or clean in seen_symbols or not _is_symbol_candidate(clean):
            return
        if _overlaps_more_specific_numeric_symbol(clean, span, numeric_quantities):
            return
        unit = normalize_unit(raw_unit or "") if raw_unit else None
        info = unit_info(unit) if unit else None
        seen_symbols.add(clean)
        symbolic.append(
            SymbolicQuantity(
                raw_text=raw_text,
                symbol=clean,
                unit=unit,
                raw_unit=raw_unit,
                dimension=info.dimension if info else _infer_symbol_dimension(clean, canonical_question),
                span=span,
                context=_context(canonical_question, span),
                confidence=confidence,
            )
        )

    for match in SYMBOL_UNIT_RE.finditer(canonical_question):
        add(match.group(0), match.group("symbol"), match.span(), match.group("unit"), 0.88)

    for match in QUOTED_SYMBOL_RE.finditer(canonical_question):
        add(match.group(0), match.group("symbol"), match.span(), None, 0.82)

    for match in PHRASE_SYMBOL_RE.finditer(canonical_question):
        add(match.group(0), match.group("symbol"), match.span(), None, 0.78)

    if re.search(r"\b(?:electric field|field strength|field intensity|field line)\b", canonical_question, re.IGNORECASE):
        for match in FIELD_SYMBOL_RE.finditer(canonical_question):
            add(match.group(0), match.group("symbol").replace(" ", "_"), match.span(), None, 0.82)

    for relation in symbolic_relations:
        if relation.span:
            add(relation.lhs, relation.lhs, (relation.span[0], relation.span[0] + len(relation.lhs)), None, 0.8)
        for symbol in re.findall(rf"(?<![{SYMBOL_CHARS}_])([{SYMBOL_CHARS}][{SYMBOL_CHARS}0-9_]*)", relation.rhs):
            if symbol.lower() in _SYMBOL_STOPWORDS:
                continue
            rhs_start = canonical_question.find(symbol, relation.span[0] if relation.span else 0)
            span = (rhs_start, rhs_start + len(symbol)) if rhs_start >= 0 else relation.span or (0, 0)
            add(symbol, symbol, span, None, 0.72)

    for match in BARE_SYMBOL_RE.finditer(canonical_question):
        symbol = match.group("symbol")
        if symbol.lower() in _SYMBOL_STOPWORDS:
            continue
        if _bare_symbol_is_geometry_label(canonical_question, match):
            continue
        add(symbol, symbol, match.span(), None, 0.58)

    return sorted(symbolic, key=lambda q: q.span or (0, 0))


def extract_concepts(canonical_question: str) -> List[str]:
    return sorted(
        concept_id
        for concept_id, pattern in CONCEPT_PATTERNS.items()
        if pattern.search(canonical_question)
    )


def extract_target_hints(canonical_question: str) -> List[str]:
    hints: List[str] = []
    seen: set[str] = set()
    for pattern in TARGET_PATTERNS:
        for match in pattern.finditer(canonical_question):
            target = re.split(r"\b(?:if|given|when|under|with)\b", match.group("target"), maxsplit=1, flags=re.IGNORECASE)[0]
            target = target.strip(" ,;:")
            if target and target.lower() not in seen:
                seen.add(target.lower())
                hints.append(target)
    return hints[:4]


def extract_entities(canonical_question: str, quantities: List[Quantity], symbolic_quantities: List[SymbolicQuantity]) -> List[Entity]:
    entities: List[Entity] = []
    seen: set[str] = set()

    def add(entity_type: str, label: str, symbol: Optional[str], span: Optional[Tuple[int, int]], confidence: float) -> None:
        symbol_clean = _clean_symbol(symbol)
        key_base = symbol_clean or label.lower().replace(" ", "_")
        entity_id = f"{entity_type}:{key_base}".lower()
        if entity_id in seen:
            return
        seen.add(entity_id)
        entities.append(
            Entity(
                entity_id=entity_id,
                label=label,
                entity_type=entity_type,
                symbol=symbol_clean,
                span=span,
                context=_context(canonical_question, span) if span else "",
                confidence=confidence,
            )
        )

    for quantity in quantities:
        entity_type = _entity_type_for_dimension(quantity.dimension)
        if entity_type and quantity.symbol:
            add(entity_type, quantity.raw_text, quantity.symbol, quantity.span, 0.86)
    for quantity in symbolic_quantities:
        entity_type = _entity_type_for_dimension(quantity.dimension)
        if entity_type:
            add(entity_type, quantity.raw_text, quantity.symbol, quantity.span, 0.74)
    for entity_type, label, pattern in ENTITY_PATTERNS:
        for match in pattern.finditer(canonical_question):
            symbol = match.groupdict().get("symbol")
            add(entity_type, match.group(0), symbol, match.span(), 0.72 if symbol else 0.62)
    return sorted(entities, key=lambda entity: entity.span or (10**9, 10**9))


def extract_states(canonical_question: str) -> List[State]:
    """Extract state scopes so repeated symbols can be interpreted temporally."""

    states: List[State] = [
        State(
            state_id="state:base",
            label="base",
            trigger_text="default problem context",
            span=None,
            confidence=0.6,
        )
    ]
    seen: set[Tuple[str, Tuple[int, int]]] = set()
    label_counts: dict[str, int] = {}
    for label, pattern in STATE_PATTERNS:
        for match in pattern.finditer(canonical_question):
            key = (label, match.span())
            if key in seen:
                continue
            seen.add(key)
            label_counts[label] = label_counts.get(label, 0) + 1
            suffix = "" if label_counts[label] == 1 else f":{label_counts[label]}"
            states.append(
                State(
                    state_id=f"state:{label}{suffix}",
                    label=label,
                    trigger_text=match.group(0).strip(),
                    span=match.span(),
                    confidence=0.78 if label != "conditional" else 0.68,
                )
            )
    return sorted(states, key=lambda state: state.span or (-1, -1))


def extract_events(canonical_question: str) -> List[Event]:
    events: List[Event] = []
    seen: set[Tuple[str, Tuple[int, int]]] = set()
    counts: dict[str, int] = {}
    for event_type, pattern in EVENT_PATTERNS:
        for match in pattern.finditer(canonical_question):
            key = (event_type, match.span())
            if key in seen:
                continue
            seen.add(key)
            counts[event_type] = counts.get(event_type, 0) + 1
            suffix = "" if counts[event_type] == 1 else f":{counts[event_type]}"
            events.append(
                Event(
                    event_id=f"event:{event_type}{suffix}",
                    event_type=event_type,
                    trigger_text=match.group(0).strip(),
                    span=match.span(),
                    confidence=0.76,
                )
            )
    return sorted(events, key=lambda event: event.span or (10**9, 10**9))


def assign_quantity_contexts(quantities: List[Quantity], entities: List[Entity], states: List[State]) -> List[Quantity]:
    grounded: List[Quantity] = []
    for quantity in quantities:
        grounded.append(
            replace(
                quantity,
                entity_id=_entity_for_quantity(quantity, entities),
                state_id=_state_for_quantity(quantity, states),
                role="inferred_hidden_unit" if quantity.raw_unit == "implicit_base_SI" else "given",
            )
        )
    return grounded


def _state_for_quantity(quantity: Quantity, states: List[State]) -> str:
    if not quantity.span:
        return "state:base"
    start, _ = quantity.span
    explicit = [state for state in states if state.span and state.span[0] <= start <= state.span[1]]
    if explicit:
        return explicit[-1].state_id
    preceding = [state for state in states if state.span and state.span[1] <= start]
    if not preceding:
        return "state:base"
    return max(preceding, key=lambda state: state.span[1] if state.span else -1).state_id


def _entity_for_quantity(quantity: Quantity, entities: List[Entity]) -> Optional[str]:
    symbol = quantity.symbol or ""
    if symbol:
        for entity in entities:
            if entity.symbol and entity.symbol == symbol:
                return entity.entity_id
        entity_type = _entity_type_for_dimension(quantity.dimension)
        if entity_type:
            for entity in entities:
                if entity.symbol and entity.entity_type == entity_type and entity.symbol.lower() == symbol.lower():
                    return entity.entity_id
    entity_type = _entity_type_for_dimension(quantity.dimension)
    if entity_type:
        candidates = [entity for entity in entities if entity.entity_type == entity_type]
        if len(candidates) == 1:
            return candidates[0].entity_id
    return None


def extract_relations(canonical_question: str, concepts: List[str]) -> List[Relation]:
    relations: List[Relation] = []
    patterns = [
        ("topology", "series", r"\bseries\b"),
        ("topology", "parallel", r"\bparallel\b(?!\s*[- ]?\s*plates?\b)"),
        ("topology", "balanced_bridge", r"\bbalanced\s+Wheatstone\b|\bWheatstone\b.*\bbalanced\b"),
        ("geometry", "midpoint", r"\bmidpoint\b"),
        ("geometry", "perpendicular_bisector", r"\bperpendicular bisector\b"),
        (
            "geometry",
            "collinear",
            r"\b(?:collinear|same\s+straight\s+line|same\s+line|straight\s+line\s+passing\s+through|"
            r"along\s+the\s+same\s+straight\s+line|opposite\s+sides\s+of|on\s+opposite\s+sides|line\s+segment)\b",
        ),
        ("geometry", "equilateral_triangle", r"\b(?:equilateral|regular)\s+triangle\b"),
        (
            "geometry",
            "right_isosceles_triangle",
            r"\bright isosceles triangle\b|\bisosceles right triangle\b|\bright[- ]?angled isosceles triangle\b",
        ),
        ("state", "resonance", r"\bresonan\w*\b"),
    ]
    for relation_type, qualifier, pattern in patterns:
        for match in re.finditer(pattern, canonical_question, flags=re.IGNORECASE):
            relations.append(
                Relation(
                    relation_type=relation_type,
                    subject="question",
                    qualifier=qualifier,
                    span=match.span(),
                    evidence=match.group(0),
                    confidence=0.82,
                )
            )
    for concept in concepts:
        if concept in {"rlc_circuit", "lc_circuit", "parallel_circuit"} and not any(r.qualifier == concept for r in relations):
            relations.append(Relation("concept", "question", qualifier=concept, evidence=concept, confidence=0.68))
    return relations


def build_topology_graph(canonical_question: str, entities: List[Entity], relations: List[Relation], quantities: List[Quantity]) -> TopologyGraph:
    """Build a conservative circuit topology sketch for verifier gating.

    This is not yet a full KCL/KVL circuit graph. Its job is to expose when a
    problem contains multiple components/branches/segments so direct scalar
    formulas cannot silently bind unrelated quantities.
    """

    text = canonical_question.lower()
    circuit_cues = ("circuit", "resistor", "capacitor", "inductor", "branch", "node", "series", "parallel", "battery")
    if not any(cue in text for cue in circuit_cues):
        return TopologyGraph()

    component_types = {"resistor", "capacitor", "inductor", "source", "branch", "node"}
    nodes: List[dict] = []
    seen_nodes: set[str] = set()
    entity_types_with_symbols = {entity.entity_type for entity in entities if entity.symbol}
    for entity in entities:
        if entity.entity_type not in component_types:
            continue
        if entity.symbol is None and entity.entity_type in entity_types_with_symbols:
            continue
        if entity.entity_id in seen_nodes:
            continue
        seen_nodes.add(entity.entity_id)
        nodes.append(
            {
                "id": entity.entity_id,
                "type": entity.entity_type,
                "symbol": entity.symbol,
                "label": entity.label,
                "span": entity.span,
            }
        )

    for quantity in quantities:
        symbol = quantity.symbol
        dimension = quantity.dimension
        if not symbol or dimension not in {"resistance", "capacitance", "inductance", "voltage", "current", "power"}:
            continue
        node_id = quantity.entity_id or f"quantity:{symbol}".lower()
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "type": _entity_type_for_dimension(dimension) or "quantity",
                "symbol": symbol,
                "label": quantity.raw_text,
                "span": quantity.span,
            }
        )

    edges: List[dict] = []
    for relation in relations:
        if relation.relation_type != "topology":
            continue
        edges.append(
            {
                "source": relation.subject,
                "target": relation.object or "question",
                "type": relation.qualifier,
                "evidence": relation.evidence,
                "span": relation.span,
            }
        )

    explicit_component_symbols = [node.get("symbol") for node in nodes if node.get("symbol")]
    numbered_component_count = sum(1 for symbol in explicit_component_symbols if re.search(r"\d+$", str(symbol)))
    branch_like = bool(
        re.search(r"\b(branch|node|across\s+[A-Z]{2}|u[A-Z]{2}|U_[A-Z]{2})\b", canonical_question)
        or re.search(r"\bcircuit\s+segment\b", text)
    )
    relation_types = {edge.get("type") for edge in edges}
    topology_relation = any(edge_type in {"series", "parallel", "balanced_bridge"} for edge_type in relation_types)
    component_count = sum(1 for node in nodes if node.get("type") in {"resistor", "capacitor", "inductor", "branch", "node"})
    is_complex = component_count >= 2 or numbered_component_count >= 2 or branch_like or (topology_relation and component_count >= 1)

    ambiguity: List[str] = []
    if is_complex and not edges:
        ambiguity.append("components_present_without_canonical_connections")
    if branch_like:
        ambiguity.append("branch_or_segment_language_requires_topology_grounding")
    if numbered_component_count >= 2 and not topology_relation:
        ambiguity.append("multiple_indexed_components")

    simple_relation = (relation_types & {"series", "parallel"}) if not branch_like and "balanced_bridge" not in relation_types else set()
    if len(simple_relation) == 1 and not ambiguity:
        canonical_form = f"{next(iter(simple_relation))}_topology"
        is_complex = False
    elif not nodes and not edges:
        canonical_form = "circuit_cues_without_grounded_components"
    elif is_complex:
        canonical_form = "complex_circuit_topology_unresolved"
    elif nodes:
        canonical_form = "single_component_or_global_circuit"
    else:
        canonical_form = "topology_relation_only"
    return TopologyGraph(nodes=nodes, edges=edges, canonical_form=canonical_form, is_complex=is_complex, ambiguity=sorted(set(ambiguity)))


def extract_constraints(canonical_question: str, concepts: List[str], relations: List[Relation]) -> List[Constraint]:
    constraints: List[Constraint] = []
    relation_qualifiers = {relation.qualifier for relation in relations}

    def add(constraint_id: str, kind: str, expression: str, source: str, span: Optional[Tuple[int, int]] = None, confidence: float = 0.76) -> None:
        if any(existing.constraint_id == constraint_id for existing in constraints):
            return
        constraints.append(Constraint(constraint_id, kind, expression, source, span, confidence))

    if "resonance" in relation_qualifiers or "resonance" in concepts:
        add("resonance_xl_equals_xc", "state", "XL = XC", "resonance")
    if "series" in relation_qualifiers:
        add("series_current_same", "topology", "I_same_in_series", "series")
    if "parallel" in relation_qualifiers:
        add("parallel_voltage_same", "topology", "U_same_in_parallel", "parallel")
    if re.search(r"\bideal\b", canonical_question, flags=re.IGNORECASE):
        add("ideal_component_assumption", "assumption", "idealized_components", "ideal")
    if re.search(r"\b(in air|in vacuum)\b", canonical_question, flags=re.IGNORECASE):
        match = re.search(r"\b(in air|in vacuum)\b", canonical_question, flags=re.IGNORECASE)
        add("medium_air_or_vacuum", "medium", "epsilon_r = 1", match.group(0), match.span(), 0.82)
    return constraints


def extract_goals(canonical_question: str, target_hints: List[str], symbolic_quantities: List[SymbolicQuantity]) -> List[Goal]:
    goals: List[Goal] = []
    for hint in target_hints:
        for fragment in _split_multi_target_hint(hint):
            symbol = _symbol_for_goal_text(fragment, symbolic_quantities)
            dimension = _dimension_for_goal_text(fragment)
            if (
                dimension is None
                and symbol
                and symbol.lower().startswith("e")
                and re.search(r"\b(?:electric field|field strength|field intensity|field line)\b", canonical_question, re.IGNORECASE)
            ):
                dimension = "electric_field"
            goals.append(
                Goal(
                    goal_id=f"goal:{len(goals)+1}",
                    text=fragment,
                    dimension=dimension,
                    symbol=symbol,
                    span=_span_for_text(canonical_question, fragment) or _span_for_text(canonical_question, hint),
                    confidence=0.78,
                )
            )
    if not goals and TARGET_REQUEST_CUE.search(canonical_question):
        for quantity in symbolic_quantities[:2]:
            if quantity.dimension:
                goals.append(
                    Goal(
                        goal_id=f"goal:{len(goals)+1}",
                        text=quantity.raw_text,
                        dimension=quantity.dimension,
                        symbol=quantity.symbol,
                        span=quantity.span,
                        confidence=0.62,
                    )
                )
    return goals


def infer_answer_type_hint(
    canonical_question: str,
    quantities: List[Quantity],
    symbolic_quantities: List[SymbolicQuantity],
    symbolic_relations: List[SymbolicRelation],
    concepts: List[str],
) -> str:
    q = canonical_question.lower()
    target_text = " ".join(extract_target_hints(canonical_question)).lower()
    numeric_target_terms = [
        "energy",
        "charge",
        "current",
        "voltage",
        "resistance",
        "capacitance",
        "reactance",
        "power factor",
        "power",
        "potential difference",
        "impedance",
        "field",
        "force",
        "mass",
        "flux",
        "frequency",
        "fraction",
        "percentage",
        "angle",
        "distance",
        "separation",
        "length",
        "period",
        "inductance",
        "uncertainty",
        "error",
        "dielectric constant",
        "relative permittivity",
    ]
    multi_output_markers = [
        " and ",
        " and the ",
        " and its ",
        " respectively",
        ";",
    ]
    if re.search(r"\b(shape of the graph|graph representing)\b", q):
        return "conceptual"
    if re.search(r"\b(?:which|what)\s+(?:of\s+the\s+following\s+)?quantit(?:y|ies)\b", q) and any(
        cue in q for cue in ["proportional", "depend", "linearly", "square of"]
    ):
        return "conceptual"
    if re.search(r"\b(?:directly|inversely)\s+proportional\b", q) and re.search(r"\b(?:which|what)\b", q):
        return "conceptual"
    if (
        canonical_question.strip().endswith("?")
        and re.match(r"\s*(?:is|are|does|do|can|will|would|should)\b", q)
        and any(cue in q for cue in ["resonant frequency", "resonance", "resonate", "true", "holds"])
    ):
        return "yes_no"
    if re.search(r"\bis\s+(?:approximately\s+|about\s+)?[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:hz|rad/s)\s+the\s+resonant\s+frequency\b", q):
        return "yes_no"
    if _looks_like_numeric_rlc_frequency_transform(q, target_text) and quantities:
        return "numeric"
    if _looks_like_qualitative_change_question(q, target_text):
        return "conceptual"
    if target_text and any(marker in target_text for marker in multi_output_markers) and not _looks_like_single_target_hint(target_text):
        if sum(1 for term in numeric_target_terms if _target_term_present(target_text, term)) >= 2:
            return "multi_output"
        if "each" in target_text and "total" in target_text:
            return "multi_output"
    if target_text and "charge and energy" in target_text:
        return "multi_output"
    if target_text and "magnitude" in target_text and "direction" in target_text and any(term in target_text for term in ["force", "field"]):
        return "numeric"
    if target_text and "direction" in target_text and any(term in target_text for term in ["force", "field"]):
        return "conceptual"
    if _looks_like_symmetry_zero_vector_query(q, target_text):
        return "numeric"
    if re.search(r"\b(?:determine|check|decide)\s+if\b|\bwhether\b", q) and any(
        cue in q for cue in ["resonance", "resonate", "occurs", "holds", "true"]
    ):
        return "yes_no"
    if re.search(
        r"\bwhat\s+charge\s+must\b|\bcharge\s+must\s+be\s+placed\b|"
        r"\bwhat\s+value\s+(?:of\s+)?charge\b|\bwhat\s+value\s+(?:must|should|can|would|will)\s+charge\b",
        q,
    ):
        return "symbolic" if symbolic_relations or symbolic_quantities else "numeric"
    if re.search(r"\b(relationship|expression|formula|in terms of|derive)\b", q):
        return "symbolic"
    numeric_query_question = re.search(
        r"\b(?:calculate|find|determine|compute|how much|"
        r"what\s+(?:capacitance|inductance|resistance|voltage|potential\s+difference|current|power|energy|frequency|fraction|percentage|period|force|field|mass|angle|distance|length|separation|dielectric\s+constant|relative\s+permittivity|random\s+error|absolute\s+error)|"
        r"what\s+(?:will|would|should|can)\s+be\s+[^.?]*(?:capacitance|inductance|resistance|voltage|potential\s+difference|current|power|energy|frequency|fraction|percentage|period|force|field|mass|angle|distance|length|separation|dielectric\s+constant|relative\s+permittivity|random\s+error|absolute\s+error)|"
        r"(?:needed|required)\s+to)\b",
        q,
    )
    if numeric_query_question and quantities:
        return "numeric"
    if (
        target_text
        and not re.search(r"\b(unit of|si unit|what is the unit)\b", q)
        and any(_target_term_present(target_text, term) for term in numeric_target_terms)
        and (symbolic_relations or symbolic_quantities or _has_symbolic_geometry_context(q))
    ):
        return "symbolic"
    yes_no_question = (
        canonical_question.strip().endswith("?")
        and not re.match(r"\s*(?:what|where|when|why|how|which)\b", q)
        and " what " not in f" {q} "
        and " which " not in f" {q} "
        and not numeric_query_question
        and not (target_text and any(_target_term_present(target_text, term) for term in numeric_target_terms))
        and (
            re.search(r"\b(?:does|do|is|are|can|will|would|should)\b", q)
            or re.search(r"\bdoes\s+[^?]*(?:occur|resonate|hold|remain)\b", q)
        )
    )
    if re.search(r"\b(yes or no|true or false)\b", q) or yes_no_question:
        return "yes_no"
    if (symbolic_relations or symbolic_quantities) and re.search(r"\b(determine|calculate|find|what is)\b", q):
        return "symbolic"
    if (
        re.search(r"\b(unit of|si unit|what happens|how does|how will|shape of the graph)\b", q)
        or re.match(r"\s*(?:where|which)\b", q)
    ):
        return "conceptual"
    if quantities:
        return "numeric"
    if symbolic_relations or symbolic_quantities:
        return "symbolic"
    if concepts:
        return "conceptual"
    return "unknown"


def _looks_like_qualitative_change_question(question_text: str, target_text: str) -> bool:
    haystack = f"{target_text} {question_text}".lower()
    physical_terms = [
        "capacitance",
        "charge",
        "current",
        "voltage",
        "energy",
        "power",
        "force",
        "field",
        "reactance",
        "impedance",
        "frequency",
        "brightness",
    ]
    change_terms = ["change", "changes", "happen", "vary", "increase", "decrease", "factor", "times"]
    question_cues = [
        "how do",
        "how does",
        "how will",
        "how would",
        "how many times",
        "by what factor",
        "describe how",
        "what happens",
        "what changes",
    ]
    return (
        any(cue in haystack for cue in question_cues)
        and any(term in haystack for term in physical_terms)
        and any(term in haystack for term in change_terms)
    )


def _looks_like_single_target_hint(target_text: str) -> bool:
    """Avoid treating object descriptions as multiple requested outputs.

    "force exerted by q1 and q2 on q0" contains multiple charge mentions, but
    the requested target is still one resultant force. Multi-output detection
    must key off multiple requested quantities, not every "and" in the noun
    phrase.
    """

    text = str(target_text or "").lower()
    single_target_heads = (
        "force",
        "net force",
        "resultant force",
        "electric force",
        "electric field",
        "field strength",
        "electric field strength",
        "field intensity",
    )
    if not any(head in text for head in single_target_heads):
        return False
    if re.search(
        r"\b(?:current|voltage|resistance|capacitance|power|energy|frequency|period|flux|inductance)\b",
        text,
    ):
        return False
    return bool(
        re.search(r"\b(?:exerted|caused|produced)\s+by\b", text)
        or re.search(r"\bacting\s+on\b", text)
        or re.search(r"\bat\s+(?:point\s+)?[a-z]\b", text)
        or "resultant" in text
        or "net " in text
        or "magnitude of" in text
        or "vector at" in text
    )


def _looks_like_symmetry_zero_vector_query(question_text: str, target_text: str) -> bool:
    haystack = f"{target_text} {question_text}".lower()
    if not any(term in haystack for term in ["electric field", "force"]):
        return False
    if not any(cue in haystack for cue in ["center", "centre", "centroid"]):
        return False
    if not any(cue in haystack for cue in ["square", "equilateral triangle", "regular triangle"]):
        return False
    return bool(
        re.search(r"\b(?:identical|equal|same)\s+(?:positive\s+|negative\s+)?charges\b", haystack)
        or re.search(r"\b(?:all|three|four)\s+(?:identical|equal)\b", haystack)
    )


def _looks_like_numeric_rlc_frequency_transform(question_text: str, target_text: str) -> bool:
    haystack = f"{target_text} {question_text}".lower()
    if not any(cue in haystack for cue in ["rlc", "reactance", "ac circuit"]):
        return False
    if not has_frequency_transform_cue(haystack):
        return False
    return any(cue in haystack for cue in ["current", "impedance", "phase", "angle", "factor", "times", "resonance", "resonant", "resonate"])


def _has_symbolic_geometry_context(question_text: str) -> bool:
    return bool(
        re.search(r"\b(?:triangle|square|rectangle|vertices?|corners?|point\s+[A-Z]|side\s+length|distance\s+[a-z])\b", question_text)
        or re.search(r"\b[A-Z]{2}\s*=", question_text)
    )


def _split_multi_target_hint(hint: str) -> list[str]:
    lowered = hint.lower()
    if "equidistant from" in lowered:
        return [hint]
    target_terms = [
        "capacitance",
        "charge",
        "current",
        "electric field",
        "energy",
        "field",
        "flux",
        "force",
        "frequency",
        "impedance",
        "inductance",
        "period",
        "power factor",
        "power",
        "reactance",
        "resistance",
        "voltage",
    ]
    if " and " not in lowered and ";" not in lowered and "," not in lowered:
        return [hint]
    if sum(1 for term in target_terms if _target_term_present(lowered, term)) < 2:
        return [hint]
    for separator in [";", " respectively ", " and ", ","]:
        if separator in lowered:
            parts = [part.strip(" .?") for part in re.split(re.escape(separator), hint, flags=re.IGNORECASE) if part.strip(" .?")]
            return parts or [hint]
    return [hint]


_SYMBOL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "charge",
    "distance",
    "electric",
    "energy",
    "field",
    "force",
    "in",
    "i",
    "is",
    "it",
    "length",
    "magnetic",
    "m",
    "meters",
    "placed",
    "points",
    "s",
    "stored",
    "t",
    "where",
    "with",
}


def _overlaps_more_specific_numeric_symbol(symbol: str, span: Tuple[int, int], numeric_quantities: List[Quantity]) -> bool:
    """Avoid creating a shorter symbolic alias from a richer numeric symbol.

    Example: ``charge q′ = -1 μC`` should produce the numeric fact ``q′``,
    not a second symbolic charge ``q`` clipped before the prime marker.
    """

    for quantity in numeric_quantities:
        if not quantity.span or not quantity.symbol:
            continue
        if not _spans_overlap(span, quantity.span):
            continue
        if str(quantity.symbol).startswith(symbol) and str(quantity.symbol) != symbol:
            return True
    return False


def _bare_symbol_is_geometry_label(text: str, match: re.Match[str]) -> bool:
    """Do not ground standalone point labels as physical quantities.

    A single uppercase label in ``vertices P, Q, and R`` is a geometry point,
    while ``charge Q`` is a charge symbol. The distinction is made from local
    context rather than a fixed label list.
    """

    symbol = match.group("symbol")
    if not symbol or len(symbol) != 1 or not symbol.isupper():
        return False
    start, end = match.span()
    local = text[max(0, start - 28) : min(len(text), end + 28)].lower()
    if "charge" in local or "current" in local or "voltage" in local:
        return False
    if any(cue in local for cue in ["vertices", "vertex", "point", "triangle", "square", "rectangle", "corner"]):
        return True
    geometry_text = re.search(r"\b(?:triangle|square|rectangle|vertices?|corners?)\b", text, flags=re.IGNORECASE)
    return bool(geometry_text and re.search(r"\b(?:at|on|from|to|toward|towards)\s+(?:point\s+)?[A-Z]\b", local))


def _spans_overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _trim_relation_rhs(rhs: str) -> str:
    rhs = re.split(r"\s+and\s+(?=[A-Za-z][A-Za-z0-9_]*\s*=)", rhs, maxsplit=1, flags=re.IGNORECASE)[0]
    rhs = re.split(r"\s+\((?:with|where)\b", rhs, maxsplit=1, flags=re.IGNORECASE)[0]
    rhs = re.split(
        r"\s+(?=(?:are|is|was|were)\s+(?:placed|located|set|put|fixed|arranged)\b)",
        rhs,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    rhs = re.split(r"\)\s+(?=(?:are|is|was|were|at|in|on|with|respectively)\b)", rhs, maxsplit=1, flags=re.IGNORECASE)[0]
    if rhs.endswith(")") and rhs.count("(") < rhs.count(")"):
        rhs = rhs[:-1]
    return rhs.strip()


def _is_plain_numeric_rhs(rhs: str) -> bool:
    parts = rhs.split()
    if not parts:
        return True
    return parse_number(parts[0]) is not None and not re.search(r"[A-Za-z_]", rhs)


def _is_symbol_candidate(symbol: str) -> bool:
    lower = symbol.lower()
    if lower in _SYMBOL_STOPWORDS:
        return False
    if "cos" in lower or "sin" in lower:
        return False
    if len(symbol) == 1:
        return symbol.isalpha() or symbol in {"ω", "φ", "μ", "ε", "θ"}
    if "_" in symbol or any(ch.isdigit() for ch in symbol):
        return True
    if symbol.isupper() and len(symbol) <= 3:
        return True
    if re.fullmatch(r"[qQ][A-Za-z0-9_′']*", symbol):
        return True
    return False


def _infer_symbol_dimension(symbol: str, text: str) -> Optional[str]:
    s = symbol.lower()
    lowered = text.lower()
    if s.startswith("q"):
        return "charge"
    if s == "k":
        return "constant"
    if s in {"rho", "ρ"}:
        return "resistivity"
    if s in {"ω", "omega"}:
        return "angular_frequency"
    if s in {"φ", "phi"}:
        return "phase_angle"
    if s in {"z"}:
        return "impedance"
    if s in {"xl"}:
        return "inductive_reactance"
    if s in {"xc"}:
        return "capacitive_reactance"
    if re.fullmatch(r"[a-z]{2}", s) and "triangle" in lowered:
        return "length"
    if s in {"b"}:
        return "magnetic_field"
    if s in {"n"} and "solenoid" in lowered:
        return "turn_density"
    if s in {"n"} and any(cue in lowered for cue in ["carrier density", "number density", "drift"]):
        return "number_density"
    length_context = any(word in lowered for word in ["distance", "side length", "separation", "radius", "triangle", "square"])
    if s in {"a", "h", "d"} and length_context:
        return "length"
    if s == "area" or (s == "a" and "area" in lowered and not length_context):
        return "area"
    if s in {"ε", "epsilon"}:
        return "permittivity"
    if s in {"μ", "mu"}:
        return "permeability_or_prefix"
    if s in {"u", "v"} or (s.startswith("u") and "voltage" in lowered):
        return "voltage"
    if s in {"i"} or (s.startswith("i") and "current" in lowered):
        return "current"
    if s in {"r"} or (s.startswith("r") and "resistance" in lowered):
        return "resistance_or_distance"
    if s in {"c"} or (s.startswith("c") and "capacitance" in lowered):
        return "capacitance"
    if s in {"l"} or (s.startswith("l") and "inductance" in lowered):
        return "inductance"
    if s in {"a", "h", "d"}:
        return "length"
    if s.startswith("e"):
        if "electric field" in lowered or "field strength" in lowered or "field intensity" in lowered or "field line" in lowered:
            return "electric_field"
        if "energy" in lowered or "work" in lowered:
            return "energy"
        return "electric_field_or_energy"
    if s.startswith("f"):
        return "force_or_frequency"
    if s.startswith("w"):
        return "energy"
    return None


def _entity_type_for_dimension(dimension: Optional[str]) -> Optional[str]:
    return {
        "capacitance": "capacitor",
        "charge": "charge",
        "current": "current_path",
        "inductance": "inductor",
        "magnetic_field": "field",
        "resistance": "resistor",
        "resistance_or_distance": "resistor",
        "voltage": "source_or_node",
    }.get(dimension or "")


def _dimension_for_goal_text(text: str) -> Optional[str]:
    lowered = text.lower()
    if re.search(r"\bcharge\s+(?:q\d*|[a-z])?\s*(?:must|should|to\s+be|is\s+to\s+be|placed|have)\b", lowered):
        return "charge"
    if any(phrase in lowered for phrase in ["magnetic flux density", "flux density"]):
        return "magnetic_field"
    if any(phrase in lowered for phrase in ["electric field energy", "magnetic field energy", "field energy", "stored energy"]):
        return "energy"
    if any(phrase in lowered for phrase in ["electric field", "field strength", "field intensity"]):
        return "electric_field"
    if re.search(r"\bforce\b", lowered):
        return "force"
    if re.search(r"\bangle\b|\bphase angle\b", lowered):
        return "angle"
    for keywords, dimension in GOAL_DIMENSION_KEYWORDS:
        if any(_target_term_present(lowered, keyword) for keyword in keywords):
            return dimension
    return None


def _target_term_present(text: str, term: str) -> bool:
    term = str(term or "").lower()
    if not term:
        return False
    if " " in term or "/" in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _symbol_for_goal_text(text: str, symbolic_quantities: List[SymbolicQuantity]) -> Optional[str]:
    lowered = text.lower()
    goal_dimension = _dimension_for_goal_text(text)
    for quantity in sorted(symbolic_quantities, key=lambda item: len(item.symbol or ""), reverse=True):
        symbol = quantity.symbol or ""
        symbol_pattern = rf"(?<![A-Za-z0-9_]){re.escape(symbol.lower())}(?![A-Za-z0-9_])"
        if re.search(symbol_pattern, lowered):
            return quantity.symbol
    inferred = _symbol_from_goal_role(text, goal_dimension)
    if inferred:
        return inferred
    same_dimension = [quantity.symbol for quantity in symbolic_quantities if quantity.dimension and quantity.dimension == goal_dimension]
    same_dimension = [symbol for symbol in same_dimension if symbol]
    if len(set(same_dimension)) == 1:
        return same_dimension[0]
    return None


def _symbol_from_goal_role(text: str, goal_dimension: Optional[str]) -> Optional[str]:
    """Infer target symbols from role phrases instead of borrowing known facts."""

    if goal_dimension != "charge":
        return None
    placement = re.search(
        r"\bcharge\b[^.?;]{0,90}?\b(?:placed|put|located|set)\s+(?:at|on)\s+"
        r"(?:point\s+|vertex\s+|corner\s+)?([A-Za-z])\b",
        text,
        flags=re.IGNORECASE,
    )
    if placement:
        return f"q{placement.group(1).upper()}"
    direct = re.search(
        r"\bcharge\b(?:(?!\belectric\s+field\b|\bfield\b).){0,90}?\b(?:at|on)\s+"
        r"(?:point\s+|vertex\s+|corner\s+)?([A-Za-z])\b",
        text,
        flags=re.IGNORECASE,
    )
    if direct:
        return f"q{direct.group(1).upper()}"
    return None


def _span_for_text(canonical_question: str, text: str) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    start = canonical_question.lower().find(text.lower())
    return (start, start + len(text)) if start >= 0 else None


def _inside_existing_span(span: Tuple[int, int], spans: Iterable[Tuple[int, int]]) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in spans)


def _warnings(
    canonical_question: str,
    quantities: List[Quantity],
    symbolic_quantities: List[SymbolicQuantity],
    symbolic_relations: List[SymbolicRelation],
    numeric_constants: List[NumericConstant],
    concepts: List[str],
) -> List[str]:
    warnings: List[str] = []
    unknown_unitish = re.findall(rf"(?<![A-Za-z0-9_]){NUMBER_PATTERN}\s+([A-Za-zμΩ/%°]+)", canonical_question)
    known_raw_units = {q.raw_unit for q in quantities}
    stopwords = {
        "/",
        "a",
        "an",
        "and",
        "are",
        "at",
        "by",
        "be",
        "calculate",
        "cos",
        "determine",
        "find",
        "for",
        "from",
        "holds",
        "if",
        "identical",
        "in",
        "is",
        "it",
        "must",
        "now",
        "of",
        "or",
        "out",
        "respectively",
        "should",
        "sin",
        "the",
        "to",
        "vertices",
        "voltages",
        "what",
        "when",
        "where",
        "which",
        "with",
        "x",
    }
    for raw_unit in unknown_unitish:
        if raw_unit.lower() in stopwords:
            continue
        if raw_unit not in known_raw_units and normalize_unit(raw_unit) is None:
            warnings.append(f"unrecognized_unit:{raw_unit}")
    if not quantities and not symbolic_quantities and not symbolic_relations and not numeric_constants and not concepts:
        warnings.append("no_quantities_extracted")
    return sorted(set(warnings))


def _parse_confidence(
    quantities: List[Quantity],
    symbolic_quantities: List[SymbolicQuantity],
    symbolic_relations: List[SymbolicRelation],
    numeric_constants: List[NumericConstant],
    concepts: List[str],
    warnings: List[str],
) -> float:
    confidence_sources = [q.confidence for q in quantities]
    confidence_sources.extend(q.confidence for q in symbolic_quantities)
    confidence_sources.extend(r.confidence for r in symbolic_relations)
    confidence_sources.extend(c.confidence for c in numeric_constants)
    if concepts:
        confidence_sources.append(0.72)
    if not confidence_sources:
        return 0.35
    base = min(confidence_sources)
    penalty = 0.08 * len(warnings)
    return max(0.0, min(1.0, base - penalty))


def normalize_question(question: str) -> NormalizedQuestion:
    canonical = canonicalize_question(question)
    quantities = extract_quantities(canonical)
    quantities = expand_chained_numeric_equalities(canonical, quantities)
    quantities = expand_grouped_numeric_equalities(canonical, quantities)
    symbolic_relations = extract_symbolic_relations(canonical, quantities)
    numeric_constants = extract_numeric_constants(canonical, quantities)
    hidden_unit_quantities = _quantities_from_numeric_constants(canonical, numeric_constants)
    if hidden_unit_quantities:
        quantities = sorted([*quantities, *hidden_unit_quantities], key=lambda q: q.span or (0, 0))
        quantities = expand_chained_numeric_equalities(canonical, quantities)
        quantities = expand_grouped_numeric_equalities(canonical, quantities)
    symbolic_quantities = extract_symbolic_quantities(canonical, quantities, symbolic_relations)
    concepts = extract_concepts(canonical)
    target_hints = extract_target_hints(canonical)
    states = extract_states(canonical)
    events = extract_events(canonical)
    entities = extract_entities(canonical, quantities, symbolic_quantities)
    quantities = assign_quantity_contexts(quantities, entities, states)
    relations = extract_relations(canonical, concepts)
    topology_graph = build_topology_graph(canonical, entities, relations, quantities)
    constraints = extract_constraints(canonical, concepts, relations)
    goals = extract_goals(canonical, target_hints, symbolic_quantities)
    answer_type_hint = infer_answer_type_hint(canonical, quantities, symbolic_quantities, symbolic_relations, concepts)
    warnings = _warnings(canonical, quantities, symbolic_quantities, symbolic_relations, numeric_constants, concepts)
    return NormalizedQuestion(
        raw_question=str(question or ""),
        canonical_question=canonical,
        quantities=quantities,
        parse_confidence=_parse_confidence(
            quantities,
            symbolic_quantities,
            symbolic_relations,
            numeric_constants,
            concepts,
            warnings,
        ),
        symbolic_quantities=symbolic_quantities,
        symbolic_relations=symbolic_relations,
        numeric_constants=numeric_constants,
        entities=entities,
        states=states,
        events=events,
        topology_graph=topology_graph,
        relations=relations,
        constraints=constraints,
        goals=goals,
        concepts=concepts,
        target_hints=target_hints,
        answer_type_hint=answer_type_hint,
        warnings=warnings,
    )


def process_question_front(question: str) -> dict:
    """Run the deterministic semantic frontend defined by NSP-Core."""

    from .canonical import build_canonical_structures
    from ..engines.logic_engine import apply_logic_rules

    normalized = normalize_question(question)
    enriched = apply_logic_rules(normalized)
    payload = {
        "raw_question": normalized.raw_question,
        "canonical_question": normalized.canonical_question,
        "quantities": [quantity.to_dict() for quantity in normalized.quantities],
        "symbolic_quantities": [quantity.to_dict() for quantity in normalized.symbolic_quantities],
        "symbolic_relations": [relation.to_dict() for relation in normalized.symbolic_relations],
        "numeric_constants": [constant.to_dict() for constant in normalized.numeric_constants],
        "entities": [entity.to_dict() for entity in normalized.entities],
        "states": [state.to_dict() for state in normalized.states],
        "events": [event.to_dict() for event in normalized.events],
        "topology_graph": normalized.topology_graph.to_dict(),
        "relations": [relation.to_dict() for relation in normalized.relations],
        "constraints": [constraint.to_dict() for constraint in normalized.constraints],
        "goals": [goal.to_dict() for goal in normalized.goals],
        "concepts": list(normalized.concepts),
        "target_hints": list(normalized.target_hints),
        "answer_type_hint": normalized.answer_type_hint,
        "parse_confidence": normalized.parse_confidence,
        "warnings": list(normalized.warnings),
        "implicit_facts": [fact.to_dict() for fact in enriched.implicit_facts],
        "derived_facts": [fact.to_dict() for fact in enriched.derived_facts],
        "premises": list(enriched.premises),
        "trace": {
            "stages": ["semantic_parser", "logic_engine"],
            "semantic_parser": {
                "quantity_count": len(normalized.quantities),
                "symbolic_quantity_count": len(normalized.symbolic_quantities),
                "symbolic_relation_count": len(normalized.symbolic_relations),
                "numeric_constant_count": len(normalized.numeric_constants),
                "entity_count": len(normalized.entities),
                "state_count": len(normalized.states),
                "event_count": len(normalized.events),
                "topology_node_count": len(normalized.topology_graph.nodes),
                "topology_edge_count": len(normalized.topology_graph.edges),
                "topology_canonical_form": normalized.topology_graph.canonical_form,
                "relation_count": len(normalized.relations),
                "constraint_count": len(normalized.constraints),
                "goal_count": len(normalized.goals),
                "concept_count": len(normalized.concepts),
                "target_hint_count": len(normalized.target_hints),
                "answer_type_hint": normalized.answer_type_hint,
                "warnings": list(normalized.warnings),
                "llm_used": False,
            },
            "logic_engine": enriched.trace,
        },
    }
    canonical_structures = build_canonical_structures(payload)
    payload["canonical_structures"] = canonical_structures
    payload["trace"]["semantic_parser"]["canonical_structure_count"] = (
        len((canonical_structures.get("geometry") or {}).get("triangles") or [])
        + len((canonical_structures.get("geometry") or {}).get("squares") or [])
        + sum(len(items) for items in (canonical_structures.get("component_groups") or {}).values())
    )
    return payload


def _quantities_from_numeric_constants(canonical_question: str, constants: List[NumericConstant]) -> List[Quantity]:
    quantities: List[Quantity] = []
    for constant in constants:
        unit = _base_unit_for_hidden_dimension(constant.dimension, canonical_question)
        if unit is None:
            continue
        info = unit_info(unit)
        if info is None:
            continue
        quantities.append(
            Quantity(
                raw_text=constant.raw_text,
                value=constant.value,
                unit=unit,
                raw_unit="implicit_base_SI",
                symbol=constant.symbol,
                dimension=info.dimension,
                span=constant.span,
                context=constant.context,
                confidence=min(0.76, constant.confidence),
            )
        )
    return quantities


def _base_unit_for_hidden_dimension(dimension: Optional[str], text: str) -> Optional[str]:
    if not dimension or dimension == "constant":
        return None
    lowered = text.lower()
    if dimension == "resistance_or_distance":
        if any(cue in lowered for cue in ["resistance", "resistor", "ohm", "circuit"]):
            return "Ω"
        if any(cue in lowered for cue in ["distance", "separation", "radius", "length"]):
            return "m"
        return None
    if dimension == "electric_field_or_energy":
        if "electric field" in lowered or "field strength" in lowered:
            return "V/m"
        if "energy" in lowered or "work" in lowered:
            return "J"
        return None
    if dimension == "force_or_frequency":
        if "frequency" in lowered or "resonance" in lowered:
            return "Hz"
        if "force" in lowered:
            return "N"
        return None
    return {
        "angle": "rad",
        "area": "m^2",
        "capacitance": "F",
        "capacitive_reactance": "Ω",
        "charge": "C",
        "current": "A",
        "dimensionless": "-",
        "energy": "J",
        "force": "N",
        "frequency": "Hz",
        "impedance": "Ω",
        "inductance": "H",
        "inductive_reactance": "Ω",
        "length": "m",
        "magnetic_field": "T",
        "magnetic_flux": "Wb",
        "number_density": "m^-3",
        "phase_angle": "rad",
        "power": "W",
        "resistance": "Ω",
        "resistivity": "Ω*m",
        "time": "s",
        "turn_density": "turns/m",
        "angular_frequency": "rad/s",
        "voltage": "V",
    }.get(dimension)
