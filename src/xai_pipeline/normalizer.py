"""Deterministic question normalization and raw quantity extraction."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional, Tuple

from .schemas import NormalizedQuestion, NumericConstant, Quantity, SymbolicQuantity, SymbolicRelation
from .units import normalize_unit, unit_info


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
    r"N\*m\^2/C\^2|N\.m\^2/C\^2|N\*m2/C2|N\.m2/C2|N×m²/C²|V/m|N/C|μC/m\^2|µC/m\^2|uC/m\^2|μC/m²|µC/m²|uC/m²|μC/m2|µC/m2|uC/m2|C/m\^2|C/m²|C/m2|C/m|km/s|m/s\^2|m/s²|m/s2|m/s|"
    r"kg/m\^3|kg/m³|kg/m3|g/cm\^3|g/cm³|g/cm3|m\^-3|m-3|cm\^-3|cm-3|J/\(kg°C\)|J/kg°C|"
    r"kΩ|kohm|Ω|ohms?|"
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
    r"turns/m|turns|turn|"
    r"ms|s|kg|g|Pa|kPa|T|Wb|atm|ml|mL|%|°C|°|deg|degree|degrees|rad"
)
UNIT_CAPTURE = rf"\(?(?:{UNIT_PATTERN})\)?"

SYMBOL_CHARS = r"A-Za-zωΩφΦμμεεθΘ"
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
    "qualitative_change": re.compile(r"\b(increases?|decreases?|brighter|halved|doubled|tripled|compare)\b", re.IGNORECASE),
    "parallel_circuit": re.compile(r"\bparallel\b", re.IGNORECASE),
    "brightness": re.compile(r"\b(bright|brighter|bulbs?|lamp)\b", re.IGNORECASE),
    "uniform_electric_field": re.compile(r"\buniform electric field\b", re.IGNORECASE),
    "magnetic_flux": re.compile(r"\bmagnetic flux\b|\bflux\b", re.IGNORECASE),
    "reactance": re.compile(r"\breactance\b|\bX_L\b|\bX_C\b|\bXL\b|\bXC\b", re.IGNORECASE),
    "impedance": re.compile(r"\bimpedance\b|\bZ\b", re.IGNORECASE),
}

TARGET_PATTERNS = (
    re.compile(r"\b(?:calculate|find|determine|compute)\s+(?P<target>[^.?]+)", re.IGNORECASE),
    re.compile(r"\bwhat is\s+(?P<target>[^.?]+)", re.IGNORECASE),
    re.compile(r"\bwhere is\s+(?P<target>[^.?]+)", re.IGNORECASE),
    re.compile(r"\bhow (?:does|will|is)\s+(?P<target>[^.?]+)", re.IGNORECASE),
)

ANSWER_TYPE_IDS = {"numeric", "symbolic", "conceptual", "yes_no", "multi_output", "unknown"}


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
        if _is_plain_numeric_rhs(rhs):
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
        if not symbol or value is None or not _is_symbol_candidate(symbol):
            continue
        if span in seen:
            continue
        seen.add(span)
        constants.append(
            NumericConstant(
                raw_text=match.group(0),
                symbol=symbol,
                value=value,
                dimension=_infer_symbol_dimension(symbol, canonical_question),
                span=span,
                context=_context(canonical_question, span),
            )
        )
    return constants


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


def infer_answer_type_hint(
    canonical_question: str,
    quantities: List[Quantity],
    symbolic_quantities: List[SymbolicQuantity],
    symbolic_relations: List[SymbolicRelation],
    concepts: List[str],
) -> str:
    q = canonical_question.lower()
    target_text = " ".join(extract_target_hints(canonical_question)).lower()
    multi_output_markers = [
        " and the ",
        " and its ",
        " respectively",
    ]
    if target_text and any(marker in target_text for marker in multi_output_markers):
        target_terms = [
            "energy",
            "charge",
            "current",
            "voltage",
            "resistance",
            "capacitance",
            "reactance",
            "power factor",
            "impedance",
            "field",
            "force",
            "flux",
        ]
        if sum(1 for term in target_terms if term in target_text) >= 2:
            return "multi_output"
        if "each" in target_text and "total" in target_text:
            return "multi_output"
    if target_text and "charge and energy" in target_text:
        return "multi_output"
    yes_no_question = canonical_question.strip().endswith("?") and (
        re.match(r"\s*(?:does|do|is|are|can|will|would|should)\b", q)
        or re.search(r"\bdoes\s+[^?]*(?:occur|resonate|hold|remain)\b", q)
    )
    if re.search(r"\b(yes or no|true or false)\b", q) or yes_no_question:
        return "yes_no"
    if re.search(r"\b(relationship|expression|formula|in terms of|derive)\b", q):
        return "symbolic"
    if (symbolic_relations or symbolic_quantities) and re.search(r"\b(determine|calculate|find|what is)\b", q):
        return "symbolic"
    if re.search(r"\b(unit of|si unit|where|what happens|which|how does|how will|shape of the graph)\b", q):
        return "conceptual"
    if quantities:
        return "numeric"
    if symbolic_relations or symbolic_quantities:
        return "symbolic"
    if concepts:
        return "conceptual"
    return "unknown"


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


def _trim_relation_rhs(rhs: str) -> str:
    rhs = re.split(r"\s+and\s+(?=[A-Za-z][A-Za-z0-9_]*\s*=)", rhs, maxsplit=1, flags=re.IGNORECASE)[0]
    rhs = re.split(r"\s+\((?:with|where)\b", rhs, maxsplit=1, flags=re.IGNORECASE)[0]
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
    if re.fullmatch(r"[qQ][A-Za-z]?", symbol):
        return True
    return False


def _infer_symbol_dimension(symbol: str, text: str) -> Optional[str]:
    s = symbol.lower()
    if s.startswith("q"):
        return "charge"
    if s == "k":
        return "constant"
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
    if s in {"b"}:
        return "magnetic_field"
    if s in {"n"} and "solenoid" in text.lower():
        return "turn_density"
    if s in {"n"} and any(cue in text.lower() for cue in ["carrier density", "number density", "drift"]):
        return "number_density"
    if s in {"a", "area"} or "area" in text.lower():
        return "area"
    if s in {"ε", "epsilon"}:
        return "permittivity"
    if s in {"μ", "mu"}:
        return "permeability_or_prefix"
    if s in {"u", "v"} or "voltage" in text.lower():
        return "voltage"
    if s in {"i"} or "current" in text.lower():
        return "current"
    if s in {"r"} or "resistance" in text.lower():
        return "resistance_or_distance"
    if s in {"c"} or "capacitance" in text.lower():
        return "capacitance"
    if s in {"l"} or "inductance" in text.lower():
        return "inductance"
    if s in {"a", "h", "d"} or any(word in text.lower() for word in ["distance", "side length", "radius"]):
        return "length"
    if s.startswith("e"):
        return "electric_field_or_energy"
    if s.startswith("f"):
        return "force_or_frequency"
    if s.startswith("w"):
        return "energy"
    return None


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
    symbolic_relations = extract_symbolic_relations(canonical, quantities)
    numeric_constants = extract_numeric_constants(canonical, quantities)
    symbolic_quantities = extract_symbolic_quantities(canonical, quantities, symbolic_relations)
    concepts = extract_concepts(canonical)
    target_hints = extract_target_hints(canonical)
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
        concepts=concepts,
        target_hints=target_hints,
        answer_type_hint=answer_type_hint,
        warnings=warnings,
    )
