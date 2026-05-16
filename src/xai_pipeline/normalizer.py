import re
from typing import Optional, Tuple


NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def normalize_logic_answer(ans: str) -> str:
    s = str(ans).strip()
    u = s.upper()
    if u in {"A", "B", "C", "D"}:
        return u
    l = s.lower()
    if l in {"yes", "true"}:
        return "Yes"
    if l in {"no", "false"}:
        return "No"
    if l in {"unknown", "uncertain", "cannot determine", "not sure"}:
        return "Unknown"
    return s


def normalize_unit(unit: str) -> str:
    u = str(unit).strip().lower().replace("ω", "ohm")
    aliases = {
        "volt": "V",
        "volts": "V",
        "v": "V",
        "amp": "A",
        "amps": "A",
        "ampere": "A",
        "a": "A",
        "ohm": "ohm",
        "j": "J",
        "joule": "J",
        "w": "W",
        "watt": "W",
        "f": "F",
        "farad": "F",
        "c": "C",
        "coulomb": "C",
        "n": "N",
        "v/m": "V/m",
        "n/c": "V/m",
    }
    return aliases.get(u, unit.strip())


def extract_number(text: str) -> Optional[float]:
    m = NUM_RE.search(str(text))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def split_answer_number_unit(answer: str) -> Tuple[Optional[float], str]:
    s = str(answer).strip()
    m = NUM_RE.search(s)
    if not m:
        return None, ""
    try:
        n = float(m.group(0))
    except Exception:
        return None, ""
    unit = (s[: m.start()] + s[m.end() :]).strip()
    return n, normalize_unit(unit)
