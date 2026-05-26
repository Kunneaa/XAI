"""Small deterministic language helpers shared across the physics pipeline.

These helpers intentionally extract only structural cues such as multiplicative
change factors. They do not encode dataset examples or final answers.
"""

from __future__ import annotations

import re


_CHANGE_WORD_FACTORS = (
    (re.compile(r"\b(?:double[ds]?|twice)\b", re.IGNORECASE), 2.0),
    (re.compile(r"\b(?:halve[ds]?|half|one\s+half)\b", re.IGNORECASE), 0.5),
    (re.compile(r"\btriple[ds]?\b", re.IGNORECASE), 3.0),
    (re.compile(r"\bquadruple[ds]?\b", re.IGNORECASE), 4.0),
)


def contains_any(text: str, cues: tuple[str, ...] | list[str] | set[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(cue).lower() in lowered for cue in cues)


def extract_change_factor(text: str) -> float | None:
    """Return the multiplicative factor implied by generic wording.

    Examples:
    - "frequency is doubled" -> 2
    - "separation is reduced by a factor of 4" -> 0.25
    - "distance changes to 3 times its initial value" -> 3
    """

    lowered = str(text or "").lower()
    for pattern, factor in _CHANGE_WORD_FACTORS:
        if pattern.search(lowered):
            return factor

    inverse_patterns = (
        r"\b(?:decrease[ds]?|reduce[ds]?)\s+by\s+(?:a\s+)?factor\s+of\s+(?P<factor>\d+(?:\.\d+)?)",
        r"\b(?:becomes?|is|are|was|were)\s+one\s+(?:over|/)\s+(?P<factor>\d+(?:\.\d+)?)\b",
    )
    for pattern in inverse_patterns:
        match = re.search(pattern, lowered)
        if match:
            factor = _positive_float(match.group("factor"))
            return 1.0 / factor if factor else None

    direct_patterns = (
        r"\b(?:increase[ds]?|raise[ds]?|multiply|multiplied|scale[ds]?|change[ds]?)\s+by\s+(?:a\s+)?factor\s+of\s+(?P<factor>\d+(?:\.\d+)?)",
        r"\b(?:increase[ds]?|raise[ds]?|multiply|multiplied|scale[ds]?|change[ds]?)\s+by\s+(?P<factor>\d+(?:\.\d+)?)\s+(?:times|x)\b",
        r"\b(?:increase[ds]?|raise[ds]?|multiply|multiplied|scale[ds]?|change[ds]?)\s+to\s+(?P<factor>\d+(?:\.\d+)?)\s+(?:times|x)\b",
        r"\b(?:becomes?|is|are|was|were)\s+(?P<factor>\d+(?:\.\d+)?)\s+(?:times|x)\b",
        r"\bby\s+(?:a\s+)?factor\s+of\s+(?P<factor>\d+(?:\.\d+)?)\b",
    )
    for pattern in direct_patterns:
        match = re.search(pattern, lowered)
        if match:
            return _positive_float(match.group("factor"))
    return None


def has_change_factor_cue(text: str) -> bool:
    lowered = str(text or "").lower()
    return extract_change_factor(lowered) is not None or contains_any(
        lowered,
        {
            "factor",
            "times",
            "proportional",
            "depends",
            "increase",
            "decrease",
            "reduced",
            "scaled",
            "changed",
        },
    )


def has_frequency_transform_cue(text: str) -> bool:
    lowered = str(text or "").lower()
    if not contains_any(lowered, {"frequency", "angular frequency", "omega", "ω"}):
        return False
    return has_change_factor_cue(lowered) or contains_any(
        lowered,
        {
            "adjusted",
            "multiple of",
            "to achieve resonance",
            "to obtain resonance",
            "to resonate",
            "resonant frequency",
        },
    )


def _positive_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
