"""Deterministic implicit knowledge base.

Rules here are code-owned allowlist entries. They may add constants or default
assumptions only when the question text contains trigger evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple

from .normalizer import canonicalize_question
from .schemas import ImplicitFact, ImplicitKBResult, NormalizedQuestion


@dataclass(frozen=True)
class ImplicitRule:
    rule_id: str
    adds: Dict[str, str]
    premise: str
    trigger_patterns: Tuple[Pattern[str], ...]
    confidence: float = 0.95


def _compile(*patterns: str) -> Tuple[Pattern[str], ...]:
    return tuple(re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns)


IMPLICIT_RULES: Dict[str, ImplicitRule] = {
    "school_coulomb_constant": ImplicitRule(
        rule_id="school_coulomb_constant",
        adds={"k": "9e9 N*m^2/C^2"},
        premise="Use the school Coulomb constant k = 9e9 N*m^2/C^2.",
        trigger_patterns=_compile(
            r"\b(point charges?|electric charges?|coulomb|electric force)\b",
            r"\b(in air|in vacuum)\b.*\b(charges?|electric field|electric force)\b",
            r"\b(charges?|electric field|electric force)\b.*\b(in air|in vacuum)\b",
        ),
        confidence=0.92,
    ),
    "vacuum_permittivity": ImplicitRule(
        rule_id="vacuum_permittivity",
        adds={"epsilon0": "8.8541878128e-12 F/m"},
        premise="Use vacuum permittivity epsilon0 = 8.8541878128e-12 F/m.",
        trigger_patterns=_compile(
            r"\b(vacuum permittivity|permittivity of free space|parallel-plate capacitor|parallel plate capacitor)\b"
        ),
        confidence=0.9,
    ),
    "magnetic_constant": ImplicitRule(
        rule_id="magnetic_constant",
        adds={"mu0": "4*pi*1e-7 N/A^2"},
        premise="Use magnetic constant mu0 = 4*pi*1e-7 N/A^2.",
        trigger_patterns=_compile(r"\b(solenoid|permeability of free space|mu0|μ0)\b"),
        confidence=0.9,
    ),
    "fully_charged_capacitor": ImplicitRule(
        rule_id="fully_charged_capacitor",
        adds={"capacitor_state": "fully_charged"},
        premise="The capacitor is treated as fully charged as stated in the question.",
        trigger_patterns=_compile(r"\bfully charged\b", r"\bcharged under\b"),
        confidence=0.96,
    ),
    "ideal_lc_no_loss": ImplicitRule(
        rule_id="ideal_lc_no_loss",
        adds={"energy_loss": "0"},
        premise="Treat the LC circuit as ideal/no-loss when the problem states an ideal LC circuit.",
        trigger_patterns=_compile(r"\bideal LC\b", r"\bLC circuit\b.*\b(no loss|lossless)\b"),
        confidence=0.9,
    ),
    "series_rlc_resonance": ImplicitRule(
        rule_id="series_rlc_resonance",
        adds={"resonance_condition": "XL = XC", "phase_difference": "0"},
        premise="At resonance in a series RLC circuit, XL = XC and the phase difference is 0.",
        trigger_patterns=_compile(r"\b(series RLC|RLC)\b.*\bresonan", r"\bresonan\w*\b.*\b(series RLC|RLC)\b"),
        confidence=0.95,
    ),
    "electron": ImplicitRule(
        rule_id="electron",
        adds={"electron.q": "-1.602176634e-19 C", "electron.m": "9.1093837015e-31 kg"},
        premise="Use the standard electron charge and rest-mass constants.",
        trigger_patterns=_compile(r"\belectron\b"),
        confidence=0.95,
    ),
    "proton": ImplicitRule(
        rule_id="proton",
        adds={"proton.q": "+1.602176634e-19 C", "proton.m": "1.67262192369e-27 kg"},
        premise="Use the standard proton charge and rest-mass constants.",
        trigger_patterns=_compile(r"\bproton\b"),
        confidence=0.95,
    ),
}


def allowed_implicit_rule_ids() -> List[str]:
    return sorted(IMPLICIT_RULES)


def _first_trigger(rule: ImplicitRule, text: str) -> Optional[Tuple[str, Tuple[int, int]]]:
    for pattern in rule.trigger_patterns:
        match = pattern.search(text)
        if match:
            return match.group(0), match.span()
    return None


def apply_implicit_kb(normalized: NormalizedQuestion | str) -> ImplicitKBResult:
    if isinstance(normalized, str):
        normalized_question = NormalizedQuestion(
            raw_question=normalized,
            canonical_question=canonicalize_question(normalized),
            quantities=[],
            parse_confidence=0.0,
            symbolic_quantities=[],
            symbolic_relations=[],
            numeric_constants=[],
            concepts=[],
            target_hints=[],
            answer_type_hint="unknown",
            warnings=["implicit_kb_received_raw_string"],
        )
    else:
        normalized_question = normalized

    text = normalized_question.canonical_question
    facts: List[ImplicitFact] = []
    for rule_id in allowed_implicit_rule_ids():
        rule = IMPLICIT_RULES[rule_id]
        trigger = _first_trigger(rule, text)
        if not trigger:
            continue
        trigger_text, span = trigger
        facts.append(
            ImplicitFact(
                rule_id=rule.rule_id,
                adds=rule.adds,
                premise=rule.premise,
                trigger_text=trigger_text,
                span=span,
                confidence=rule.confidence,
            )
        )

    trace = {
        "stage": "implicit_kb",
        "rules_checked": allowed_implicit_rule_ids(),
        "rules_applied": [fact.rule_id for fact in facts],
        "llm_used": False,
    }
    return ImplicitKBResult(
        normalized=normalized_question,
        implicit_facts=facts,
        premises=[fact.premise for fact in facts],
        trace=trace,
    )
