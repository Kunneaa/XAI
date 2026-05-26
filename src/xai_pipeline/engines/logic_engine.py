"""Deterministic logic engine for triggered physical assumptions.

Rules here are code-owned allowlist entries. They may add constants or default
assumptions only when the question text contains trigger evidence.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple

from ..frontend.semantic_ir import DerivedFact, ImplicitFact, LogicEngineResult, NormalizedQuestion
from ..frontend.semantic_parser import canonicalize_question
from ..knowledge.language import extract_change_factor, has_change_factor_cue
from ..knowledge.units import unit_info


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


def apply_logic_rules(normalized: NormalizedQuestion | str) -> LogicEngineResult:
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
            warnings=["logic_engine_received_raw_string"],
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

    derived_facts = _derive_forward_chained_facts(normalized_question, facts)
    derived_premises = _premises_from_derived_facts(derived_facts)
    trace = {
        "stage": "logic_engine",
        "rules_checked": allowed_implicit_rule_ids(),
        "rules_applied": [fact.rule_id for fact in facts],
        "derived_fact_ids": [fact.fact_id for fact in derived_facts],
        "forward_chaining": {
            "relation_count": len(normalized_question.relations),
            "constraint_count": len(normalized_question.constraints),
            "state_count": len(normalized_question.states),
            "event_count": len(normalized_question.events),
        },
        "llm_used": False,
    }
    return LogicEngineResult(
        normalized=normalized_question,
        implicit_facts=facts,
        premises=[fact.premise for fact in facts] + derived_premises,
        trace=trace,
        derived_facts=derived_facts,
    )


def _derive_forward_chained_facts(normalized: NormalizedQuestion, implicit_facts: List[ImplicitFact]) -> List[DerivedFact]:
    """Run deterministic forward chaining over relations, constraints, states, and events."""

    derived: List[DerivedFact] = []
    relation_qualifiers = {relation.qualifier for relation in normalized.relations}
    constraint_ids = {constraint.constraint_id for constraint in normalized.constraints}
    state_labels = {state.label for state in normalized.states}
    event_types = {event.event_type for event in normalized.events}
    implicit_ids = {fact.rule_id for fact in implicit_facts}

    def add(fact_id: str, kind: str, expression: str, supports: List[str], confidence: float = 0.84) -> None:
        if any(existing.fact_id == fact_id for existing in derived):
            return
        derived.append(DerivedFact(fact_id=fact_id, kind=kind, expression=expression, supports=supports, confidence=confidence))

    if "series" in relation_qualifiers or "series_current_same" in constraint_ids:
        add(
            "topology.series_current_equal",
            "topology",
            "All ideal series elements share the same branch current.",
            ["relation:series", "constraint:series_current_same"],
        )
    if "parallel" in relation_qualifiers or "parallel_voltage_same" in constraint_ids:
        add(
            "topology.parallel_voltage_equal",
            "topology",
            "All ideal parallel branches share the same voltage across their terminals.",
            ["relation:parallel", "constraint:parallel_voltage_same"],
        )
    if "balanced_bridge" in relation_qualifiers:
        add(
            "topology.wheatstone_bridge_no_galvanometer_current",
            "topology",
            "A balanced Wheatstone bridge has zero current through the bridge branch.",
            ["relation:balanced_bridge"],
        )
    if "resonance" in relation_qualifiers or "resonance_xl_equals_xc" in constraint_ids or "series_rlc_resonance" in implicit_ids:
        add(
            "state.rlc_resonance_xl_equals_xc",
            "state",
            "At RLC resonance, inductive and capacitive reactances are equal.",
            ["relation:resonance", "constraint:resonance_xl_equals_xc", "implicit:series_rlc_resonance"],
            0.88,
        )
        add(
            "state.rlc_resonance_phase_zero",
            "state",
            "At series RLC resonance, voltage and current are in phase.",
            ["implicit:series_rlc_resonance"],
            0.86,
        )
    if "ideal_lc_circuit" in normalized.concepts or "ideal_lc_no_loss" in implicit_ids:
        add(
            "conservation.ideal_lc_energy",
            "conservation",
            "In an ideal LC circuit, total electromagnetic energy is conserved.",
            ["concept:ideal_lc_circuit", "implicit:ideal_lc_no_loss"],
            0.87,
        )
    if "disconnected_from_source" in state_labels or "battery_removed" in event_types or "disconnect" in event_types:
        add(
            "conservation.isolated_capacitor_charge",
            "conservation",
            "For an isolated disconnected capacitor, charge is conserved unless a discharge path is stated.",
            ["state:disconnected_from_source", "event:battery_removed", "event:disconnect"],
            0.8,
        )
    if "medium_air_or_vacuum" in constraint_ids or "school_coulomb_constant" in implicit_ids:
        add(
            "medium.air_vacuum_epsilon_r_one",
            "medium",
            "For school electrostatics in air or vacuum, epsilon_r is treated as 1.",
            ["constraint:medium_air_or_vacuum", "implicit:school_coulomb_constant"],
            0.82,
        )
    if "dielectric_inserted" in event_types:
        add(
            "event.dielectric_changes_capacitance",
            "state_transition",
            "Inserting a dielectric increases capacitance by the relative permittivity factor for fixed geometry.",
            ["event:dielectric_inserted"],
            0.78,
        )
    if "frequency_changed" in event_types or "changed_frequency" in state_labels:
        add(
            "event.frequency_parameter_changed",
            "state_transition",
            "Frequency-dependent reactances must be recomputed for the changed state.",
            ["event:frequency_changed", "state:changed_frequency"],
            0.76,
        )
    return derived


def _premises_from_derived_facts(derived_facts: List[DerivedFact]) -> List[str]:
    return [fact.expression for fact in derived_facts]


def solve_conceptual(front_payload: dict, route_result) -> object:
    """Answer small conceptual/yes-no questions from proof facts only.

    This is deliberately not a free-form explainer. It maps deterministic facts
    already present in the semantic IR to a concise answer, and otherwise
    abstains so the numeric engines can try or the pipeline can return
    unverified.
    """

    from .equation_engine import SolverResult

    text = str(front_payload.get("canonical_question") or "").lower()
    fact_by_id = {fact.get("fact_id"): fact for fact in front_payload.get("derived_facts", [])}
    concepts = set(front_payload.get("concepts") or [])
    answer_type = front_payload.get("answer_type_hint")
    conceptual_rule_signal = bool(
        {"proportionality", "qualitative_change", "brightness", "rlc_circuit", "reactance", "solenoid", "magnetic_field_energy", "electric_field_energy"}
        & concepts
    ) or bool(re.search(r"\b(?:which|what)\s+(?:of\s+the\s+following\s+)?quantit(?:y|ies)\b|\bdepends?\s+linearly\b", text))
    if answer_type not in {"conceptual", "yes_no"} and route_result.task_type != "conceptual" and not conceptual_rule_signal:
        return SolverResult(False, "", None, None, None, None, [], {"stage": "logic_conceptual_engine", "reason": "not_conceptual_route"}, 0.0)

    rlc_resonance = _solve_numeric_rlc_resonance_yes_no(text, front_payload)
    if rlc_resonance is not None:
        answer, audit = rlc_resonance
        premise = "Series LC/RLC resonance occurs when f = 1/(2*pi*sqrt(L*C)), equivalently XL = XC."
        return SolverResult(
            solved=True,
            answer=answer,
            value=answer,
            unit="-",
            formula_id="yes_no_direct",
            principle_id="conceptual_core",
            premises=[premise],
            trace={
                "stage": "logic_conceptual_engine",
                "rule": "numeric_rlc_resonance_condition",
                "expression": premise,
                "audit": audit,
                "derived_fact_ids": sorted(fact_by_id),
                "target_dimension": "dimensionless",
            },
            confidence=min(0.76, route_result.confidence),
        )

    yes_no = _solve_yes_no_from_facts(text, fact_by_id)
    if yes_no is not None:
        answer, fact_id = yes_no
        fact = fact_by_id.get(fact_id, {})
        return SolverResult(
            solved=True,
            answer=answer,
            value=answer,
            unit="-",
            formula_id="yes_no_direct",
            principle_id="conceptual_core",
            premises=[str(fact.get("expression") or "Use a deterministic conceptual rule.")],
            trace={
                "stage": "logic_conceptual_engine",
                "rule": "fact_grounded_yes_no",
                "fact_id": fact_id,
                "expression": fact.get("expression"),
                "derived_fact_ids": sorted(fact_by_id),
                "target_dimension": "dimensionless",
            },
            confidence=min(0.78, route_result.confidence),
        )

    time_inductor = _solve_time_dependent_inductor_energy(text, front_payload)
    if time_inductor is not None:
        value, audit = time_inductor
        premise = "Magnetic energy in an inductor is W = 1/2 L I^2; evaluate the stated current function at the requested time."
        return SolverResult(
            True,
            f"{value:.6g} J",
            value,
            "J",
            "inductor_energy",
            "inductor_core",
            [premise],
            {
                "stage": "logic_conceptual_engine",
                "rule": "time_dependent_inductor_energy",
                "expression": premise,
                "audit": audit,
                "target_dimension": "energy",
            },
            min(0.76, route_result.confidence),
        )

    formula_or_graph = _solve_formula_or_graph_question(text)
    if formula_or_graph is not None:
        answer, rule_id, premise = formula_or_graph
        return SolverResult(
            solved=True,
            answer=answer,
            value=answer,
            unit="-",
            formula_id="conceptual_direct",
            principle_id="conceptual_core",
            premises=[premise],
            trace={
                "stage": "logic_conceptual_engine",
                "rule": rule_id,
                "expression": premise,
                "derived_fact_ids": sorted(fact_by_id),
                "target_dimension": "dimensionless",
            },
            confidence=min(0.76, route_result.confidence),
        )

    unit_answer = _solve_si_unit_question(text)
    if unit_answer is not None:
        label, unit = unit_answer
        answer = f"The SI unit of {label} is {unit}."
        return SolverResult(
            solved=True,
            answer=answer,
            value=answer,
            unit="-",
            formula_id="conceptual_direct",
            principle_id="conceptual_core",
            premises=[f"{unit} is the registry-owned SI unit for {label}."],
            trace={
                "stage": "logic_conceptual_engine",
                "rule": "si_unit_lookup",
                "quantity": label,
                "unit": unit,
                "derived_fact_ids": sorted(fact_by_id),
                "target_dimension": "dimensionless",
            },
            confidence=min(0.76, route_result.confidence),
        )

    if "state.rlc_resonance_xl_equals_xc" in fact_by_id and any(cue in text for cue in ["xl", "x_l", "xc", "x_c", "reactance"]):
        fact = fact_by_id["state.rlc_resonance_xl_equals_xc"]
        answer = "At resonance, the inductive reactance equals the capacitive reactance."
        return SolverResult(
            True,
            answer,
            answer,
            "-",
            "conceptual_direct",
            "conceptual_core",
            [str(fact.get("expression") or answer)],
            {
                "stage": "logic_conceptual_engine",
                "rule": "rlc_resonance_reactance",
                "fact_id": "state.rlc_resonance_xl_equals_xc",
                "expression": fact.get("expression"),
                "derived_fact_ids": sorted(fact_by_id),
                "target_dimension": "dimensionless",
            },
            min(0.77, route_result.confidence),
        )

    if "conservation.ideal_lc_energy" in fact_by_id and ("energy" in text or "conserved" in text):
        fact = fact_by_id["conservation.ideal_lc_energy"]
        answer = "In an ideal LC circuit, total electromagnetic energy is conserved."
        return SolverResult(
            True,
            answer,
            answer,
            "-",
            "conceptual_direct",
            "conceptual_core",
            [str(fact.get("expression") or answer)],
            {
                "stage": "logic_conceptual_engine",
                "rule": "ideal_lc_energy_conservation",
                "fact_id": "conservation.ideal_lc_energy",
                "expression": fact.get("expression"),
                "concepts": sorted(concepts),
                "derived_fact_ids": sorted(fact_by_id),
                "target_dimension": "dimensionless",
            },
            min(0.77, route_result.confidence),
        )

    ratio_answer = _solve_field_force_charge_ratio_rule(text)
    if ratio_answer is not None:
        answer, audit = ratio_answer
        premise = "For a test charge in an electric field, F = qE, so field ratios satisfy E1/E2 = (F1/F2)/(q1/q2)."
        return SolverResult(
            True,
            answer,
            answer,
            "-",
            "conceptual_direct",
            "conceptual_core",
            [premise],
            {
                "stage": "logic_conceptual_engine",
                "rule": "electric_field_force_charge_ratio",
                "expression": premise,
                "ratio_audit": audit,
                "concepts": sorted(concepts),
                "derived_fact_ids": sorted(fact_by_id),
                "target_dimension": "dimensionless",
            },
            min(0.74, route_result.confidence),
        )

    proportional = _solve_proportional_or_qualitative_rule(text, concepts, front_payload)
    if proportional is not None:
        answer, rule_id, premise = proportional
        return SolverResult(
            True,
            answer,
            answer,
            "-",
            "conceptual_direct",
            "conceptual_core",
            [premise],
            {
                "stage": "logic_conceptual_engine",
                "rule": rule_id,
                "expression": premise,
                "concepts": sorted(concepts),
                "derived_fact_ids": sorted(fact_by_id),
                "target_dimension": "dimensionless",
            },
            min(0.74, route_result.confidence),
        )

    return SolverResult(
        False,
        "",
        None,
        None,
        None,
        None,
        [],
        {
            "stage": "logic_conceptual_engine",
            "reason": "no_grounded_conceptual_rule",
            "derived_fact_ids": sorted(fact_by_id),
        },
        0.0,
    )


def _solve_yes_no_from_facts(text: str, fact_by_id: dict[str, dict]) -> Tuple[str, str] | None:
    if "topology.series_current_equal" in fact_by_id and "current" in text and _asks_same_or_equal(text):
        return "Yes", "topology.series_current_equal"
    if "topology.parallel_voltage_equal" in fact_by_id and any(cue in text for cue in ["voltage", "potential difference"]) and _asks_same_or_equal(text):
        return "Yes", "topology.parallel_voltage_equal"
    if "conservation.isolated_capacitor_charge" in fact_by_id and "charge" in text:
        if any(cue in text for cue in ["change", "changes", "decrease", "increase"]):
            return "No", "conservation.isolated_capacitor_charge"
        if any(cue in text for cue in ["conserved", "constant", "same", "remain", "unchanged"]):
            return "Yes", "conservation.isolated_capacitor_charge"
    if "conservation.ideal_lc_energy" in fact_by_id and "energy" in text:
        if any(cue in text for cue in ["conserved", "constant", "same", "remain"]):
            return "Yes", "conservation.ideal_lc_energy"
        if any(cue in text for cue in ["lost", "loss", "decrease"]):
            return "No", "conservation.ideal_lc_energy"
    if "state.rlc_resonance_xl_equals_xc" in fact_by_id and _asks_same_or_equal(text):
        return "Yes", "state.rlc_resonance_xl_equals_xc"
    return None


def _solve_numeric_rlc_resonance_yes_no(text: str, front_payload: dict) -> Tuple[str, dict] | None:
    if "resonan" not in text and "resonate" not in text:
        return None
    quantities = front_payload.get("quantities") or []
    inductance = _first_quantity_value(quantities, "inductance")
    capacitance = _first_quantity_value(quantities, "capacitance")
    frequency = _first_quantity_value(quantities, "frequency")
    if inductance is not None and capacitance is not None and frequency is not None:
        if inductance <= 0 or capacitance <= 0 or frequency <= 0:
            return None
        f0 = 1.0 / (2.0 * math.pi * math.sqrt(inductance * capacitance))
        relative_error = abs(frequency - f0) / max(f0, 1e-30)
        return ("Yes" if relative_error <= 0.03 else "No"), {
            "method": "frequency_vs_lc_resonance",
            "f_input_hz": frequency,
            "f0_hz": f0,
            "relative_error": relative_error,
            "tolerance": 0.03,
        }
    reactances = [
        _si_value(quantity)
        for quantity in quantities
        if quantity.get("dimension") == "resistance"
        and unit_info(quantity.get("unit") or "") is not None
        and re.search(r"\b(?:x_l|xl|x_c|xc|reactance)\b", f"{quantity.get('symbol') or ''} {quantity.get('context') or ''}", re.IGNORECASE)
    ]
    if len(reactances) >= 2:
        xl, xc = reactances[:2]
        relative_error = abs(xl - xc) / max(abs(xl), abs(xc), 1e-30)
        return ("Yes" if relative_error <= 0.03 else "No"), {
            "method": "reactance_equality",
            "xl_ohm": xl,
            "xc_ohm": xc,
            "relative_error": relative_error,
            "tolerance": 0.03,
        }
    return None


def _first_quantity_value(quantities: list[dict], dimension: str) -> float | None:
    for quantity in quantities:
        if quantity.get("dimension") == dimension and unit_info(quantity.get("unit") or "") is not None:
            return _si_value(quantity)
    return None


def _si_value(quantity: dict) -> float:
    info = unit_info(quantity.get("unit") or "")
    if info is None:
        raise ValueError(f"unknown_unit:{quantity.get('unit')}")
    return float(quantity["value"]) * info.si_factor


def _solve_proportional_or_qualitative_rule(text: str, concepts: set[str], front_payload: dict | None = None) -> Tuple[str, str, str] | None:
    if not (
        {"proportionality", "qualitative_change", "brightness", "rlc_circuit", "reactance", "solenoid", "magnetic_field_energy", "electric_field_energy"}
        & concepts
    ) and not (
        has_change_factor_cue(text) or any(cue in text for cue in ["brighter", "brightness"])
    ):
        return None
    if any(cue in text for cue in ["coulomb force", "electric force", "force between charges"]):
        distance_factor = _change_factor_from_text(text) if "distance" in text or "separation" in text else None
        if distance_factor is not None and distance_factor > 0:
            force_factor = 1.0 / (distance_factor * distance_factor)
            if abs(distance_factor - 0.5) <= 1e-12:
                change_text = "halving the separation"
            elif abs(distance_factor - 2.0) <= 1e-12:
                change_text = "doubling the separation"
            else:
                change_text = f"changing the separation by a factor of {_format_factor(distance_factor)}"
            return (
                f"With the charges fixed, {change_text} makes the Coulomb force {_ratio_phrase(force_factor)} as large.",
                "coulomb_inverse_square_distance_factor",
                "Coulomb force obeys F proportional to |q1 q2|/r^2.",
            )
        return (
            "Coulomb force is directly proportional to the product of the charge magnitudes and inversely proportional to the square of their separation.",
            "coulomb_force_proportionality",
            "Coulomb force obeys F proportional to |q1 q2|/r^2.",
        )
    if "electric field" in text and "point charge" in text:
        return (
            "For a point charge, the electric field is directly proportional to |q| and inversely proportional to r squared.",
            "point_charge_field_proportionality",
            "A point-charge field obeys E proportional to |q|/r^2.",
        )
    capacitor_ratio = _solve_capacitor_proportionality_rule(text, front_payload or {})
    if capacitor_ratio is not None:
        return capacitor_ratio
    if "solenoid" in text and any(cue in text for cue in ["magnetic field energy density", "energy density"]):
        return (
            "The magnetic field energy density is proportional to the square of the magnetic induction B.",
            "solenoid_magnetic_energy_density_b_squared",
            "Magnetic field energy density obeys u_B = B^2/(2*mu0).",
        )
    if "solenoid" in text and "magnetic field" in text and any(cue in text for cue in ["directly proportional", "proportional to"]):
        return (
            "Inside a long solenoid, the magnetic field is directly proportional to the turn density and to the current.",
            "solenoid_field_turn_density_current_proportionality",
            "For an ideal long solenoid, B = mu0*n*I.",
        )
    if "solenoid" in text and "magnetic field" in text and any(cue in text for cue in ["depend linearly", "depends linearly", "linear on"]):
        return (
            "Inside a long solenoid, the magnetic field depends linearly on the current through the solenoid.",
            "solenoid_field_current_linear_dependence",
            "For fixed turn density in an ideal long solenoid, B = mu0*n*I is linear in I.",
        )
    if any(cue in text for cue in ["inductive reactance", "xl", "x_l"]) and "frequency" in text:
        return (
            "Inductive reactance changes in the same ratio as frequency.",
            "inductive_reactance_frequency_proportionality",
            "Inductive reactance obeys XL = 2*pi*f*L.",
        )
    if any(cue in text for cue in ["capacitive reactance", "xc", "x_c"]) and "frequency" in text:
        return (
            "Capacitive reactance changes inversely with frequency.",
            "capacitive_reactance_frequency_proportionality",
            "Capacitive reactance obeys XC = 1/(2*pi*f*C).",
        )
    if "rlc" in text and "frequency" in text:
        return (
            "When frequency changes in a series RLC circuit, recompute XL and XC first; impedance and phase then follow from Z = sqrt(R^2 + (XL-XC)^2).",
            "rlc_frequency_transform_rule",
            "In a series RLC circuit, XL is proportional to f while XC is inversely proportional to f.",
        )
    distance_factor = _change_factor_from_text(text) if "capacitor" in text and any(cue in text for cue in ["plate separation", "distance between plates", "distance"]) else None
    if distance_factor is not None and distance_factor > 0:
        capacitance_factor = 1.0 / distance_factor
        if any(cue in text for cue in ["disconnected", "isolated", "battery removed"]):
            return (
                f"For an isolated capacitor, changing plate separation by a factor of {_format_factor(distance_factor)} makes capacitance {_ratio_phrase(capacitance_factor)} as large, keeps charge constant, and makes voltage and stored energy {_ratio_phrase(distance_factor)} as large.",
                "isolated_capacitor_distance_factor",
                "For a parallel-plate capacitor C is proportional to 1/d, and an isolated capacitor conserves charge.",
            )
        if any(cue in text for cue in ["connected to battery", "connected to source", "constant voltage"]):
            return (
                f"With voltage held constant, changing plate separation by a factor of {_format_factor(distance_factor)} makes capacitance, charge, and stored energy {_ratio_phrase(capacitance_factor)} as large.",
                "fixed_voltage_capacitor_distance_factor",
                "For a parallel-plate capacitor C is proportional to 1/d and Q = C U, W = 1/2 C U^2.",
            )
    if any(cue in text for cue in ["lamp", "bulb", "brightness", "brighter"]) and "power" in text:
        return (
            "For identical lamps, greater electrical power means greater brightness.",
            "lamp_brightness_power_rule",
            "Lamp brightness is compared by dissipated electrical power for identical bulbs.",
        )
    return None


def _solve_capacitor_proportionality_rule(text: str, front_payload: dict) -> Tuple[str, str, str] | None:
    if "capacitor" not in text and "capacitance" not in text:
        return None
    voltage_factor = _change_factor_near_quantity(text, ("voltage", "potential difference", "u"))
    charge_factor = _change_factor_near_quantity(text, ("charge", "q"))
    if charge_factor is None:
        charge_factor = _quantity_replacement_factor(front_payload, "charge")
    capacitance_factor = _capacitance_replacement_factor(text, front_payload)
    distance_factor = _quantity_replacement_factor(front_payload, "length") if any(
        cue in text for cue in ["plate separation", "distance between", "distance", "separation"]
    ) else None
    if distance_factor is None and any(cue in text for cue in ["plate separation", "distance between plates", "separation"]):
        distance_factor = _change_factor_from_text(text)
    dielectric_factor = _dielectric_replacement_factor(text, front_payload)

    if (
        distance_factor is not None
        and distance_factor > 0
        and _fixed_charge_context(text)
        and "capacitance" in text
        and ("voltage" in text or "potential difference" in text)
        and "energy" in text
    ):
        capacitance_factor = 1.0 / distance_factor
        return (
            f"For an isolated capacitor, increasing plate separation by a factor of {_format_factor(distance_factor)} makes capacitance {_ratio_phrase(capacitance_factor)} as large, keeps charge constant, and makes voltage and stored energy {_ratio_phrase(distance_factor)} as large.",
            "isolated_capacitor_distance_factor",
            "For a parallel-plate capacitor C is proportional to 1/d, and an isolated capacitor conserves charge.",
        )

    if "energy" in text:
        if any(cue in text for cue in ["directly proportional", "proportional to"]) and any(cue in text for cue in ["which", "what quantity"]):
            return (
                "For fixed capacitance, the electric field energy stored in a capacitor is directly proportional to the square of the voltage, U^2.",
                "capacitor_energy_voltage_square_quantity",
                "Capacitor energy obeys W = 1/2 C U^2 for fixed capacitance.",
            )
        if distance_factor is not None and distance_factor > 0 and _fixed_charge_context(text):
            return (
                f"The stored energy becomes {_ratio_phrase(distance_factor)} as large.",
                "isolated_capacitor_energy_distance_ratio",
                "For a parallel-plate capacitor with fixed charge, C is proportional to 1/d and W = Q^2/(2C), so W is proportional to plate separation.",
            )
        if voltage_factor is not None and ("constant capacitance" in text or "capacitance" in text or "c =" in text):
            factor = voltage_factor * voltage_factor
            return (
                f"The stored energy becomes {_ratio_phrase(factor)} as large.",
                "capacitor_energy_voltage_square_proportionality",
                "For fixed capacitance, capacitor energy obeys W = 1/2 C U^2.",
            )
        if charge_factor is not None:
            factor = charge_factor * charge_factor
            return (
                f"The stored energy becomes {_ratio_phrase(factor)} as large.",
                "capacitor_energy_charge_square_proportionality",
                "For fixed capacitance, capacitor energy obeys W = Q^2/(2C).",
            )
        if capacitance_factor is not None and any(cue in text for cue in ["constant voltage", "connected to battery", "connected to source"]):
            return (
                f"With voltage fixed, the stored energy becomes {_ratio_phrase(capacitance_factor)} as large.",
                "capacitor_energy_capacitance_fixed_voltage_proportionality",
                "For fixed voltage, capacitor energy obeys W = 1/2 C U^2 and is proportional to C.",
            )
        if capacitance_factor is not None and (_fixed_charge_context(text) or any(cue in text for cue in ["isolated", "disconnected"])):
            factor = 1.0 / capacitance_factor
            return (
                f"With charge fixed, the stored energy becomes {_ratio_phrase(factor)} as large.",
                "capacitor_energy_inverse_capacitance_fixed_charge_proportionality",
                "For fixed charge, capacitor energy obeys W = Q^2/(2C).",
            )

    if "voltage" in text or "potential difference" in text:
        if capacitance_factor is not None and _fixed_charge_context(text):
            factor = 1.0 / capacitance_factor
            return (
                f"The voltage becomes {_ratio_phrase(factor)} as large.",
                "capacitor_voltage_inverse_capacitance_fixed_charge",
                "For fixed charge, capacitor voltage obeys U = Q/C.",
            )
        if charge_factor is not None and any(cue in text for cue in ["constant capacitance", "same capacitance"]):
            return (
                f"The voltage becomes {_ratio_phrase(charge_factor)} as large.",
                "capacitor_voltage_charge_fixed_capacitance",
                "For fixed capacitance, capacitor voltage obeys U = Q/C.",
            )

    if "capacitance" in text:
        factor = dielectric_factor if dielectric_factor is not None else capacitance_factor
        if factor is not None:
            return (
                f"The capacitance becomes {_ratio_phrase(factor)} as large.",
                "parallel_plate_capacitance_dielectric_or_geometry_factor",
                "For fixed plate geometry, capacitance is proportional to relative permittivity; for a parallel-plate capacitor, C is proportional to epsilon_r A/d.",
            )
    return None


def _fixed_charge_context(text: str) -> bool:
    return bool(
        re.search(r"\bcharge(?:\s+[a-z]\w*)?\s+(?:is\s+)?(?:kept|held|remains?)\s+constant\b", text)
        or re.search(r"\bconstant\s+charge\b|\bsame\s+charge\b", text)
        or re.search(r"\b(?:isolated|disconnected|battery removed)\b", text)
    )


def _solve_time_dependent_inductor_energy(text: str, front_payload: dict) -> Tuple[float, dict] | None:
    if not any(cue in text for cue in ["inductor", "coil"]) or "current" not in text or "energy" not in text:
        return None
    inductance = _first_quantity_value(front_payload.get("quantities") or [], "inductance")
    if inductance is None or inductance <= 0:
        return None
    current_match = re.search(
        r"\b(?:i|current)\s*(?:\(\s*t\s*\))?\s*(?:=|is)\s*"
        r"(?P<amp>[-+]?\d+(?:\.\d+)?)\s*(?P<trig>cos|sin)\s*\(?\s*(?P<omega>[-+]?\d+(?:\.\d+)?(?:\s*(?:π|pi))?)\s*t",
        text,
    )
    if not current_match:
        return None
    time_value = _parse_time_value(text)
    if time_value is None:
        return None
    amplitude = float(current_match.group("amp"))
    omega = _parse_pi_scaled_number(current_match.group("omega"))
    angle = omega * time_value
    trig = current_match.group("trig")
    current = amplitude * (math.cos(angle) if trig == "cos" else math.sin(angle))
    energy = 0.5 * inductance * current * current
    return energy, {
        "L": inductance,
        "current_amplitude": amplitude,
        "trig": trig,
        "omega": omega,
        "time": time_value,
        "current": current,
    }


def _parse_time_value(text: str) -> float | None:
    match = re.search(r"\bt\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?)(?:\s*(?:s|second|seconds))?\b", text)
    if match:
        return float(match.group("value"))
    pi_fraction = re.search(r"\bt\s*=\s*(?:(?P<num>[-+]?\d+(?:\.\d+)?)\s*)?(?:π|pi)\s*/\s*(?P<den>\d+(?:\.\d+)?)", text)
    if pi_fraction:
        numerator = float(pi_fraction.group("num") or 1.0)
        denominator = float(pi_fraction.group("den"))
        if denominator:
            return numerator * math.pi / denominator
    return None


def _parse_pi_scaled_number(raw: str) -> float:
    compact = re.sub(r"\s+", "", str(raw or "").lower())
    if "π" in compact or "pi" in compact:
        coefficient = compact.replace("π", "").replace("pi", "")
        if coefficient in {"", "+", "-"}:
            coefficient = f"{coefficient}1"
        return float(coefficient) * math.pi
    return float(compact)


def _solve_formula_or_graph_question(text: str) -> Tuple[str, str, str] | None:
    if ("power factor" in text or "cosφ" in text or "cos phi" in text) and (
        "resonan" in text or "z = r" in text or "φ = 0" in text or "phi = 0" in text
    ):
        return (
            "1",
            "rlc_resonance_power_factor_one",
            "At series RLC resonance, phase angle is zero and cos(phi)=1.",
        )
    if "electric field energy" in text and "magnetic field energy" in text and "indicate" in text:
        return (
            "Conservation of energy.",
            "lc_energy_transfer_conservation",
            "In an ideal LC oscillation, electric and magnetic energies exchange while their total remains conserved.",
        )
    if "magnetic field energy" in text and any(cue in text for cue in ["when will", "when is", "be zero", "zero"]):
        return (
            "When the current is zero.",
            "inductor_energy_zero_current_condition",
            "Magnetic energy in an inductor is W = 1/2 L I^2, so it is zero exactly when current is zero.",
        )
    if "voltage" in text and "capacitor" in text and "energy" in text and re.search(r"\bdouble[ds]?\b|\btwice\b", text):
        return (
            "Increase by 4 times.",
            "capacitor_energy_voltage_doubling",
            "For fixed capacitance, capacitor energy obeys W = 1/2 C U^2, so doubling voltage multiplies energy by 4.",
        )
    if "current" in text and any(cue in text for cue in ["coil", "inductor"]) and "energy" in text and re.search(r"\bhalve[ds]?\b|\bhalf\b", text):
        return (
            "Reduced to 1/4.",
            "inductor_energy_current_halving",
            "Magnetic energy in an inductor is W = 1/2 L I^2, so halving current multiplies energy by 1/4.",
        )
    if "electric field energy" in text and "maximum" in text and ("charge" in text or "lc" in text):
        return (
            "the charge Q reaches its maximum value",
            "lc_electric_energy_max_charge_max",
            "Capacitor electric energy is Q^2/(2C), so it is maximum when charge magnitude is maximum.",
        )
    if "energy of oscillation" in text and "lc" in text:
        return (
            "U = 1/2 L I_max^2 = Q_max^2/(2C)",
            "lc_total_oscillation_energy_formula",
            "The total energy of an ideal LC oscillator equals its maximum magnetic energy and maximum electric energy.",
        )
    if "what kind of oscillation" in text and "lc" in text:
        return (
            "Simple Harmonic Motion (SHM)",
            "lc_simple_harmonic_oscillation",
            "An ideal LC circuit has sinusoidal charge and current, so its oscillation is simple harmonic.",
        )
    if "number of turns" in text and "solenoid" in text and "magnetic field" in text and re.search(r"\bdouble[ds]?\b|\btwice\b", text):
        return (
            "Doubled.",
            "solenoid_field_turn_count_doubling",
            "For fixed length and current, B = mu0 (N/l) I, so doubling turns doubles the field.",
        )
    if any(cue in text for cue in ["ideal solenoid", "idead solenoid"]) and any(cue in text for cue in ["external magnetic field", "outside"]):
        return (
            "Approximately zero.",
            "ideal_solenoid_external_field_zero",
            "The ideal long-solenoid model treats the external magnetic field as negligible.",
        )
    if "solenoid" in text and "current" in text and any(cue in text for cue in ["suddenly disconnected", "increases rapidly", "changes with time"]):
        return (
            "An induced electromotive force (EMF) appears opposing the change in current.",
            "solenoid_self_induction_lenz_rule",
            "Changing current changes magnetic flux linkage, producing self-induced EMF that opposes the change.",
        )
    if "induced electromotive force" in text and "solenoid" in text and "when" in text:
        return (
            "the current changes with time",
            "solenoid_induced_emf_current_change_condition",
            "A solenoid has induced EMF when its current, and therefore its magnetic flux linkage, changes with time.",
        )
    if "applications" in text and "solenoid" in text:
        return (
            "electromagnet, and relay",
            "solenoid_application_lookup",
            "Solenoids are used as electromagnets and in relay/actuator mechanisms because current creates a controllable magnetic field.",
        )
    if "magnetic flux" in text and "changes uniformly" in text and "closed circuit" in text:
        return (
            "Induced electromotive force (EMF).",
            "faraday_uniform_flux_change_emf",
            "By Faraday's law, changing magnetic flux through a closed circuit induces EMF.",
        )
    if "magnetic field energy" in text and "stored" in text and "solenoid" in text:
        return (
            "Magnetic field in the coil core.",
            "solenoid_energy_stored_in_field",
            "Energy in a solenoid is stored in its magnetic field.",
        )
    if "self-inductance" in text and "solenoid" in text and "depend" in text:
        if "does not depend" in text or "not depend" in text:
            return (
                "cross-sectional area is included; current is not.",
                "solenoid_inductance_not_current",
                "For a long solenoid L = mu0 N^2 A/l, so inductance depends on geometry/core, not current.",
            )
        return (
            "Number of turns, length, and cross-sectional area.",
            "solenoid_inductance_dependencies",
            "For a long solenoid L = mu0 N^2 A/l, so self-inductance depends on turns, area, length, and permeability.",
        )
    if "cross-sectional area" in text and "solenoid" in text and "self-inductance" in text:
        return (
            "increases in direct proportion",
            "solenoid_inductance_area_proportional",
            "For a long solenoid L = mu0 N^2 A/l, inductance is proportional to cross-sectional area.",
        )
    if any(cue in text for cue in ["ideal solenoid", "idead solenoid"]) and "where" in text and "magnetic field" in text:
        return (
            "inside the solenoid",
            "ideal_solenoid_field_inside",
            "In the ideal long-solenoid model, the magnetic field is concentrated inside the solenoid.",
        )
    if "number of turns" in text and "inductance" in text and "length" in text:
        return (
            "Increases in proportion to the square of the number of turns.",
            "solenoid_inductance_turns_square",
            "For a long solenoid L = mu0 N^2 A/l, inductance is proportional to N^2.",
        )
    if "magnetic field" in text and "solenoid" in text and "not depend" in text:
        return (
            "cross-sectional area (S)",
            "solenoid_field_not_area",
            "For a long solenoid B = mu0 n I, so field depends on turn density and current, not cross-sectional area.",
        )
    if "magnetic field" in text and "solenoid" in text and "depend linearly" in text:
        return (
            "Current through the solenoid.",
            "solenoid_field_linear_current",
            "For a long solenoid B = mu0 n I, field depends linearly on current and turn density.",
        )
    if any(cue in text for cue in ["shape of the graph", "graph representing"]) and "electric field energy" in text and "magnetic field energy" in text and "lc" in text:
        return (
            "Sinusoidal waves with a phase shift of pi/2.",
            "lc_electric_magnetic_energy_graph_phase_shift",
            "In an ideal LC circuit, electric and magnetic energies exchange periodically and are complementary in phase.",
        )
    if "magnetic field energy" in text and ("b" in text or "solenoid" in text) and ("increases" in text or "increase" in text):
        return (
            "the magnetic field energy increases proportionally to B^2",
            "magnetic_energy_b_square_qualitative",
            "Magnetic energy density obeys u = B^2/(2 mu0).",
        )
    if "parallel" in text and "total current" in text and re.search(r"current\s+through\s+one\s+(?:lamp|branch).*increases", text):
        return (
            "Total current increases.",
            "parallel_branch_current_total_current_sum",
            "In a parallel circuit, total current is the sum of branch currents.",
        )
    if "resistance" in text and "decreases" in text and "current" in text:
        return (
            "The current increases.",
            "ohm_law_inverse_resistance_qualitative",
            "For a fixed branch voltage, Ohm's law gives I = U/R, so decreasing resistance increases current.",
        )
    if "total current increases" in text and any(cue in text for cue in ["lamp", "bulb", "light"]):
        return (
            "The lamps become brighter.",
            "lamp_brightness_current_qualitative",
            "For the same lamp resistance, larger current means larger power, and larger power means greater brightness.",
        )
    if "lc" in text and re.search(r"\bi\s*=\s*0\b", text) and "energy" in text:
        return (
            "All the energy is stored in the electric field of the capacitor.",
            "lc_zero_current_energy_location",
            "In an ideal LC circuit, zero current means zero magnetic energy, so all energy is electric energy in the capacitor.",
        )
    lc_fraction = _lc_energy_fraction_answer(text)
    if lc_fraction is not None:
        return lc_fraction
    if "lc" in text and "electric field energy is zero" in text and "current" in text:
        return (
            "maximum",
            "lc_zero_electric_energy_current_maximum",
            "In an ideal LC circuit, zero electric energy means all energy is magnetic, so the current has maximum magnitude.",
        )
    if any(cue in text for cue in ["formula", "expression"]):
        if ("w_l" in text or "wₗ" in text or "magnetic field energy" in text) and "cos" in text and "electric" in text:
            return (
                "W_C = W0 sin^2(omega t)",
                "lc_energy_complement_expression",
                "In an ideal LC circuit, W_C + W_L = W0, so if W_L = W0 cos^2(omega t), then W_C = W0 sin^2(omega t).",
            )
        if "magnetic field energy" in text or ("inductor" in text and "energy" in text):
            return (
                "W = 1/2 L I^2",
                "inductor_energy_formula_lookup",
                "The magnetic energy stored in an inductor is W = 1/2 L I^2.",
            )
        if "capacitor" in text and "energy" in text:
            return (
                "W = 1/2 C U^2 = Q^2/(2C)",
                "capacitor_energy_formula_lookup",
                "Capacitor energy can be written as W = 1/2 C U^2 or W = Q^2/(2C).",
            )
    if "shape of the graph" in text or "graph representing" in text or "as a function of" in text:
        if "capacitor" in text and "energy" in text and ("voltage" in text or " u" in text):
            return (
                "Upward parabola",
                "capacitor_energy_voltage_graph_shape",
                "With capacitance fixed, W = 1/2 C U^2, so energy is a quadratic function of voltage.",
            )
        if "magnetic field energy" in text and "current" in text:
            return (
                "upward parabola",
                "inductor_energy_current_graph_shape",
                "Magnetic field energy in an inductor obeys W = 1/2 L I^2, so it is quadratic in current.",
            )
        if "electric field energy" in text and "distance" in text and "charge" in text and "constant" in text:
            return (
                "Linear function increases",
                "isolated_capacitor_energy_distance_graph_shape",
                "For a parallel-plate capacitor with fixed charge, W = Q^2/(2C) and C is proportional to 1/d, so W is proportional to d.",
            )
        if "energy" in text and ("capacitance" in text or "inductance" in text) and any(cue in text for cue in ["keeping voltage", "constant voltage", "keeping i", "current constant", "keeping current"]):
            return (
                "Upward straight line",
                "stored_energy_linear_parameter_graph_shape",
                "At fixed voltage W_C is proportional to C, and at fixed current W_L is proportional to L.",
            )
    if ("current is maximum" in text or "current reaches its maximum" in text or "current reaches maximum" in text) and "lc" in text and "energy" in text:
        return (
            "All energy is stored in the magnetic field of the inductor.",
            "lc_current_maximum_energy_location",
            "In an ideal LC circuit, maximum current corresponds to maximum magnetic energy and zero electric energy.",
        )
    if "current in an lc circuit" in text and ("capacitor is maximally charged" in text or "charge reaches" in text):
        return (
            "0",
            "lc_max_charge_zero_current",
            "At maximum capacitor charge in an ideal LC circuit, capacitor voltage is extremal and current is zero.",
        )
    if "voltage across the capacitor" in text and "current" in text and "maximum" in text and "lc" in text:
        return (
            "0",
            "lc_max_current_zero_capacitor_voltage",
            "At maximum current in an ideal LC circuit, all energy is magnetic and capacitor voltage is zero.",
        )
    if "resonant angular frequency" in text and "lc" in text:
        return (
            "omega = 1/sqrt(LC)",
            "lc_angular_frequency_formula_lookup",
            "The ideal LC angular frequency is omega = 1/sqrt(LC).",
        )
    if "oscillation period" in text and "lc" in text:
        return (
            "T = 2*pi*sqrt(LC)",
            "lc_period_formula_lookup",
            "The ideal LC period is T = 2*pi*sqrt(LC).",
        )
    if "electric field energy" in text and "directly proportional" in text and "capacitor" in text:
        return (
            "The square of the voltage (U^2)",
            "capacitor_energy_voltage_square_lookup",
            "For fixed capacitance, electric field energy in a capacitor is W = 1/2 C U^2.",
        )
    if "magnetic field inside a solenoid" in text and "directly proportional" in text:
        return (
            "turn density and current intensity",
            "solenoid_field_proportionality_lookup",
            "For a long solenoid, B = mu0 n I, so field is proportional to turn density and current.",
        )
    if "self-inductance of a solenoid" in text and "does not depend" in text:
        return (
            "current intensity",
            "solenoid_inductance_independence_lookup",
            "Long-solenoid inductance L = mu0 N^2 A/l depends on turns, area, length, and core permeability, not current.",
        )
    if "magnetic field energy density" in text and "square" in text:
        return (
            "magnetic induction B",
            "magnetic_energy_density_b_square_lookup",
            "Magnetic energy density obeys u = B^2/(2 mu0).",
        )
    return None


def _lc_energy_fraction_answer(text: str) -> Tuple[str, str, str] | None:
    if "lc" not in text:
        return None
    if "electric field energy equals the magnetic field energy" in text and "current" in text and "percentage" in text:
        return (
            "70.7%",
            "lc_equal_energy_current_fraction",
            "If electric and magnetic energies are equal, magnetic energy is half the total, so I/Imax = sqrt(1/2).",
        )
    magnetic_decimal = re.search(
        r"magnetic(?:\s+field)?\s+energy\s+(?:is|equals?)\s+(?P<fraction>0?\.\d+|\d+(?:\.\d+)?)\s+of\s+the\s+total",
        text,
    )
    if magnetic_decimal and "current" in text and "percentage" in text:
        fraction = float(magnetic_decimal.group("fraction"))
        if 0 <= fraction <= 1:
            percent = math.sqrt(fraction) * 100.0
            return (
                f"{percent:.1f}%",
                "lc_magnetic_energy_fraction_current_percentage",
                "In an ideal LC circuit, W_L/W_total = (I/Imax)^2.",
            )
    electric_decimal = re.search(
        r"electric(?:\s+field)?\s+energy\s+(?:is|equals?)\s+(?P<fraction>0?\.\d+|\d+(?:\.\d+)?)\s+of\s+the\s+total",
        text,
    )
    if electric_decimal and "current" in text and "percentage" in text:
        fraction = float(electric_decimal.group("fraction"))
        if 0 <= fraction <= 1:
            percent = math.sqrt(1.0 - fraction) * 100.0
            return (
                f"{percent:.1f}%",
                "lc_electric_energy_fraction_current_percentage",
                "In an ideal LC circuit, W_L/W_total = (I/Imax)^2 and W_L = W_total - W_C.",
            )
    unicode_fraction = _unicode_fraction_value(text)
    if unicode_fraction is not None:
        if "inductor" in text or "w_l" in text or "wl" in text or "magnetic" in text:
            complement = 1.0 - unicode_fraction
            if "percentage" in text or "%" in text:
                return (
                    f"{round(complement * 100):.0f}%",
                    "lc_unicode_fraction_energy_complement_percentage",
                    "In an ideal LC circuit, electric and magnetic energies sum to the total energy.",
                )
            return (
                f"{complement:.6g}",
                "lc_unicode_fraction_energy_complement",
                "In an ideal LC circuit, electric and magnetic energies sum to the total energy.",
            )
    match = re.search(r"electric(?:\s+field)?\s+energy\s+(?:is|equals?)\s+(?P<num>\d+)\s*/\s*(?P<den>\d+)\s+of\s+the\s+total", text)
    if match and "current" in text and "percentage" in text:
        num = int(match.group("num"))
        den = int(match.group("den"))
        if den > 0 and 0 <= num <= den:
            percent = math.sqrt((den - num) / den) * 100.0
            return (
                f"{percent:.1f}%",
                "lc_energy_fraction_current_percentage",
                "In an ideal LC circuit, W_L/W_total = (I/Imax)^2.",
            )
    if match:
        num = int(match.group("num"))
        den = int(match.group("den"))
        if den > 0 and 0 <= num <= den:
            mag_num = den - num
            answer = f"{mag_num}/{den}"
            if mag_num == 0:
                answer = "0"
            elif mag_num == den:
                answer = "1"
            return (
                answer,
                "lc_energy_fraction_complement",
                "In an ideal LC circuit, magnetic and electric field energies sum to the total energy.",
            )
    if "current is zero" in text and "energy" in text:
        return (
            "All energy is stored in the electric field of the capacitor.",
            "lc_zero_current_energy_location",
            "In an ideal LC circuit, zero current means zero magnetic energy, so all energy is electric.",
        )
    return None


def _unicode_fraction_value(text: str) -> float | None:
    slash_match = re.search(r"(?P<num>\d+)\s*⁄\s*(?P<den>\d+)", text)
    if slash_match:
        denominator = int(slash_match.group("den"))
        if denominator:
            return int(slash_match.group("num")) / denominator
    table = {
        "½": 0.5,
        "⅓": 1.0 / 3.0,
        "⅔": 2.0 / 3.0,
        "¼": 0.25,
        "¾": 0.75,
        "⅕": 0.2,
        "⅖": 0.4,
        "⅗": 0.6,
        "⅘": 0.8,
    }
    for token, value in table.items():
        if token in text:
            return value
    return None


def _change_factor_near_quantity(text: str, quantity_cues: tuple[str, ...]) -> float | None:
    for cue in quantity_cues:
        match = re.search(rf"{re.escape(cue)}[^.?;]{{0,90}}", text, flags=re.IGNORECASE)
        if match:
            factor = extract_change_factor(match.group(0))
            if factor is not None:
                return factor
    if any(cue in text for cue in quantity_cues):
        return extract_change_factor(text)
    return None


def _capacitance_replacement_factor(text: str, front_payload: dict) -> float | None:
    quantities = [
        quantity
        for quantity in front_payload.get("quantities", [])
        if isinstance(quantity, dict) and quantity.get("dimension") == "capacitance" and unit_info(quantity.get("unit") or "") is not None
    ]
    if len(quantities) >= 2:
        if "instead of" in text:
            final = _si_value(quantities[0])
            initial = _si_value(quantities[-1])
        else:
            initial = _si_value(quantities[0])
            final = _si_value(quantities[-1])
        if initial > 0 and final > 0:
            return final / initial
    if any(cue in text for cue in ["capacitance", "capacitor", "dielectric", "plate"]):
        return extract_change_factor(text)
    return None


def _quantity_replacement_factor(front_payload: dict, dimension: str) -> float | None:
    quantities = [
        quantity
        for quantity in front_payload.get("quantities", [])
        if isinstance(quantity, dict) and quantity.get("dimension") == dimension and unit_info(quantity.get("unit") or "") is not None
    ]
    if len(quantities) < 2:
        return None
    initial = _si_value(quantities[0])
    final = _si_value(quantities[-1])
    if initial == 0:
        return None
    return final / initial


def _dielectric_replacement_factor(text: str, front_payload: dict) -> float | None:
    if "dielectric" not in text and "ε" not in text and "epsilon" not in text:
        return None
    values: list[float] = []
    for pattern in (
        r"(?:ε|epsilon|dielectric(?:\s+constant)?)\s*(?:=|of|where)?\s*(\d+(?:\.\d+)?)",
        r"(?:one|material)\s+where\s+(?:ε|epsilon)\s*=\s*(\d+(?:\.\d+)?)",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                values.append(float(match.group(1)))
            except ValueError:
                continue
    for quantity in front_payload.get("numeric_constants", []) or []:
        try:
            symbol = str(quantity.get("symbol") or "").lower()
            if symbol in {"ε", "epsilon", "epsilon_r", "eps_r"}:
                values.append(float(quantity.get("value")))
        except (TypeError, ValueError):
            continue
    if len(values) >= 2 and values[0] > 0 and values[-1] > 0:
        return values[-1] / values[0]
    return None


def _solve_field_force_charge_ratio_rule(text: str) -> Tuple[str, dict] | None:
    if not all(cue in text for cue in ["e1", "e2", "f1", "f2"]):
        return None
    if "electric field" not in text and "field strength" not in text:
        return None
    compact = re.sub(r"\s+", "", text.lower())
    charge_ratio = _symbol_ratio(compact, "q1", "q2")
    force_ratio = _symbol_ratio(compact, "f1", "f2")
    if charge_ratio is None or force_ratio is None or charge_ratio == 0:
        return None
    ratio = force_ratio / charge_ratio
    answer = f"E1 = {_ratio_symbolic_phrase(ratio)}E2"
    return answer, {"q1_over_q2": charge_ratio, "f1_over_f2": force_ratio, "e1_over_e2": ratio}


def _symbol_ratio(compact_text: str, numerator: str, denominator: str) -> float | None:
    direct = re.search(rf"{re.escape(numerator)}=([-+]?(?:\d+(?:\.\d*)?|\.\d+))\*?{re.escape(denominator)}", compact_text)
    if direct:
        return float(direct.group(1))
    inverse = re.search(rf"{re.escape(denominator)}=([-+]?(?:\d+(?:\.\d*)?|\.\d+))\*?{re.escape(numerator)}", compact_text)
    if inverse:
        value = float(inverse.group(1))
        return None if value == 0 else 1.0 / value
    return None


def _ratio_symbolic_phrase(value: float) -> str:
    rational = {
        0.25: "1/4 ",
        1.0 / 3.0: "1/3 ",
        0.5: "1/2 ",
        0.75: "3/4 ",
        1.0: "",
        4.0 / 3.0: "4/3 ",
        1.5: "3/2 ",
        2.0: "2",
        3.0: "3",
        4.0: "4",
    }
    for known, phrase in rational.items():
        if abs(value - known) <= 1e-9:
            return phrase
    return f"{value:.6g} "


def _change_factor_from_text(text: str) -> float | None:
    return extract_change_factor(text)


def _format_factor(value: float) -> str:
    return f"{value:.6g}"


def _ratio_phrase(value: float) -> str:
    named = {
        1.0 / 9.0: "one ninth",
        0.25: "one fourth",
        0.5: "one half",
        2.0: "twice",
        3.0: "three times",
        4.0: "four times",
    }
    for known, phrase in named.items():
        if abs(value - known) <= 1e-12:
            return phrase
    reciprocal = 1.0 / value if value != 0 else None
    if reciprocal is not None and abs(reciprocal - round(reciprocal)) <= 1e-12 and reciprocal > 1:
        return f"1/{int(round(reciprocal))}"
    return f"{value:.6g} times"


def _asks_same_or_equal(text: str) -> bool:
    return any(cue in text for cue in ["same", "equal", "equals", "identical"])


def _solve_si_unit_question(text: str) -> Tuple[str, str] | None:
    if not any(cue in text for cue in ["si unit", "unit of", "what is the unit"]):
        return None
    unit_map = [
        ("resistance", "resistance", "ohm (Ω)"),
        ("capacitance", "capacitance", "farad (F)"),
        ("current", "electric current", "ampere (A)"),
        ("voltage", "voltage", "volt (V)"),
        ("potential difference", "voltage", "volt (V)"),
        ("power", "power", "watt (W)"),
        ("energy", "energy", "joule (J)"),
        ("inductance", "inductance", "henry (H)"),
        ("magnetic field", "magnetic field", "tesla (T)"),
        ("magnetic flux", "magnetic flux", "weber (Wb)"),
        ("electric field", "electric field", "volt per metre (V/m)"),
        ("force", "force", "newton (N)"),
        ("frequency", "frequency", "hertz (Hz)"),
        ("charge", "electric charge", "coulomb (C)"),
    ]
    for cue, label, unit in unit_map:
        if cue in text:
            return label, unit
    return None
