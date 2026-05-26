"""Registry-backed deterministic formula solver.

This module is intentionally small. It executes only formulas that already
exist in ``FORMULA_REGISTRY`` and whose right-hand side can be evaluated through
the safe expression evaluator below. Geometry, conceptual rules, and arbitrary
systems are left to their dedicated engines or fail closed.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ..knowledge.language import extract_change_factor, has_change_factor_cue
from ..knowledge.registries import FORMULA_IDS, FORMULA_REGISTRY, FormulaSpec
from ..knowledge.units import unit_info


@dataclass(frozen=True)
class SolverResult:
    solved: bool
    answer: str
    value: Optional[Any]
    unit: Optional[str]
    formula_id: Optional[str]
    principle_id: Optional[str]
    premises: List[str]
    trace: Dict[str, object]
    confidence: float

    def to_dict(self):
        return {
            "solved": self.solved,
            "answer": self.answer,
            "value": self.value,
            "unit": self.unit,
            "formula_id": self.formula_id,
            "principle_id": self.principle_id,
            "premises": list(self.premises),
            "trace": dict(self.trace),
            "confidence": self.confidence,
        }


def solve_fast(front_payload: dict, route_result, allowed_formula_ids: Iterable[str] | None = None) -> SolverResult:
    front_answer_type = front_payload.get("answer_type_hint")
    if front_answer_type == "multi_output":
        return _unsolved("non_numeric_answer_type_not_formula_path", route_result.task_type)
    if front_answer_type in {"conceptual", "yes_no"} and route_result.answer_type in {"conceptual", "yes_no"}:
        return _unsolved("non_numeric_answer_type_not_formula_path", route_result.task_type)
    if route_result.task_type in {"conceptual", "unknown", "multi_output"}:
        return _unsolved("route_not_direct_formula_path", route_result.task_type)

    ac_waveform = _solve_ac_waveform_quantities(front_payload, route_result)
    if ac_waveform is not None:
        return ac_waveform

    capacitor_energy_ratio = _solve_capacitor_energy_voltage_ratio(front_payload, route_result)
    if capacitor_energy_ratio is not None:
        return capacitor_energy_ratio

    if route_result.task_type == "measurement_error":
        measurement = _solve_measurement_error(front_payload, route_result)
        if measurement is not None:
            return measurement

    if route_result.task_type == "charged_particle_motion":
        stopping_distance = _solve_charged_particle_stopping_distance(front_payload, route_result)
        if stopping_distance is not None:
            return stopping_distance

    if route_result.task_type == "faraday_induction":
        flux_emf = _solve_faraday_flux_per_turn_emf(front_payload, route_result)
        if flux_emf is not None:
            return flux_emf
        induction = _solve_self_induced_emf(front_payload, route_result)
        if induction is not None:
            return induction

    if route_result.task_type == "inductance":
        inductance = _solve_self_inductance_from_emf(front_payload, route_result)
        if inductance is not None:
            return inductance

    rlc_quadrature = _solve_rlc_quadrature_split_circuit(front_payload, route_result)
    if rlc_quadrature is not None:
        return rlc_quadrature

    rlc_section_voltage = _solve_rlc_resonance_capacitor_voltage_from_sections(front_payload, route_result)
    if rlc_section_voltage is not None:
        return rlc_section_voltage

    rlc_inductor_voltage = _solve_rlc_resonance_inductor_voltage(front_payload, route_result)
    if rlc_inductor_voltage is not None:
        return rlc_inductor_voltage

    additive_branch = _solve_additive_branch_quantity(front_payload, route_result)
    if additive_branch is not None:
        return additive_branch

    equilibrium = _solve_electric_equilibrium_mass_angle(front_payload, route_result)
    if equilibrium is not None:
        return equilibrium

    if route_result.task_type == "electric_power":
        ac_power = _solve_rlc_power_impedance(front_payload, route_result)
        if ac_power is not None:
            return ac_power

    if route_result.task_type == "electric_field_point":
        midpoint_inverse = _solve_inverse_square_midpoint_field_expression(front_payload, route_result)
        if midpoint_inverse is not None:
            return midpoint_inverse
        two_charge_angle = _solve_two_charge_field_angle(front_payload, route_result)
        if two_charge_angle is not None:
            return two_charge_angle
        dielectric_scaled = _solve_electric_field_dielectric_scaling(front_payload, route_result)
        if dielectric_scaled is not None:
            return dielectric_scaled
        proportional_scaled = _solve_electric_field_proportional_scaling(front_payload, route_result)
        if proportional_scaled is not None:
            return proportional_scaled
        sheet_field = _solve_parallel_sheet_field(front_payload, route_result)
        if sheet_field is not None:
            return sheet_field

    if route_result.task_type == "capacitance":
        series_unknown = _solve_capacitor_series_unknown_from_final_charge(front_payload, route_result)
        if series_unknown is not None:
            return series_unknown
        scaled_capacitance = _solve_capacitor_geometry_scaled_capacitance(front_payload, route_result)
        if scaled_capacitance is not None:
            return scaled_capacitance

    if route_result.task_type == "capacitor_charge":
        breakdown_charge = _solve_parallel_plate_breakdown_charge(front_payload, route_result)
        if breakdown_charge is not None:
            return breakdown_charge

    if route_result.task_type == "capacitor_energy":
        plate_energy = _solve_parallel_plate_capacitor_energy_or_density(front_payload, route_result)
        if plate_energy is not None:
            return plate_energy
        source_work = _solve_connected_capacitor_source_work(front_payload, route_result)
        if source_work is not None:
            return source_work
        isolated_energy = _solve_isolated_capacitor_energy_scaled(front_payload, route_result)
        if isolated_energy is not None:
            return isolated_energy
        shared_energy = _solve_series_identical_capacitor_energy_sharing(front_payload, route_result)
        if shared_energy is not None:
            return shared_energy
        capacitor_energy = _solve_capacitor_energy_special(front_payload, route_result)
        if capacitor_energy is not None:
            return capacitor_energy

    if route_result.task_type == "inductor_energy":
        lc_extreme = _solve_lc_energy_extreme_statement(front_payload, route_result)
        if lc_extreme is not None:
            return lc_extreme
        lc_symbolic = _solve_lc_energy_symbolic_complement_expression(front_payload, route_result)
        if lc_symbolic is not None:
            return lc_symbolic
        lc_expression = _solve_lc_energy_complement_from_time_expression(front_payload, route_result)
        if lc_expression is not None:
            return lc_expression
        efficiency = _solve_energy_efficiency_from_loss(front_payload, route_result)
        if efficiency is not None:
            return efficiency
        magnetic_special = _solve_solenoid_magnetic_energy_special(front_payload, route_result)
        if magnetic_special is not None:
            return magnetic_special
        lc_energy = _solve_lc_or_inductor_energy_special(front_payload, route_result)
        if lc_energy is not None:
            return lc_energy

    magnetic_special = _solve_solenoid_magnetic_field_or_flux_special(front_payload, route_result)
    if magnetic_special is not None:
        return magnetic_special

    if route_result.task_type == "capacitor_final_voltage":
        lc_voltage = _solve_lc_capacitor_voltage_from_energy_partition(front_payload, route_result)
        if lc_voltage is not None:
            return lc_voltage
        branch_voltage = _solve_parallel_capacitor_voltage_from_branch_charge(front_payload, route_result)
        if branch_voltage is not None:
            return branch_voltage
        sharing_voltage = _solve_capacitor_charge_sharing_voltage(front_payload, route_result)
        if sharing_voltage is not None:
            return sharing_voltage
        isolated_dielectric_voltage = _solve_isolated_capacitor_dielectric_voltage(front_payload, route_result)
        if isolated_dielectric_voltage is not None:
            return isolated_dielectric_voltage
        isolated_distance_voltage = _solve_isolated_capacitor_distance_scaled_voltage(front_payload, route_result)
        if isolated_distance_voltage is not None:
            return isolated_distance_voltage
        connected_voltage = _solve_connected_capacitor_voltage_constant(front_payload, route_result)
        if connected_voltage is not None:
            return connected_voltage
        series_cap_voltage = _solve_capacitor_series_voltage(front_payload, route_result)
        if series_cap_voltage is not None:
            return series_cap_voltage

    rlc_multiplier = _solve_rlc_resonance_frequency_multiplier(front_payload, route_result)
    if rlc_multiplier is not None:
        return rlc_multiplier

    rlc_current_ratio = _solve_rlc_resonance_current_ratio_transform(front_payload, route_result)
    if rlc_current_ratio is not None:
        return rlc_current_ratio

    rlc_transformed = _solve_rlc_frequency_transform(front_payload, route_result)
    if rlc_transformed is not None:
        return rlc_transformed

    if route_result.task_type == "resultant_force":
        resultant = _solve_resultant_force(front_payload, route_result)
        if resultant is not None:
            return resultant

    topology_result = _solve_canonical_topology(front_payload, route_result)
    if topology_result is not None:
        return topology_result

    if _multicharge_goal_requires_spatial_execution(front_payload, route_result):
        return _unsolved("multicharge_force_goal_requires_spatial_grounding", route_result.task_type)

    dims = _by_dimension(front_payload)
    candidates = _select_candidates(front_payload, route_result, dims, allowed_formula_ids=allowed_formula_ids)
    attempts: list[dict] = []
    for spec in candidates:
        executed = _execute_formula_spec(spec, front_payload, dims)
        attempts.append(executed["trace"])
        if not executed["ok"]:
            continue
        value = float(executed["value"])
        answer = _format(value, spec.target_unit)
        return SolverResult(
            solved=True,
            answer=answer,
            value=value,
            unit=spec.target_unit,
            formula_id=spec.formula_id,
            principle_id=spec.principle_id,
            premises=[spec.premise],
            trace={
                "stage": "registry_formula_solver",
                "formula_id": spec.formula_id,
                "expression": spec.expression,
                "target_dimension": spec.target_dimension,
                "inputs": executed["inputs"],
                "constants": executed["constants"],
                "binding_audit": executed["binding_audit"],
                "attempted_formula_ids": [item.get("formula_id") for item in attempts],
            },
            confidence=min(0.9, route_result.confidence),
        )

    return _unsolved(
        "no_registry_formula_executed",
        route_result.task_type,
        {
            "candidate_formula_ids": [spec.formula_id for spec in candidates],
            "attempts": attempts[:8],
        },
    )


def _multicharge_goal_requires_spatial_execution(front_payload: dict, route_result) -> bool:
    if route_result.task_type != "coulomb_force":
        return False
    charges = [
        quantity
        for quantity in front_payload.get("quantities") or []
        if quantity.get("dimension") == "charge" and unit_info(quantity.get("unit") or "") is not None
    ]
    if len(charges) < 3:
        return False
    target_text = " ".join(
        str(goal.get("text") or "")
        for goal in front_payload.get("goals") or []
        if isinstance(goal, dict)
    ).lower()
    if not target_text:
        return False
    symbols = [str(quantity.get("symbol") or "").lower() for quantity in charges]
    target_mentions_non_source = any(
        symbol and symbol in target_text and symbol not in {symbols[0], symbols[1]}
        for symbol in symbols[2:]
    )
    target_mentions_probe = bool(re.search(r"\b(?:third|test|probe|q0|q_0)\s+charge\b|\bacting\s+on\s+q0\b", target_text))
    if not (target_mentions_non_source or target_mentions_probe):
        return False
    lengths = [
        quantity
        for quantity in front_payload.get("quantities") or []
        if quantity.get("dimension") == "length" and unit_info(quantity.get("unit") or "") is not None
    ]
    return len(lengths) < 2


def _solve_canonical_topology(front_payload: dict, route_result) -> SolverResult | None:
    """Execute small canonical circuit topologies without pattern examples.

    This accepts only explicit series/parallel topology already canonicalized by
    the semantic frontend. Ambiguous branches, nodes, and unlabeled component
    collections fail closed so the scalar Ohm/capacitor formulas cannot bind a
    random component.
    """

    relation = _canonical_topology_relation(front_payload)
    if relation is None:
        return None

    target_dimensions = _target_dimensions(front_payload)
    text = str(front_payload.get("canonical_question") or "").lower()

    resistors = _topology_component_quantities(front_payload, "resistance")
    capacitors = _topology_component_quantities(front_payload, "capacitance")
    voltages = _topology_component_quantities(front_payload, "voltage")
    currents = _topology_component_quantities(front_payload, "current")

    if route_result.task_type == "equivalent_resistance" or (
        "resistance" in target_dimensions and _target_requests_equivalent_resistance(text)
    ):
        if len(resistors) >= 2:
            equivalent = _equivalent_resistance(relation, resistors)
            if equivalent["ok"]:
                formula_id = f"{relation}_resistance_equivalent"
                return _topology_solver_result(
                    formula_id=formula_id,
                    relation=relation,
                    value=equivalent["value"],
                    components=resistors,
                    topology=front_payload.get("topology_graph") or {},
                    target_dimension="resistance",
                    confidence=min(0.86, route_result.confidence),
                    extra_trace={"equivalent_resistance_ohm": equivalent["value"]},
                )
            return _topology_unsolved_result(equivalent["issue"], route_result.task_type)

    if route_result.task_type == "capacitance" and "capacitance" in target_dimensions and len(capacitors) >= 2:
        equivalent = _equivalent_capacitance(relation, capacitors)
        if equivalent["ok"]:
            formula_id = f"{relation}_capacitance_equivalent"
            return _topology_solver_result(
                formula_id=formula_id,
                relation=relation,
                value=equivalent["value"],
                components=capacitors,
                topology=front_payload.get("topology_graph") or {},
                target_dimension="capacitance",
                confidence=min(0.86, route_result.confidence),
                extra_trace={"equivalent_capacitance_f": equivalent["value"]},
            )
        return _topology_unsolved_result(equivalent["issue"], route_result.task_type)

    if route_result.task_type == "ohm_law" and len(resistors) >= 2:
        equivalent = _equivalent_resistance(relation, resistors)
        if not equivalent["ok"]:
            return _topology_unsolved_result(equivalent["issue"], route_result.task_type)
        if "current" in target_dimensions and len(voltages) == 1:
            voltage = _si_value(voltages[0])
            value = voltage / equivalent["value"]
            formula_id = f"topology_ohm_current_{relation}_resistance"
            return _topology_solver_result(
                formula_id=formula_id,
                relation=relation,
                value=value,
                components=resistors,
                topology=front_payload.get("topology_graph") or {},
                target_dimension="current",
                confidence=min(0.84, route_result.confidence),
                extra_trace={
                    "equivalent_resistance_ohm": equivalent["value"],
                    "driving_voltage_v": voltage,
                    "source": _quantity_trace(voltages[0]),
                },
            )
        if "voltage" in target_dimensions and len(currents) == 1:
            current = _si_value(currents[0])
            value = current * equivalent["value"]
            formula_id = f"topology_ohm_voltage_{relation}_resistance"
            return _topology_solver_result(
                formula_id=formula_id,
                relation=relation,
                value=value,
                components=resistors,
                topology=front_payload.get("topology_graph") or {},
                target_dimension="voltage",
                confidence=min(0.84, route_result.confidence),
                extra_trace={
                    "equivalent_resistance_ohm": equivalent["value"],
                    "total_current_a": current,
                    "source": _quantity_trace(currents[0]),
                },
            )
    return None


def _solve_resultant_force(front_payload: dict, route_result) -> SolverResult | None:
    forces = _topology_component_quantities(front_payload, "force")
    if len(forces) == 1 and re.search(r"\beach\b", str(front_payload.get("canonical_question") or ""), flags=re.IGNORECASE):
        forces = [forces[0], forces[0]]
    angle_result = _solve_resultant_force_angle(front_payload, route_result, forces)
    if angle_result is not None:
        return angle_result
    if len(forces) < 2:
        return None
    f1 = _si_value(forces[0])
    f2 = _si_value(forces[1])
    theta = _resultant_angle_rad(front_payload)
    if theta is None:
        return None
    value = math.sqrt(max(0.0, f1 * f1 + f2 * f2 + 2.0 * f1 * f2 * math.cos(theta)))
    spec = FORMULA_REGISTRY["resultant_two_forces"]
    theta_deg = math.degrees(theta)
    answer = _format(value, spec.target_unit)
    if abs(theta - math.pi) <= 1e-9 and abs(f1 - f2) > 1e-12:
        answer = f"{answer}, toward the larger force"
    return SolverResult(
        solved=True,
        answer=answer,
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {
                "F1": _quantity_trace(forces[0]),
                "F2": _quantity_trace(forces[1]),
                "theta": {
                    "dimension": "angle",
                    "raw_text": _resultant_angle_source(front_payload),
                    "unit": "rad",
                    "si_value": theta,
                    "symbol": "theta",
                    "binding_policy": "deterministic_angle_context",
                    "theta_deg": theta_deg,
                },
            },
            "constants": {},
            "binding_audit": {
                "F1": {"policy": "ordered_force_component", "selected_index": 0},
                "F2": {"policy": "ordered_force_component", "selected_index": 1},
                "theta": {"policy": "angle_quantity_or_textual_direction", "theta_deg": theta_deg},
            },
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.88, route_result.confidence),
    )


def _solve_resultant_force_angle(front_payload: dict, route_result, forces: list[dict]) -> SolverResult | None:
    target_dimensions = _target_dimensions(front_payload)
    text = str(front_payload.get("canonical_question") or "").lower()
    if "angle" not in target_dimensions:
        return None
    if len(forces) < 2:
        return None
    resultant, components = _resultant_force_angle_roles(front_payload, forces)
    if resultant is None or len(components) < 2:
        return None
    f1 = _si_value(components[0])
    f2 = _si_value(components[1])
    r = _si_value(resultant)
    denominator = 2.0 * f1 * f2
    if denominator <= 0:
        return None
    cos_theta = (r * r - f1 * f1 - f2 * f2) / denominator
    if cos_theta < -1.0 - 1e-9 or cos_theta > 1.0 + 1e-9:
        return None
    theta = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))
    spec = FORMULA_REGISTRY["resultant_two_forces_angle"]
    return SolverResult(
        solved=True,
        answer=_format(theta, spec.target_unit),
        value=theta,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {
                "F1": _quantity_trace(components[0]),
                "F2": _quantity_trace(components[1]),
                "R": _quantity_trace(resultant),
            },
            "constants": {},
            "binding_audit": {
                "F1": {"policy": "component_force_for_inverse_resultant", "selected_index": 0},
                "F2": {"policy": "component_force_for_inverse_resultant", "selected_index": 1},
                "R": {"policy": "resultant_force_context", "raw_text": resultant.get("raw_text")},
            },
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.84, route_result.confidence),
    )


def _resultant_force_angle_roles(front_payload: dict, forces: list[dict]) -> tuple[dict | None, list[dict]]:
    text = str(front_payload.get("canonical_question") or "")
    resultant: dict | None = None
    components: list[dict] = []
    for quantity in forces:
        span = quantity.get("span") or (0, 0)
        before = text[max(0, int(span[0]) - 48) : int(span[0])].lower()
        after = text[int(span[1]) : min(len(text), int(span[1]) + 24)].lower()
        if any(cue in before for cue in ["resultant", "net force"]) or re.search(r"\b(?:is|equals?)\s+(?:the\s+)?resultant\b", after):
            resultant = quantity
        else:
            components.append(quantity)
    if resultant is not None:
        if len(components) == 1 and re.search(r"\b(?:each|equal)\b", text, flags=re.IGNORECASE):
            components = [components[0], components[0]]
    return resultant, components


def _solve_capacitor_energy_voltage_ratio(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text or "energy" not in text:
        return None
    target_text = " ".join(str(goal.get("text") or "") for goal in front_payload.get("goals") or [] if isinstance(goal, dict)).lower()
    if "%" not in text and "percent" not in text and "percentage" not in text and "percentage" not in target_text:
        return None
    voltages = _topology_component_quantities(front_payload, "voltage")
    if len(voltages) < 2:
        return None
    u_initial = _si_value(voltages[0])
    u_final = _si_value(voltages[-1])
    if u_initial <= 0:
        return None
    value = (u_final / u_initial) ** 2 * 100.0
    spec = FORMULA_REGISTRY["capacitor_energy_voltage_percent"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"U_initial": _quantity_trace(voltages[0]), "U_final": _quantity_trace(voltages[-1])},
            "constants": {},
            "binding_audit": {"policy": "capacitor_energy_ratio_from_voltage_ratio"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.84, route_result.confidence),
    )


def _solve_ac_waveform_quantities(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["cos", "sin"]) or "rlc" not in text:
        return None
    waveform = _parse_time_dependent_voltage(front_payload)
    if waveform is None or waveform.get("omega") is None:
        return None
    target_dimensions = _target_dimensions(front_payload)
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    omega = float(waveform["omega"])
    u_rms = float(waveform["amplitude"]) / math.sqrt(2.0)
    l_value = _parse_assignment_with_unit(text, "l", "h")
    inductance = (
        _synthetic_quantity("L", l_value, "H", "inductance", "parsed L assignment")
        if l_value is not None
        else _quantity_by_symbol_or_context(front_payload, "inductance", {"l"}, context_cues={"inductance", "inductor"})
    )
    c_value = _parse_assignment_with_unit(text, "c", "f")
    capacitance = (
        _synthetic_quantity("C", c_value, "F", "capacitance", "parsed C assignment")
        if c_value is not None
        else _quantity_by_symbol_or_context(front_payload, "capacitance", {"c"}, context_cues={"capacitance", "capacitor"})
    )
    resistance = _quantity_by_symbol_or_context(front_payload, "resistance", {"r"}, reject_symbols={"xl", "xc", "z"})
    xl = _si_value(inductance) * omega if inductance is not None else None
    xc = 1.0 / (omega * _si_value(capacitance)) if capacitance is not None and _si_value(capacitance) > 0 else None

    formula_id: str | None = None
    value: float | None = None
    target_dimension: str | None = None
    inputs: dict[str, dict] = {"source_waveform": {"source": "time_dependent_voltage", **waveform, "u_rms": u_rms}}

    if any(cue in target_text for cue in ["inductive reactance", "x_l", "xl"]):
        if xl is None or inductance is None:
            return None
        formula_id = "inductive_reactance"
        value = xl
        target_dimension = "resistance"
        inputs["L"] = _quantity_trace(inductance)
    elif any(cue in target_text for cue in ["capacitive reactance", "x_c", "xc"]):
        if xc is None or capacitance is None:
            return None
        formula_id = "capacitive_reactance"
        value = xc
        target_dimension = "resistance"
        inputs["C"] = _quantity_trace(capacitance)
    elif "angular_frequency" in target_dimensions or "angular frequency" in target_text or re.search(r"\bω\b|\bomega\b", target_text):
        formula_id = "ac_source_angular_frequency"
        value = omega
        target_dimension = "angular_frequency"
    elif "current" in target_dimensions or "current" in target_text or re.search(r"\b(?:calculate|find|determine)[^.?]*(?:effective|rms)?\s*current\b", text):
        if resistance is None or xl is None or xc is None or inductance is None or capacitance is None:
            return None
        impedance = math.sqrt(_si_value(resistance) ** 2 + (xl - xc) ** 2)
        if impedance <= 0:
            return None
        formula_id = "rlc_current_from_rlcf_voltage"
        value = u_rms / impedance
        target_dimension = "current"
        inputs.update({"R": _quantity_trace(resistance), "L": _quantity_trace(inductance), "C": _quantity_trace(capacitance)})
    elif ("rms" in target_text or "effective" in target_text) and "voltage" in target_dimensions and not any(cue in target_text for cue in ["inductor", "capacitor", "resistor"]):
        formula_id = "ac_source_rms_from_sinusoid"
        value = u_rms
        target_dimension = "voltage"
    elif "impedance" in target_text or "z" in target_text:
        if resistance is None or xl is None or xc is None or inductance is None or capacitance is None:
            return None
        formula_id = "rlc_impedance_from_rlcf"
        value = math.sqrt(_si_value(resistance) ** 2 + (xl - xc) ** 2)
        target_dimension = "resistance"
        inputs.update({"R": _quantity_trace(resistance), "L": _quantity_trace(inductance), "C": _quantity_trace(capacitance)})
    elif "current" in target_dimensions or "current" in target_text or re.search(r"\b(?:effective|rms)\s+current\b", text):
        if resistance is None or xl is None or xc is None or inductance is None or capacitance is None:
            return None
        impedance = math.sqrt(_si_value(resistance) ** 2 + (xl - xc) ** 2)
        if impedance <= 0:
            return None
        formula_id = "rlc_current_from_rlcf_voltage"
        value = u_rms / impedance
        target_dimension = "current"
        inputs.update({"R": _quantity_trace(resistance), "L": _quantity_trace(inductance), "C": _quantity_trace(capacitance)})
    elif "voltage" in target_dimensions and any(cue in target_text for cue in ["inductor", "ul", "u_l", "capacitor", "uc", "u_c"]):
        if resistance is None or xl is None or xc is None:
            return None
        impedance = math.sqrt(_si_value(resistance) ** 2 + (xl - xc) ** 2)
        if impedance <= 0:
            return None
        current = u_rms / impedance
        if any(cue in target_text for cue in ["inductor", "ul", "u_l"]):
            if inductance is None:
                return None
            formula_id = "rlc_component_voltage"
            value = current * xl
            inputs["L"] = _quantity_trace(inductance)
        else:
            if capacitance is None:
                return None
            formula_id = "rlc_component_voltage"
            value = current * xc
            inputs["C"] = _quantity_trace(capacitance)
        target_dimension = "voltage"
        inputs["R"] = _quantity_trace(resistance)
    else:
        return None

    if formula_id is None or value is None or target_dimension is None:
        return None
    spec = FORMULA_REGISTRY[formula_id]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": target_dimension,
            "inputs": inputs,
            "constants": {"omega": omega, "U_rms": u_rms, **({"XL": xl} if xl is not None else {}), **({"XC": xc} if xc is not None else {})},
            "binding_audit": {"policy": "sinusoidal_source_parameter_extraction_and_rlc_phasor_execution"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.82, route_result.confidence),
    )

    if len(forces) >= 3:
        return forces[-1], forces[:2]
    if len(forces) == 2 and re.search(r"\b(?:each|equal)\b", text, flags=re.IGNORECASE):
        return forces[-1], [forces[0], forces[0]]
    return None, components


def _solve_measurement_error(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    average = _solve_measurement_average(front_payload, route_result)
    if average is not None:
        return average
    absolute_random = _solve_absolute_random_error(front_payload, route_result)
    if absolute_random is not None:
        return absolute_random
    if not any(cue in text for cue in ["uncertainty", "percent error", "percentage error", "relative error", "relative uncertainty", "measurement error"]):
        return None
    pairs = _measurement_reference_uncertainty_pairs(front_payload)
    if not pairs:
        return None
    if "power" in text and len(pairs) >= 2:
        relative_terms = [
            abs(_si_value(uncertainty)) / abs(_si_value(reference))
            for reference, uncertainty in pairs
            if abs(_si_value(reference)) > 0
        ]
        if len(relative_terms) < 2:
            return None
        value = sum(relative_terms) * 100.0
        reference, uncertainty = pairs[0]
        input_trace = {
            f"reference_value_{index}": _quantity_trace(pair_reference)
            for index, (pair_reference, _) in enumerate(pairs, start=1)
        }
        input_trace.update(
            {
                f"absolute_uncertainty_{index}": _quantity_trace(pair_uncertainty)
                for index, (_, pair_uncertainty) in enumerate(pairs, start=1)
            }
        )
        binding_policy = "product_relative_uncertainty_sum"
    else:
        reference, uncertainty = pairs[0]
        reference_value = abs(_si_value(reference))
        uncertainty_value = abs(_si_value(uncertainty))
        if reference_value <= 0:
            return None
        value = uncertainty_value / reference_value * 100.0
        input_trace = {
            "reference_value": _quantity_trace(reference),
            "absolute_uncertainty": _quantity_trace(uncertainty),
        }
        binding_policy = "explicit_same_dimension_reference_and_uncertainty"
    if value < 0:
        return None
    spec = FORMULA_REGISTRY["measurement_error_direct"]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "measurement_uncertainty_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": input_trace,
            "constants": {},
            "binding_audit": {
                "policy": binding_policy,
                "reference_raw_text": reference.get("raw_text"),
                "uncertainty_raw_text": uncertainty.get("raw_text"),
            },
        },
        confidence=min(0.78, route_result.confidence),
    )


def _solve_measurement_average(front_payload: dict, route_result) -> SolverResult | None:
    target_text = " ".join(front_payload.get("target_hints") or []).lower()
    if "average" not in target_text and "mean" not in target_text:
        return None
    if any(cue in target_text for cue in ["error", "uncertainty"]):
        return None
    quantities = [
        quantity
        for quantity in sorted(front_payload.get("quantities", []), key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") not in {None, "percent", "dimensionless", "constant"}
        and unit_info(quantity.get("unit") or "") is not None
    ]
    by_dimension: dict[str, list[dict]] = defaultdict(list)
    for quantity in quantities:
        by_dimension[str(quantity.get("dimension"))].append(quantity)
    repeated = sorted((values for values in by_dimension.values() if len(values) >= 2), key=lambda values: -len(values))
    if not repeated:
        return None
    measurements = repeated[0]
    values_si = [_si_value(quantity) for quantity in measurements]
    mean_si = sum(values_si) / len(values_si)
    first_unit = unit_info(measurements[0].get("unit") or "")
    if first_unit is None or first_unit.si_factor <= 0:
        return None
    display_value = mean_si / first_unit.si_factor
    spec = FORMULA_REGISTRY["measurement_average"]
    return SolverResult(
        solved=True,
        answer=_format(display_value, first_unit.canonical),
        value=mean_si,
        unit=first_unit.canonical,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "measurement_uncertainty_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": "mean_value",
            "inputs": {f"measurement_{index}": _quantity_trace(quantity) for index, quantity in enumerate(measurements, start=1)},
            "constants": {},
            "binding_audit": {"policy": "arithmetic_mean_from_repeated_measurements", "measurement_count": len(measurements)},
        },
        confidence=min(0.8, route_result.confidence),
    )


def _solve_absolute_random_error(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["random error", "absolute error", "average absolute error"]):
        return None
    quantities = [
        quantity
        for quantity in sorted(front_payload.get("quantities", []), key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") not in {None, "percent", "dimensionless", "constant"}
        and unit_info(quantity.get("unit") or "") is not None
    ]
    by_dimension: dict[str, list[dict]] = defaultdict(list)
    for quantity in quantities:
        by_dimension[str(quantity.get("dimension"))].append(quantity)
    repeated = sorted(
        (values for values in by_dimension.values() if len(values) >= 2),
        key=lambda values: -len(values),
    )
    if not repeated:
        return None
    measurements = repeated[0]
    values_si = [_si_value(quantity) for quantity in measurements]
    if "average absolute error" in text or "mean absolute error" in text:
        mean_value = sum(values_si) / len(values_si)
        value_si = sum(abs(value - mean_value) for value in values_si) / len(values_si)
        policy = "mean_absolute_deviation_from_repeated_measurements"
    else:
        value_si = (max(values_si) - min(values_si)) / 2.0
        policy = "half_range_random_error_from_repeated_measurements"
    if value_si < 0 or not math.isfinite(value_si):
        return None
    first_unit = unit_info(measurements[0].get("unit") or "")
    if first_unit is None or first_unit.si_factor <= 0:
        return None
    display_value = value_si / first_unit.si_factor
    spec = FORMULA_REGISTRY["measurement_absolute_error"]
    return SolverResult(
        solved=True,
        answer=_format(display_value, first_unit.canonical),
        value=value_si,
        unit=first_unit.canonical,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "measurement_uncertainty_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": "uncertainty",
            "inputs": {f"measurement_{index}": _quantity_trace(quantity) for index, quantity in enumerate(measurements, start=1)},
            "constants": {},
            "binding_audit": {
                "policy": policy,
                "measurement_count": len(measurements),
                "dimension": measurements[0].get("dimension"),
                "display_unit": first_unit.canonical,
            },
        },
        confidence=min(0.74, route_result.confidence),
    )


def _measurement_reference_uncertainty_pairs(front_payload: dict) -> list[tuple[dict, dict]]:
    quantities = sorted(front_payload.get("quantities", []), key=lambda item: item.get("span") or (10**9, 10**9))
    by_dimension: dict[str, list[dict]] = defaultdict(list)
    for quantity in quantities:
        dimension = quantity.get("dimension")
        if not dimension or dimension in {"percent", "dimensionless", "constant"}:
            continue
        if unit_info(quantity.get("unit") or "") is None:
            continue
        by_dimension[dimension].append(quantity)
    pairs: list[tuple[dict, dict]] = []
    for values in by_dimension.values():
        if len(values) < 2:
            continue
        uncertainty_candidates = [quantity for quantity in values if _quantity_is_uncertainty(quantity)]
        reference_candidates = [quantity for quantity in values if quantity not in uncertainty_candidates]
        if uncertainty_candidates and reference_candidates:
            pairs.append((reference_candidates[0], uncertainty_candidates[0]))
            continue
        text = str(front_payload.get("canonical_question") or "").lower()
        if "uncertainty" in text or "error" in text:
            pairs.append((values[0], values[1]))
    return pairs


def _quantity_is_uncertainty(quantity: dict) -> bool:
    symbol = str(quantity.get("symbol") or "").lower()
    raw = str(quantity.get("raw_text") or "").lower()
    context = str(quantity.get("context") or "").lower()
    return (
        symbol.startswith(("d", "delta"))
        or "δ" in symbol
        or "Δ" in str(quantity.get("symbol") or "")
        or any(cue in raw for cue in ["uncertainty", "error"])
        or any(cue in context for cue in ["uncertainty", "error", "±", "+/-"])
    )


def _solve_faraday_flux_per_turn_emf(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if route_result.task_type != "faraday_induction" and "induced" not in text:
        return None
    turns = _largest_count_quantity(front_payload)
    flux = _quantity_by_context_strict(front_payload, "magnetic_flux", {"flux", "per turn"})
    time_quantity = _quantity_by_symbol_or_context(front_payload, "time", {"t", "delta_t"}, context_cues={"time", "in"})
    if turns is None or flux is None or time_quantity is None:
        return None
    n = _si_value(turns)
    delta_phi = abs(_si_value(flux))
    delta_t = _si_value(time_quantity)
    if n <= 0 or delta_phi < 0 or delta_t <= 0:
        return None
    value = n * delta_phi / delta_t
    spec = FORMULA_REGISTRY["faraday_flux_emf"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"N": _quantity_trace(turns), "Delta_phi": _quantity_trace(flux), "Delta_t": _quantity_trace(time_quantity)},
            "constants": {},
            "binding_audit": {"policy": "faraday_emf_from_flux_per_turn_change"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.82, route_result.confidence),
    )


def _solve_self_induced_emf(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["self-inductance", "self inductance", "inductance", "solenoid"]):
        return None
    if not any(cue in text for cue in ["current", "emf", "electromotive force", "induced"]):
        return None
    inductance = _quantity_by_symbol_or_context(front_payload, "inductance", {"l"})
    time_quantity = _quantity_by_symbol_or_context(front_payload, "time", {"t", "delta_t", "dt"}, context_cues={"time", "second"})
    currents = _topology_component_quantities(front_payload, "current")
    if inductance is None or time_quantity is None or len(currents) < 2:
        return None
    delta_i = abs(_si_value(currents[-1]) - _si_value(currents[0]))
    delta_t = abs(_si_value(time_quantity))
    inductance_value = _si_value(inductance)
    if delta_t <= 0 or inductance_value < 0:
        return None
    value = inductance_value * delta_i / delta_t
    spec = FORMULA_REGISTRY["self_induced_emf"]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {
                "L": _quantity_trace(inductance),
                "I_initial": _quantity_trace(currents[0]),
                "I_final": _quantity_trace(currents[-1]),
                "delta_t": _quantity_trace(time_quantity),
            },
            "constants": {},
            "binding_audit": {
                "policy": "ordered_current_change_from_temporal_language",
                "delta_i": delta_i,
                "delta_t": delta_t,
            },
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.84, route_result.confidence),
    )


def _solve_self_inductance_from_emf(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["self-inductance", "self inductance", "inductance"]):
        return None
    if not any(cue in text for cue in ["emf", "electromotive force", "induced"]):
        return None
    emf = _quantity_by_symbol_or_context(
        front_payload,
        "voltage",
        {"emf", "e"},
        context_cues={"emf", "electromotive force", "induced voltage"},
    )
    time_quantity = _quantity_by_symbol_or_context(front_payload, "time", {"t", "delta_t", "dt"}, context_cues={"time", "second"})
    currents = _topology_component_quantities(front_payload, "current")
    if emf is None or time_quantity is None or not currents:
        return None
    if len(currents) >= 2:
        delta_i = abs(_si_value(currents[-1]) - _si_value(currents[0]))
        current_trace = {"I_initial": _quantity_trace(currents[0]), "I_final": _quantity_trace(currents[-1])}
        policy = "self_inductance_from_two_current_states"
    else:
        delta_i = abs(_si_value(currents[0]))
        current_trace = {"delta_i": _quantity_trace(currents[0])}
        policy = "self_inductance_from_explicit_current_change"
    delta_t = abs(_si_value(time_quantity))
    if delta_i <= 0 or delta_t <= 0:
        return None
    value = abs(_si_value(emf)) * delta_t / delta_i
    spec = FORMULA_REGISTRY["self_inductance_from_emf"]
    inputs = {"emf": _quantity_trace(emf), "delta_t": _quantity_trace(time_quantity), **current_trace}
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": inputs,
            "constants": {},
            "binding_audit": {"policy": policy, "delta_i": delta_i, "delta_t": delta_t},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.82, route_result.confidence),
    )


def _solve_rlc_power_impedance(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not (any(cue in text for cue in ["rlc", "ac circuit", "impedance", "reactance", "power factor"]) or re.search(r"\bz\s*=", text)):
        return None
    target_dimensions = _target_dimensions(front_payload)
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    if "power" not in target_dimensions and "power" not in target_text:
        return None
    voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v", "emf"})
    resistance = _quantity_by_symbol_or_context(front_payload, "resistance", {"r"}, reject_symbols={"z", "xl", "x_l", "xc", "x_c"})
    impedance = _quantity_by_symbol_or_context(front_payload, "resistance", {"z"}, context_cues={"impedance"})
    if impedance is not None and not _quantity_matches_symbol_or_context(impedance, {"z"}, {"impedance"}):
        impedance = None
    if voltage is None or resistance is None:
        return None
    r = _si_value(resistance)
    if r <= 0:
        return None
    resonance = "resonan" in text
    if impedance is None and resonance:
        z = r
        formula_id = "rlc_power_resonance"
        inputs = {"U": _quantity_trace(voltage), "R": _quantity_trace(resistance)}
        policy = "series_rlc_resonance_real_power"
    elif impedance is not None:
        z = _si_value(impedance)
        formula_id = "rlc_power_impedance"
        inputs = {"U": _quantity_trace(voltage), "R": _quantity_trace(resistance), "Z": _quantity_trace(impedance)}
        policy = "series_ac_real_power_from_impedance_and_resistance"
    else:
        return None
    if z <= 0:
        return None
    value = (_si_value(voltage) ** 2) * r / (z * z)
    spec = FORMULA_REGISTRY[formula_id]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": inputs,
            "constants": {},
            "binding_audit": {"policy": policy},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.82, route_result.confidence),
    )


def _solve_rlc_resonance_capacitor_voltage_from_sections(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    target_text = " ".join(front_payload.get("target_hints") or []).lower()
    if "resonan" not in text or not any(cue in target_text for cue in ["capacitor", "uc", "u_c"]):
        return None
    if not any(cue in text for cue in ["r-c", "rc", "c-l", "cl", "series combination", "section", "combination"]):
        return None
    voltages = _topology_component_quantities(front_payload, "voltage")
    if len(voltages) < 2:
        return None
    source = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v"}, context_cues={"applied", "source", "entire", "whole", "circuit"})
    values = [_si_value(quantity) for quantity in voltages if _si_value(quantity) > 0]
    if not values:
        return None
    source_value = _si_value(source) if source is not None else min(values)
    section_candidates = [value for value in values if not math.isclose(value, source_value, rel_tol=1e-9, abs_tol=1e-12)]
    if not section_candidates:
        return None
    section_value = max(section_candidates)
    active_component = abs(source_value - section_value) if "internal resistance" in text else source_value
    radicand = section_value * section_value - active_component * active_component
    if radicand < -1e-9:
        return None
    value = math.sqrt(max(0.0, radicand))
    spec = FORMULA_REGISTRY["rlc_resonance_capacitor_voltage_from_sections"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"U_source": {"dimension": "voltage", "si_value": source_value, "unit": "V"}, "U_section": {"dimension": "voltage", "si_value": section_value, "unit": "V"}},
            "constants": {},
            "binding_audit": {"policy": "rlc_resonance_section_voltage_quadrature", "active_component_v": active_component},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.82, route_result.confidence),
    )


def _solve_rlc_resonance_inductor_voltage(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    target_text = " ".join(front_payload.get("target_hints") or []).lower()
    if "resonan" not in text or not any(cue in target_text for cue in ["inductor", "ul", "u_l", "voltage across l"]):
        return None
    voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v"}, context_cues={"source", "applied", "voltage"})
    resistance = _quantity_by_symbol_or_context(front_payload, "resistance", {"r"}, reject_symbols={"xl", "x_l", "xc", "x_c", "z"})
    inductance = _quantity_by_symbol_or_context(front_payload, "inductance", {"l"}, context_cues={"inductance", "inductor"})
    capacitance = _quantity_by_symbol_or_context(front_payload, "capacitance", {"c"}, context_cues={"capacitance", "capacitor"})
    if voltage is None or resistance is None or inductance is None or capacitance is None:
        return None
    u = _si_value(voltage)
    r = _si_value(resistance)
    l_value = _si_value(inductance)
    c_value = _si_value(capacitance)
    if u <= 0 or r <= 0 or l_value <= 0 or c_value <= 0:
        return None
    x_l = math.sqrt(l_value / c_value)
    value = (u / r) * x_l
    spec = FORMULA_REGISTRY["rlc_resonance_inductor_voltage"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"U": _quantity_trace(voltage), "R": _quantity_trace(resistance), "L": _quantity_trace(inductance), "C": _quantity_trace(capacitance)},
            "constants": {"X_L_resonance": x_l},
            "binding_audit": {"policy": "series_rlc_resonance_inductor_voltage"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.84, route_result.confidence),
    )


def _solve_rlc_quadrature_split_circuit(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if route_result.task_type not in {"ohm_law", "electric_power", "power_factor", "capacitor_final_voltage"}:
        return None
    if not (
        "lcω" in text
        or "lcw" in text
        or "lc omega" in text
        or re.search(r"\blc\s*(?:ω|w|omega)\s*\^?\s*2\s*=\s*1\b", text)
    ):
        return None
    if not any(cue in text for cue in ["quadrature", "90", "perpendicular", "out of phase"]):
        return None
    resistors = [
        quantity
        for quantity in _topology_component_quantities(front_payload, "resistance")
        if not re.search(r"\b(?:z|xl|x_l|xc|x_c|impedance|reactance)\b", f"{quantity.get('symbol') or ''} {quantity.get('context') or ''}", re.IGNORECASE)
    ]
    target_dimensions = _target_dimensions(front_payload)
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v", "emf", "u_ab"}, context_cues={"rms", "effective", "applied", "source", "across ab"})
    powers = _topology_component_quantities(front_payload, "power")

    if len(resistors) == 1 and voltage is not None and powers and re.search(r"\br\s*2\b|value of r2|resistor r2", target_text + " " + text):
        r1 = _si_value(resistors[0])
        power = _si_value(powers[0])
        if r1 <= 0 or power <= 0:
            return None
        total_r = (_si_value(voltage) ** 2) / power
        value = total_r - r1
        if value <= 0:
            return None
        spec = FORMULA_REGISTRY["rlc_quadrature_unknown_resistance"]
        return SolverResult(
            True,
            _format(value, spec.target_unit),
            value,
            spec.target_unit,
            spec.formula_id,
            spec.principle_id,
            [spec.premise],
            {
                "stage": "registry_formula_solver",
                "formula_id": spec.formula_id,
                "expression": spec.expression,
                "target_dimension": spec.target_dimension,
                "inputs": {"U": _quantity_trace(voltage), "P": _quantity_trace(powers[0]), "R_known": _quantity_trace(resistors[0])},
                "constants": {},
                "binding_audit": {
                    "policy": "rlc_quadrature_unknown_series_resistance_from_total_power",
                    "equivalent_resistance": total_r,
                },
                "attempted_formula_ids": [spec.formula_id],
            },
            min(0.82, route_result.confidence),
        )

    if len(resistors) < 2:
        return None
    r1 = _si_value(resistors[0])
    r2 = _si_value(resistors[1])
    total_r = r1 + r2
    if total_r <= 0:
        return None
    formula_id: str | None = None
    value: float | None = None
    target_dimension: str | None = None
    if route_result.task_type == "power_factor" or "power factor" in target_text or "power factor" in text:
        formula_id = "power_factor"
        value = 1.0
        target_dimension = "dimensionless"
    else:
        if voltage is None:
            return None
        if route_result.task_type == "electric_power" or "power" in target_dimensions or "power" in target_text:
            if powers and any(cue in text for cue in ["same voltage", "same rms voltage", "same voltage is applied"]):
                formula_id = "rlc_quadrature_power_transfer_same_voltage"
                value = _si_value(powers[0])
            else:
                formula_id = "rlc_quadrature_segment_power"
                value = (_si_value(voltage) ** 2) / total_r
            target_dimension = "power"
        elif "current" in target_dimensions or "current" in target_text or "current" in text:
            formula_id = "rlc_quadrature_segment_current"
            value = _si_value(voltage) / total_r
            target_dimension = "current"
        elif "voltage" in target_dimensions and any(cue in target_text for cue in ["mb", "am", "segment"]):
            formula_id = "rlc_quadrature_segment_voltage"
            selected_resistor = r2 if "mb" in target_text else r1
            value = _si_value(voltage) * math.sqrt(selected_resistor / total_r)
            target_dimension = "voltage"
        else:
            return None
    if formula_id is None or value is None or target_dimension is None:
        return None
    spec = FORMULA_REGISTRY[formula_id]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": target_dimension,
            "inputs": {
                **({"U": _quantity_trace(voltage)} if voltage is not None else {}),
                **({"P_total": _quantity_trace(powers[0])} if formula_id == "rlc_quadrature_power_transfer_same_voltage" and powers else {}),
                "R1": _quantity_trace(resistors[0]),
                "R2": _quantity_trace(resistors[1]),
            },
            "constants": {},
            "binding_audit": {
                "policy": "rlc_two_section_quadrature_cancellation",
                "LC_omega_squared": 1,
                "phase_condition": "u_AM_perpendicular_u_MB",
                "equivalent_resistance": total_r,
            },
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.84, route_result.confidence),
    )


def _solve_additive_branch_quantity(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if route_result.task_type == "electric_current":
        if not any(cue in text for cue in ["branch", "ammeter", "lamp", "bulb", "parallel", "total current", "main current"]):
            return None
        currents = _topology_component_quantities(front_payload, "current")
        if not currents:
            return None
        target_text = " ".join(front_payload.get("target_hints", [])).lower()
        total_indices = [
            index
            for index, quantity in enumerate(currents)
            if re.search(r"\b(?:total|main|source|entire)\s+current\b", f"{quantity.get('raw_text') or ''} {quantity.get('context') or ''}", re.IGNORECASE)
        ]
        if total_indices and len(currents) >= 2 and not any(cue in target_text for cue in ["total", "main", "source"]):
            total_index = total_indices[0]
            total = _si_value(currents[total_index])
            known_sum = sum(_si_value(quantity) for index, quantity in enumerate(currents) if index != total_index)
            value = total - known_sum
            formula_id = "current_branch_difference"
            policy = "missing_branch_current_from_total_minus_known_branches"
            used = {"I_total": _quantity_trace(currents[total_index])}
            used.update({f"I_known_{i}": _quantity_trace(quantity) for i, quantity in enumerate(currents) if i != total_index})
        elif len(currents) == 1 and any(cue in text for cue in ["removed", "disconnected", "burned out", "switched off"]) and any(cue in target_text or cue in text for cue in ["total current", "main current"]):
            value = _si_value(currents[0])
            formula_id = "current_sum"
            policy = "single_remaining_parallel_branch_current_equals_total"
            used = {"I_branch": _quantity_trace(currents[0])}
        elif len(currents) >= 2 and any(cue in target_text or cue in text for cue in ["total current", "main current", "entire circuit"]):
            value = sum(_si_value(quantity) for quantity in currents)
            formula_id = "current_sum"
            policy = "total_current_sum_of_parallel_branches"
            used = {f"I_branch_{index + 1}": _quantity_trace(quantity) for index, quantity in enumerate(currents)}
        else:
            return None
        spec = FORMULA_REGISTRY[formula_id]
        return SolverResult(
            True,
            _format(value, spec.target_unit),
            value,
            spec.target_unit,
            spec.formula_id,
            spec.principle_id,
            [spec.premise],
            {
                "stage": "registry_formula_solver",
                "formula_id": spec.formula_id,
                "expression": spec.expression,
                "target_dimension": spec.target_dimension,
                "inputs": used,
                "constants": {},
                "binding_audit": {"policy": policy},
                "attempted_formula_ids": [spec.formula_id],
            },
            min(0.78, route_result.confidence),
        )

    if route_result.task_type == "electric_power":
        if "total power" not in text and "power consumed" not in text:
            return None
        powers = _topology_component_quantities(front_payload, "power")
        if len(powers) < 2:
            return None
        value = sum(_si_value(quantity) for quantity in powers)
        spec = FORMULA_REGISTRY["power_sum"]
        return SolverResult(
            True,
            _format(value, spec.target_unit),
            value,
            spec.target_unit,
            spec.formula_id,
            spec.principle_id,
            [spec.premise],
            {
                "stage": "registry_formula_solver",
                "formula_id": spec.formula_id,
                "expression": spec.expression,
                "target_dimension": spec.target_dimension,
                "inputs": {f"P_load_{index + 1}": _quantity_trace(quantity) for index, quantity in enumerate(powers)},
                "constants": {},
                "binding_audit": {"policy": "total_power_sum_of_load_powers"},
                "attempted_formula_ids": [spec.formula_id],
            },
            min(0.78, route_result.confidence),
        )
    return None


def _solve_two_charge_field_angle(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if route_result.task_type != "electric_field_point":
        return None
    if not any(cue in text for cue in ["angle", "perpendicular", "right angle"]):
        return None
    fields = _topology_component_quantities(front_payload, "electric_field")
    phi = _resultant_angle_rad(front_payload)
    if len(fields) >= 2 and phi is not None:
        e1 = abs(_si_value(fields[0]))
        e2 = abs(_si_value(fields[1]))
        value_sq = e1 * e1 + e2 * e2 + 2.0 * e1 * e2 * math.cos(phi)
        if value_sq < -1e-6:
            return None
        value = math.sqrt(max(0.0, value_sq))
        spec = FORMULA_REGISTRY["electric_field_resultant_two_vectors"]
        return SolverResult(
            True,
            _format(value, spec.target_unit),
            value,
            spec.target_unit,
            spec.formula_id,
            spec.principle_id,
            [spec.premise],
            {
                "stage": "registry_formula_solver",
                "formula_id": spec.formula_id,
                "expression": spec.expression,
                "target_dimension": spec.target_dimension,
                "inputs": {
                    "E1": _quantity_trace(fields[0]),
                    "E2": _quantity_trace(fields[1]),
                    "phi": {
                        "dimension": "angle",
                        "raw_text": _resultant_angle_source(front_payload),
                        "unit": "rad",
                        "si_value": phi,
                        "symbol": "phi",
                        "binding_policy": "deterministic_angle_context",
                    },
                },
                "constants": {},
                "component_values": {"E1": e1, "E2": e2},
                "binding_audit": {"policy": "known_field_vectors_law_of_cosines"},
                "attempted_formula_ids": [spec.formula_id],
            },
            min(0.84, route_result.confidence),
        )
    charges = _topology_component_quantities(front_payload, "charge")
    lengths = _topology_component_quantities(front_payload, "length")
    if len(charges) < 2 or not lengths:
        return None
    if phi is None:
        return None
    q1, q2 = charges[0], charges[1]
    if len(lengths) == 1:
        r1 = r2 = _si_value(lengths[0])
        length_inputs = {"r": _quantity_trace(lengths[0])}
    else:
        r1 = _si_value(lengths[0])
        r2 = _si_value(lengths[1])
        length_inputs = {"r1": _quantity_trace(lengths[0]), "r2": _quantity_trace(lengths[1])}
    if r1 <= 0 or r2 <= 0:
        return None
    q1_value = _si_value(q1)
    q2_value = _si_value(q2)
    if q1_value * q2_value < 0:
        phi = math.pi - phi
    e1 = 9e9 * abs(q1_value) / (r1 * r1)
    e2 = 9e9 * abs(q2_value) / (r2 * r2)
    value_sq = e1 * e1 + e2 * e2 + 2.0 * e1 * e2 * math.cos(phi)
    if value_sq < -1e-9:
        return None
    value = math.sqrt(max(0.0, value_sq))
    spec = FORMULA_REGISTRY["electric_field_two_charge_angle"]
    inputs = {"q1": _quantity_trace(q1), "q2": _quantity_trace(q2), **length_inputs}
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": inputs,
            "constants": {"k": 9e9, "included_angle_rad": phi},
            "component_values": {"E1": e1, "E2": e2},
            "binding_audit": {"policy": "two_point_charge_field_law_of_cosines", "opposite_charge_angle_adjusted": q1_value * q2_value < 0},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.78, route_result.confidence),
    )


def _solve_electric_equilibrium_mass_angle(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "equilibrium" not in text or "mass" not in text:
        return None
    charge = _quantity_by_symbol_or_context(front_payload, "charge", {"q"}, context_cues={"carries a charge", "charge"})
    field = _quantity_by_symbol_or_context(front_payload, "electric_field", {"e"}, context_cues={"electric field", "field strength"})
    angle = _quantity_by_symbol_or_context(front_payload, "angle", {"theta", "α", "alpha"}, context_cues={"angle", "vertical"})
    if charge is None or field is None or angle is None:
        return None
    theta = _si_value(angle)
    tangent = math.tan(theta)
    if abs(tangent) <= 1e-15:
        return None
    value = abs(_si_value(charge)) * _si_value(field) / (9.8 * abs(tangent))
    spec = FORMULA_REGISTRY["electric_equilibrium_mass_angle"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"q": _quantity_trace(charge), "E": _quantity_trace(field), "theta": _quantity_trace(angle)},
            "constants": {"g": 9.8},
            "binding_audit": {"policy": "horizontal_electric_force_vertical_weight_equilibrium"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.8, route_result.confidence),
    )


def _solve_electric_field_dielectric_scaling(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "dielectric" not in text:
        return None
    field = _quantity_by_symbol_or_context(front_payload, "electric_field", {"e"}, context_cues={"electric field", "field strength"})
    epsilon_r = _dielectric_constant_value(front_payload)
    if field is None or epsilon_r is None or epsilon_r <= 0:
        return None
    value = _si_value(field) / epsilon_r
    spec = FORMULA_REGISTRY["dielectric_field_scaled"]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"E_initial": _quantity_trace(field)},
            "constants": {"epsilon_r": epsilon_r},
            "binding_audit": {"policy": "uniform_dielectric_field_scaling"},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.78, route_result.confidence),
    )


def _solve_inverse_square_midpoint_field_expression(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "midpoint" not in text or "field" not in text:
        return None
    symbolic_fields = _symbolic_field_symbols(front_payload)
    relation_text = " ".join(
        f"{relation.get('lhs') or ''} {relation.get('rhs') or ''} {relation.get('raw_text') or ''}"
        for relation in front_payload.get("symbolic_relations") or []
        if isinstance(relation, dict)
    ).lower()
    inverse_sqrt_cue = "1/sqrt" in text or "1 / sqrt" in text or "1/sqrt" in relation_text or "1 / sqrt" in relation_text
    if not inverse_sqrt_cue:
        return None
    if len(symbolic_fields) < 3 and not re.search(r"\b(?:endpoints?|field\s+line|two\s+points?)\b", text):
        return None
    midpoint_symbol, endpoint_symbols = _midpoint_field_symbols(symbolic_fields)
    spec = FORMULA_REGISTRY["point_charge_field_midpoint_inverse_expression"]
    answer = f"1/sqrt({midpoint_symbol}) = 1/2*(1/sqrt({endpoint_symbols[0]}) + 1/sqrt({endpoint_symbols[1]}))"
    return SolverResult(
        True,
        answer,
        answer,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {
                endpoint_symbols[0]: {"dimension": "electric_field", "source": "symbolic_endpoint_field_1"},
                endpoint_symbols[1]: {"dimension": "electric_field", "source": "symbolic_endpoint_field_2"},
                midpoint_symbol: {"dimension": "electric_field", "source": "symbolic_midpoint_field"},
            },
            "constants": {},
            "binding_audit": {
                "policy": "inverse_square_field_line_midpoint_linearized_by_inverse_sqrt",
                "midpoint_symbol": midpoint_symbol,
                "endpoint_symbols": endpoint_symbols,
            },
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.76, route_result.confidence),
    )


def _symbolic_field_symbols(front_payload: dict) -> list[str]:
    symbols: list[str] = []
    for quantity in front_payload.get("symbolic_quantities") or []:
        if not isinstance(quantity, dict):
            continue
        symbol = str(quantity.get("symbol") or "").strip()
        if not symbol:
            continue
        if quantity.get("dimension") == "electric_field" or symbol.lower().startswith("e"):
            if symbol not in symbols:
                symbols.append(symbol)
    return symbols


def _midpoint_field_symbols(symbols: list[str]) -> tuple[str, list[str]]:
    midpoint_symbol = next((symbol for symbol in symbols if re.search(r"(?:^|[_-])m(?:$|[_-])|mid", symbol, re.IGNORECASE)), None)
    if midpoint_symbol is None:
        midpoint_symbol = symbols[2] if len(symbols) >= 3 else "E_mid"
    endpoints = [symbol for symbol in symbols if symbol != midpoint_symbol]
    if len(endpoints) < 2:
        endpoints = ["E_endpoint_1", "E_endpoint_2"]
    return midpoint_symbol, endpoints[:2]


def _solve_electric_field_proportional_scaling(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["replaced by", "halved", "doubled", "tripled", "distance"]):
        return None
    field_symbol = _symbolic_quantity_by_dimension(front_payload, "electric_field")
    if field_symbol is None:
        return None
    charge_factor = _charge_replacement_factor(text)
    distance_factor = _distance_replacement_factor(text)
    epsilon_r = _dielectric_constant_value(front_payload) if "dielectric" in text else 1.0
    if charge_factor is None and distance_factor is None and epsilon_r == 1.0:
        return None
    scale = (charge_factor or 1.0) / ((distance_factor or 1.0) ** 2) / epsilon_r
    if scale <= 0 or not math.isfinite(scale):
        return None
    symbol = str(field_symbol.get("symbol") or "E")
    answer = _format_symbolic_multiple(scale, symbol)
    spec = FORMULA_REGISTRY["electric_field_proportional_scaling"]
    return SolverResult(
        solved=True,
        answer=answer,
        value=answer,
        unit=None,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"E_initial": dict(field_symbol)},
            "constants": {"charge_factor": charge_factor or 1.0, "distance_factor": distance_factor or 1.0, "epsilon_r": epsilon_r},
            "binding_audit": {"policy": "point_charge_field_proportional_scaling"},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.74, route_result.confidence),
    )


def _solve_parallel_sheet_field(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["sheet", "plate"]):
        return None
    sigma = _quantity_by_symbol_or_context(
        front_payload,
        "surface_charge_density",
        {"sigma", "σ"},
        context_cues={"surface charge", "charge density"},
    )
    derived_sigma = None
    if sigma is None:
        derived_sigma = _surface_density_from_charge_area(front_payload)
    if sigma is None and derived_sigma is None:
        return None
    sigma_value = abs(_si_value(sigma)) if sigma is not None else abs(float(derived_sigma["si_value"]))
    epsilon0 = _constant_value("epsilon0", front_payload)
    if epsilon0 is None or epsilon0 <= 0:
        return None
    spec = FORMULA_REGISTRY["electric_field_parallel_sheets"]
    two_surfaces = bool(re.search(r"\b(two|pair of|parallel)\b", text))
    same_sign_between = two_surfaces and "between" in text and any(cue in text for cue in ["same sign", "same charge", "same surface charge"])
    if same_sign_between:
        value = 0.0
        policy = "between_same_sign_parallel_sheets_cancel"
        expression = "E = 0"
    elif two_surfaces:
        value = sigma_value / epsilon0
        policy = "between_opposite_parallel_sheets_or_capacitor_plates"
        expression = spec.expression
    else:
        value = sigma_value / (2.0 * epsilon0)
        policy = "single_infinite_sheet_field"
        expression = "E = sigma/(2*epsilon0)"
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"sigma": _quantity_trace(sigma) if sigma is not None else dict(derived_sigma)},
            "constants": {"epsilon0": epsilon0},
            "binding_audit": {"policy": policy, "two_surfaces": two_surfaces},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.78, route_result.confidence),
    )


def _solve_charged_particle_stopping_distance(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["reduces to zero", "comes to rest", "stops", "before its velocity"]):
        return None
    field = _quantity_by_symbol_or_context(front_payload, "electric_field", {"e"}, context_cues={"electric field", "field strength"})
    velocity = _quantity_by_symbol_or_context(front_payload, "velocity", {"v", "v0"}, context_cues={"initial velocity", "velocity"})
    if field is None or velocity is None:
        return None
    if "electron" in text:
        charge = 1.602176634e-19
        mass = 9.1093837e-31
        particle = "electron"
    else:
        charge_quantity = _quantity_by_symbol_or_context(front_payload, "charge", {"q"}, context_cues={"charge"})
        mass_quantity = _quantity_by_symbol_or_context(front_payload, "mass", {"m"}, context_cues={"mass"})
        if charge_quantity is None or mass_quantity is None:
            return None
        charge = abs(_si_value(charge_quantity))
        mass = _si_value(mass_quantity)
        particle = "explicit_particle"
    if charge <= 0 or mass <= 0 or _si_value(field) <= 0:
        return None
    value = mass * (_si_value(velocity) ** 2) / (2.0 * charge * abs(_si_value(field)))
    spec = FORMULA_REGISTRY["charged_particle_stopping_distance_uniform_field"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"E": _quantity_trace(field), "v0": _quantity_trace(velocity)},
            "constants": {"particle": particle, "abs_q": charge, "m": mass},
            "binding_audit": {"policy": "kinetic_energy_stopped_by_uniform_field_work"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.8, route_result.confidence),
    )


def _surface_density_from_charge_area(front_payload: dict) -> dict | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["sheet", "plate", "surface", "flat", "area"]):
        return None
    charge = _first_quantity(front_payload, "charge")
    if charge is None:
        return None
    area = _first_quantity(front_payload, "area")
    area_value = _si_value(area) if area is not None else None
    if area_value is None:
        length_values = [
            _si_value(quantity)
            for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
            if quantity.get("dimension") == "length" and unit_info(quantity.get("unit") or "") is not None
        ]
        if len(length_values) >= 2 and any(cue in text for cue in ["rectangular area", "area is", "area of", "m x", "m ×", "m by"]):
            area_value = length_values[0] * length_values[1]
    if area_value is None or area_value <= 0:
        return None
    sigma_value = _si_value(charge) / area_value
    return {
        "dimension": "surface_charge_density",
        "raw_text": "derived from total charge and charged area",
        "unit": "C/m^2",
        "si_value": sigma_value,
        "source_charge": _quantity_trace(charge),
        "area_m2": area_value,
        "binding_policy": "surface_density_from_total_charge_over_area",
    }


def _solve_parallel_plate_breakdown_charge(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["breakdown", "maximum charge", "max charge", "without causing"]):
        return None
    radius = _length_by_context(front_payload, {"radius", "circular plate", "circular plates"})
    field = _quantity_by_symbol_or_context(front_payload, "electric_field", {"e", "emax", "e_max"}, context_cues={"breakdown", "electric field"})
    if radius is None or field is None:
        return None
    radius_value = _si_value(radius)
    if radius_value <= 0:
        return None
    epsilon0 = _constant_value("epsilon0", front_payload)
    if epsilon0 is None:
        return None
    value = epsilon0 * math.pi * radius_value * radius_value * abs(_si_value(field))
    spec = FORMULA_REGISTRY["parallel_plate_breakdown_charge"]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"r": _quantity_trace(radius), "E": _quantity_trace(field)},
            "constants": {"epsilon0": epsilon0, "pi": math.pi},
            "binding_audit": {"policy": "circular_parallel_plate_area_from_radius"},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.82, route_result.confidence),
    )


def _solve_parallel_plate_capacitor_energy_or_density(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text or not any(cue in text for cue in ["parallel-plate", "parallel plate", "plate separation", "plate area"]):
        return None
    voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v"}, context_cues={"voltage", "applied", "charged"})
    separation = _quantity_by_symbol_or_context(front_payload, "length", {"d"}, context_cues={"separation", "distance", "plate"})
    if voltage is None or separation is None or _si_value(separation) <= 0:
        return None
    epsilon0 = _constant_value("epsilon0", front_payload)
    if epsilon0 is None:
        return None
    epsilon_r = _dielectric_constant_value(front_payload) or 1.0
    field = _si_value(voltage) / _si_value(separation)
    target_text = " ".join(str(goal.get("text") or "") for goal in front_payload.get("goals") or [] if isinstance(goal, dict)).lower()
    if "density" in text or "density" in target_text:
        value = 0.5 * epsilon0 * epsilon_r * field * field
        spec = FORMULA_REGISTRY["parallel_plate_energy_density"]
        inputs = {"U": _quantity_trace(voltage), "d": _quantity_trace(separation)}
        policy = "parallel_plate_energy_density_from_uniform_field"
    else:
        area = _quantity_by_symbol_or_context(front_payload, "area", {"s", "a"}, context_cues={"area", "plate"})
        if area is None or _si_value(area) <= 0:
            return None
        value = 0.5 * epsilon0 * epsilon_r * _si_value(area) * (_si_value(voltage) ** 2) / _si_value(separation)
        spec = FORMULA_REGISTRY["parallel_plate_energy"]
        inputs = {"A": _quantity_trace(area), "d": _quantity_trace(separation), "U": _quantity_trace(voltage)}
        policy = "parallel_plate_energy_from_capacitance_geometry"
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": inputs,
            "constants": {"epsilon0": epsilon0, "epsilon_r": epsilon_r},
            "binding_audit": {"policy": policy},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.82, route_result.confidence),
    )


def _solve_connected_capacitor_source_work(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text or not any(cue in text for cue in ["still connected", "remains connected", "connected to the source", "connected to battery"]):
        return None
    if not any(cue in text for cue in ["work supplied by the source", "work by the source", "source work"]):
        return None
    area = _quantity_by_symbol_or_context(front_payload, "area", {"s", "a"}, context_cues={"area", "plate"})
    separation = _quantity_by_symbol_or_context(front_payload, "length", {"d"}, context_cues={"separation", "distance", "plate"})
    voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v"}, context_cues={"voltage", "source", "charged"})
    if area is None or separation is None or voltage is None:
        return None
    epsilon0 = _constant_value("epsilon0", front_payload)
    if epsilon0 is None or _si_value(area) <= 0 or _si_value(separation) <= 0:
        return None
    epsilon_r = _dielectric_constant_value(front_payload) or 1.0
    distance_factor = _explicit_distance_change_factor(front_payload) or extract_change_factor(text)
    if distance_factor is None or distance_factor <= 0:
        return None
    c_initial = epsilon0 * epsilon_r * _si_value(area) / _si_value(separation)
    c_final = c_initial / distance_factor
    value = _si_value(voltage) * (c_final - c_initial) * _si_value(voltage)
    spec = FORMULA_REGISTRY["capacitor_energy_source_work_constant_voltage"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"A": _quantity_trace(area), "d_initial": _quantity_trace(separation), "U": _quantity_trace(voltage)},
            "constants": {"epsilon0": epsilon0, "epsilon_r": epsilon_r, "distance_factor": distance_factor},
            "binding_audit": {
                "policy": "connected_capacitor_source_work_from_charge_change",
                "C_initial": c_initial,
                "C_final": c_final,
            },
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.78, route_result.confidence),
    )


def _solve_isolated_capacitor_energy_scaled(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text or "energy" not in text:
        return None
    if not any(cue in text for cue in ["disconnected", "isolated", "battery removed", "source removed"]):
        return None
    if not any(cue in text for cue in ["permittivity", "dielectric"]):
        return None
    energy = _quantity_by_symbol_or_context(front_payload, "energy", {"w"}, context_cues={"initial", "energy"})
    if energy is None:
        return None
    epsilon_r = _dielectric_constant_value(front_payload)
    if epsilon_r is None:
        factor = extract_change_factor(text)
        epsilon_r = factor if factor and factor > 0 else None
    if epsilon_r is None or epsilon_r <= 0:
        return None
    value = _si_value(energy) / epsilon_r
    spec = FORMULA_REGISTRY["capacitor_energy_isolated_dielectric_scaled"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"W_initial": _quantity_trace(energy)},
            "constants": {"epsilon_r_factor": epsilon_r},
            "binding_audit": {"policy": "isolated_capacitor_energy_inverse_to_capacitance_scale"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.8, route_result.confidence),
    )


def _solve_series_identical_capacitor_energy_sharing(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text or "energy" not in text or "uncharged" not in text:
        return None
    if "series" not in text:
        return None
    energies = _topology_component_quantities(front_payload, "energy")
    capacitances = _topology_component_quantities(front_payload, "capacitance")
    if not energies or len(capacitances) < 2:
        return None
    c_values = [_si_value(item) for item in capacitances[:2]]
    if min(c_values) <= 0:
        return None
    initial_energy = _si_value(energies[0])
    # For two equal capacitors, isolated charge sharing leaves half the initial
    # total energy. For unequal capacitors, Q is inferred from W=Q²/(2C1) and
    # final energy is Q²/(2*Ceq_series).
    if math.isclose(c_values[0], c_values[1], rel_tol=1e-9, abs_tol=0.0):
        value = initial_energy / 2.0
        policy = "two_identical_series_capacitors_share_initial_charge"
    else:
        q_initial = math.sqrt(2.0 * c_values[0] * initial_energy)
        c_eq = c_values[0] * c_values[1] / (c_values[0] + c_values[1])
        value = q_initial * q_initial / (2.0 * c_eq)
        policy = "two_series_capacitors_final_energy_from_shared_charge"
    spec = FORMULA_REGISTRY["capacitor_energy_shared_identical"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "topology_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"W_initial": _quantity_trace(energies[0]), "C1": _quantity_trace(capacitances[0]), "C2": _quantity_trace(capacitances[1])},
            "constants": {"n": 2.0} if policy == "two_identical_series_capacitors_share_initial_charge" else {},
            "binding_audit": {"policy": policy},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.78, route_result.confidence),
    )


def _solve_capacitor_energy_special(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text and "lc circuit" not in text:
        return None
    capacitances = _topology_component_quantities(front_payload, "capacitance")
    if not capacitances:
        return None
    capacitance = capacitances[0]
    c_value = _si_value(capacitance)
    if c_value <= 0:
        return None

    voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v"}, context_cues={"charged", "voltage", "potential difference"})
    if voltage is not None and any(cue in text for cue in ["total oscillation energy", "oscillation energy", "connected to an inductor"]):
        value = 0.5 * c_value * (_si_value(voltage) ** 2)
        spec = FORMULA_REGISTRY["capacitor_energy_voltage"]
        return SolverResult(
            solved=True,
            answer=_format(value, spec.target_unit),
            value=value,
            unit=spec.target_unit,
            formula_id=spec.formula_id,
            principle_id=spec.principle_id,
            premises=[spec.premise, "When a charged capacitor is connected to an ideal inductor, the initial capacitor energy is the total LC oscillation energy."],
            trace={
                "stage": "registry_formula_solver",
                "formula_id": spec.formula_id,
                "expression": spec.expression,
                "target_dimension": spec.target_dimension,
                "inputs": {"C": _quantity_trace(capacitance), "U": _quantity_trace(voltage)},
                "constants": {},
                "binding_audit": {"policy": "initial_capacitor_energy_becomes_lc_total_energy"},
                "attempted_formula_ids": [spec.formula_id],
            },
            confidence=min(0.82, route_result.confidence),
        )

    if any(cue in text for cue in ["reduction in energy", "energy reduction", "reduced in energy"]):
        if not any(cue in text for cue in ["same voltage", "constant voltage", "maintaining the same voltage"]):
            return None
        if len(capacitances) < 2:
            return None
        c_initial = _si_value(capacitances[0])
        c_final = _si_value(capacitances[-1])
        if c_initial <= 0 or c_final < 0:
            return None
        value = max(0.0, (c_initial - c_final) / c_initial * 100.0)
        spec = FORMULA_REGISTRY["energy_loss_percent"]
        return SolverResult(
            solved=True,
            answer=_format(value, spec.target_unit),
            value=value,
            unit=spec.target_unit,
            formula_id=spec.formula_id,
            principle_id=spec.principle_id,
            premises=[spec.premise, "For fixed capacitor voltage, stored energy is proportional to capacitance."],
            trace={
                "stage": "registry_formula_solver",
                "formula_id": spec.formula_id,
                "expression": "loss_percent = (C_initial-C_final)/C_initial*100 at fixed voltage",
                "target_dimension": spec.target_dimension,
                "inputs": {"C_initial": _quantity_trace(capacitances[0]), "C_final": _quantity_trace(capacitances[-1])},
                "constants": {},
                "binding_audit": {"policy": "fixed_voltage_capacitor_energy_scales_with_capacitance"},
                "attempted_formula_ids": [spec.formula_id],
            },
            confidence=min(0.8, route_result.confidence),
        )

    waveform = _parse_time_dependent_voltage(front_payload)
    if waveform is None:
        return None
    if waveform["amplitude"] <= 0:
        return None
    target_text = " ".join(str(goal.get("text") or "") for goal in front_payload.get("goals") or [] if isinstance(goal, dict)).lower()
    maximum = "maximum" in text or "maximum" in target_text or "max" in target_text
    time_quantity = _quantity_by_symbol_or_context(front_payload, "time", {"t"}, context_cues={"at t", "time"})
    if maximum:
        voltage_value = waveform["amplitude"]
        policy = "maximum_energy_from_voltage_amplitude"
    elif time_quantity is not None and waveform.get("omega") is not None:
        phase = float(waveform["omega"]) * _si_value(time_quantity)
        voltage_value = waveform["amplitude"] * (math.cos(phase) if waveform["function"] == "cos" else math.sin(phase))
        policy = "instantaneous_energy_from_time_dependent_voltage"
    else:
        return None
    value = 0.5 * c_value * voltage_value * voltage_value
    spec = FORMULA_REGISTRY["capacitor_energy_voltage"]
    inputs = {"C": _quantity_trace(capacitance), "U": {"source": "time_dependent_voltage", **waveform}}
    if time_quantity is not None:
        inputs["t"] = _quantity_trace(time_quantity)
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": inputs,
            "constants": {},
            "binding_audit": {"policy": policy, "voltage_used_v": voltage_value},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.82, route_result.confidence),
    )


def _solve_lc_or_inductor_energy_special(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["inductor", "lc circuit", "oscillation", "magnetic field energy"]):
        return None

    energies = _topology_component_quantities(front_payload, "energy")
    total_energy = _quantity_by_context_strict(front_payload, "energy", {"total", "oscillation", "w0", "w₀"})
    electric_energy = _quantity_by_context_strict(front_payload, "energy", {"electric", "capacitor", "w_e", "we", "w_c", "wc"})
    magnetic_energy = _quantity_by_context_strict(front_payload, "energy", {"magnetic", "inductor", "w_l", "wl", "w_m", "wm"})
    if magnetic_energy is None and len(energies) == 1 and any(cue in text for cue in ["magnetic field energy", "magnetic energy"]):
        magnetic_energy = energies[0]

    if total_energy is not None and electric_energy is not None and any(cue in text for cue in ["magnetic field energy", "magnetic energy", "inductor energy"]):
        value = _si_value(total_energy) - _si_value(electric_energy)
        if value < -1e-12:
            return None
        spec = FORMULA_REGISTRY["lc_energy_complement"]
        return _energy_result(
            value=max(0.0, value),
            spec=spec,
            route_result=route_result,
            inputs={"W_total": _quantity_trace(total_energy), "W_electric": _quantity_trace(electric_energy)},
            policy="lc_magnetic_energy_is_total_minus_electric",
            extra_premises=["In an ideal LC circuit, electric and magnetic energies sum to the conserved total energy."],
        )

    if total_energy is not None and any(cue in text for cue in ["voltage across the capacitor", "capacitor voltage", "charged to"]):
        capacitance = _quantity_by_symbol_or_context(front_payload, "capacitance", {"c"}, context_cues={"capacitance", "capacitor"})
        voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v"}, context_cues={"capacitor", "voltage", "charged"})
        if capacitance is not None and voltage is not None and any(cue in text for cue in ["magnetic", "inductor"]):
            electric_value = 0.5 * _si_value(capacitance) * (_si_value(voltage) ** 2)
            value = _si_value(total_energy) - electric_value
            if value < -1e-12:
                return None
            spec = FORMULA_REGISTRY["lc_energy_complement"]
            return _energy_result(
                value=max(0.0, value),
                spec=spec,
                route_result=route_result,
                inputs={"W_total": _quantity_trace(total_energy), "C": _quantity_trace(capacitance), "U": _quantity_trace(voltage)},
                policy="lc_magnetic_energy_is_total_minus_capacitor_energy",
                extra_premises=["Capacitor electric energy is W_C = 1/2 C U^2."],
            )

    if magnetic_energy is not None and "current" in text and has_change_factor_cue(text):
        factor = _change_factor_near_current(text)
        if factor is not None and factor >= 0:
            value = _si_value(magnetic_energy) * factor * factor
            spec = FORMULA_REGISTRY["inductor_energy_current_scaled"]
            return _energy_result(
                value=value,
                spec=spec,
                route_result=route_result,
                inputs={"W_initial": _quantity_trace(magnetic_energy)},
                policy="inductor_energy_scales_with_current_squared",
                constants={"current_factor": factor},
                extra_premises=["For fixed inductance, magnetic energy scales as I^2."],
            )

    capacitance = _quantity_by_symbol_or_context(front_payload, "capacitance", {"c"}, context_cues={"capacitance", "capacitor"})
    voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v"}, context_cues={"charged", "voltage"})
    if capacitance is not None and voltage is not None and any(cue in text for cue in ["total energy", "oscillation energy", "connected to an inductor"]):
        value = 0.5 * _si_value(capacitance) * (_si_value(voltage) ** 2)
        spec = FORMULA_REGISTRY["capacitor_energy_voltage"]
        return _energy_result(
            value=value,
            spec=spec,
            route_result=route_result,
            inputs={"C": _quantity_trace(capacitance), "U": _quantity_trace(voltage)},
            policy="initial_capacitor_energy_becomes_lc_total_energy",
            extra_premises=["When a charged capacitor is connected to an ideal inductor, its initial electric energy becomes the total LC oscillation energy."],
        )

    waveform = _parse_time_dependent_current(front_payload)
    inductance = _quantity_by_symbol_or_context(front_payload, "inductance", {"l"}, context_cues={"inductance", "inductor"})
    if waveform is not None and inductance is not None:
        maximum = any(cue in text for cue in ["maximum", "max", "peak"])
        time_quantity = _quantity_by_symbol_or_context(front_payload, "time", {"t"}, context_cues={"at t", "time"})
        if maximum:
            current_value = waveform["amplitude"]
            policy = "maximum_inductor_energy_from_current_amplitude"
        elif time_quantity is not None and waveform.get("omega") is not None:
            phase = float(waveform["omega"]) * _si_value(time_quantity)
            current_value = waveform["amplitude"] * (math.cos(phase) if waveform["function"] == "cos" else math.sin(phase))
            policy = "instantaneous_inductor_energy_from_time_dependent_current"
        else:
            return None
        value = 0.5 * _si_value(inductance) * current_value * current_value
        spec = FORMULA_REGISTRY["inductor_energy"]
        return _energy_result(
            value=value,
            spec=spec,
            route_result=route_result,
            inputs={"L": _quantity_trace(inductance), "I": {"source": "time_dependent_current", **waveform}},
            policy=policy,
        )
    return None


def _solve_lc_energy_complement_from_time_expression(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "")
    lowered = text.lower()
    if "lc" not in lowered or "energy" not in lowered:
        return None
    if "magnetic" not in lowered:
        return None
    compact = re.sub(r"\s+", "", text.lower()).replace("²", "2")
    match = re.search(
        r"w[_-]?c=(?P<amp>\d+(?:\.\d+)?)\s*(?P<func>cos|sin)(?:\^?2|2)\(?(?P<omega>\d+(?:\.\d+)?)t\)?",
        compact,
    )
    if not match:
        return None
    time_match = re.search(r"t\s*=\s*(?:π|pi)\s*/\s*(?P<den>\d+(?:\.\d+)?)\s*s?", text, flags=re.IGNORECASE)
    if not time_match:
        return None
    amplitude = float(match.group("amp"))
    omega = float(match.group("omega"))
    t_value = math.pi / float(time_match.group("den"))
    phase = omega * t_value
    trig = math.cos(phase) if match.group("func") == "cos" else math.sin(phase)
    electric = amplitude * trig * trig
    value = max(0.0, amplitude - electric)
    spec = FORMULA_REGISTRY["lc_energy_complement"]
    return _energy_result(
        value=value,
        spec=spec,
        route_result=route_result,
        inputs={"W_total": {"si_value": amplitude, "source": "energy_expression_amplitude"}, "W_C(t)": {"si_value": electric, "source": match.group(0)}},
        policy="lc_magnetic_energy_is_total_minus_time_dependent_electric_energy",
        constants={"omega": omega, "t": t_value, "phase": phase},
        extra_premises=["In an ideal LC circuit, W_C + W_L is constant."],
    )


def _solve_lc_energy_extreme_statement(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "")
    lowered = text.lower()
    if "lc" not in lowered or "energy" not in lowered:
        return None
    compact = re.sub(r"\s+", "", lowered).replace("_", "")
    target_text = " ".join(front_payload.get("target_hints") or []).lower().replace("_", "")
    asks_wc = "wc" in target_text or "electric" in target_text
    asks_wl = "wl" in target_text or "magnetic" in target_text
    if not asks_wc and not asks_wl:
        return None
    if asks_wc and re.search(r"\bwl\s*=\s*0(?!\.\d)(?!\d)", compact):
        answer = "W_C = W_total (maximum)"
        known_symbol = "W_L"
    elif asks_wl and re.search(r"\bwc\s*=\s*0(?!\.\d)(?!\d)", compact):
        answer = "W_L = W_total (maximum)"
        known_symbol = "W_C"
    else:
        return None
    spec = FORMULA_REGISTRY["lc_energy_complement"]
    return SolverResult(
        True,
        answer,
        answer,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise, "In an ideal LC oscillator, if one energy store is zero, the other equals the conserved total energy."],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {known_symbol: {"si_value": 0.0, "dimension": "energy", "source": f"{known_symbol}=0"}},
            "constants": {},
            "binding_audit": {"policy": "lc_energy_extreme_from_zero_complement"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.78, route_result.confidence),
    )


def _solve_lc_energy_symbolic_complement_expression(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "")
    lowered = text.lower().replace("_", "")
    if "lc" not in lowered or "energy" not in lowered:
        return None
    target_text = " ".join(front_payload.get("target_hints") or []).lower().replace("_", "")
    asks_wc = "wc" in target_text or "electric" in target_text
    asks_wl = "wl" in target_text or "magnetic" in target_text
    if not asks_wc and not asks_wl:
        return None
    compact = re.sub(r"\s+", "", lowered).replace("²", "2")
    known = None
    if asks_wc and re.search(r"wl=w[₀0o](?:cos|sin)(?:\^?2|2)", compact):
        known = "W_L"
    elif asks_wl and re.search(r"wc=w[₀0o](?:cos|sin)(?:\^?2|2)", compact):
        known = "W_C"
    if known is None:
        return None
    known_match = re.search(r"w[lc]=w[₀0o](?P<trig>cos|sin)(?:\^?2|2)\(?ω?t\)?", compact)
    trig = known_match.group("trig") if known_match else "cos"
    complement = "sin" if trig == "cos" else "cos"
    target_symbol = "W_C" if asks_wc else "W_L"
    answer = f"{target_symbol} = W0{complement}²(ωt)"
    spec = FORMULA_REGISTRY["lc_energy_complement"]
    return SolverResult(
        True,
        answer,
        answer,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise, "In an ideal LC circuit, the complementary energy has the opposite sin²/cos² factor."],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {known: {"dimension": "energy", "source": "symbolic_time_expression"}},
            "constants": {},
            "binding_audit": {"policy": "lc_symbolic_energy_complement", "known_trig": trig, "target_trig": complement},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.78, route_result.confidence),
    )


def _energy_result(
    *,
    value: float,
    spec: FormulaSpec,
    route_result,
    inputs: dict,
    policy: str,
    constants: dict | None = None,
    extra_premises: list[str] | None = None,
) -> SolverResult:
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise, *(extra_premises or [])],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": inputs,
            "constants": constants or {},
            "binding_audit": {"policy": policy},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.82, route_result.confidence),
    )


def _solve_solenoid_magnetic_energy_special(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "solenoid" not in text or "energy" not in text:
        return None
    mu0 = 4.0 * math.pi * 1e-7
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    magnetic_field = _quantity_by_symbol_or_context(front_payload, "magnetic_field", {"b"}, context_cues={"magnetic field", "inside"})
    turn_density = _quantity_by_symbol_or_context(front_payload, "turn_density", {"n"}, context_cues={"turn density", "turns per meter"})
    if turn_density is None:
        n_value = _parse_turn_density_text(text)
        if n_value is not None:
            turn_density = _synthetic_quantity("n", n_value, "turns/m", "turn_density", "parsed turn density phrase")
    current = _quantity_by_symbol_or_context(front_payload, "current", {"i"}, context_cues={"current"})
    area = _quantity_by_symbol_or_context(front_payload, "area", {"a", "s"}, context_cues={"area", "cross-sectional"})
    length = _quantity_by_symbol_or_context(front_payload, "length", {"l"}, context_cues={"length", "long solenoid"})
    turns = _largest_count_quantity(front_payload)

    if "density" in target_text or "energy density" in text:
        if magnetic_field is not None:
            value = (_si_value(magnetic_field) ** 2) / (2.0 * mu0)
            spec = FORMULA_REGISTRY["magnetic_energy_density_field"]
            inputs = {"B": _quantity_trace(magnetic_field)}
            policy = "magnetic_energy_density_from_field"
        elif turn_density is not None and current is not None:
            value = 0.5 * mu0 * (_si_value(turn_density) ** 2) * (_si_value(current) ** 2)
            spec = FORMULA_REGISTRY["solenoid_magnetic_energy_density"]
            inputs = {"n": _quantity_trace(turn_density), "I": _quantity_trace(current)}
            policy = "solenoid_energy_density_from_turn_density_current"
        else:
            return None
        return _energy_density_result(value=value, spec=spec, route_result=route_result, inputs=inputs, policy=policy)

    if length is not None and area is not None and turns is not None and current is not None:
        l_value = _si_value(length)
        a_value = _si_value(area)
        n_turns = _si_value(turns)
        i_value = _si_value(current)
        if l_value <= 0 or a_value <= 0 or n_turns <= 0:
            return None
        value = 0.5 * (mu0 * n_turns * n_turns * a_value / l_value) * i_value * i_value
        spec = FORMULA_REGISTRY["solenoid_magnetic_energy"]
        return _energy_result(
            value=value,
            spec=spec,
            route_result=route_result,
            inputs={"N": _quantity_trace(turns), "A": _quantity_trace(area), "l": _quantity_trace(length), "I": _quantity_trace(current)},
            policy="solenoid_energy_from_inductance_and_current",
        )
    return None


def _solve_energy_efficiency_from_loss(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "efficiency" not in text:
        return None
    energies = _topology_component_quantities(front_payload, "energy")
    if len(energies) < 2:
        return None
    loss = next(
        (
            quantity
            for quantity in energies
            if any(cue in f"{quantity.get('raw_text') or ''} {quantity.get('context') or ''}".lower() for cue in ["dissipated", "loss", "lost", "wasted"])
        ),
        energies[0],
    )
    useful_candidates = [quantity for quantity in energies if quantity is not loss]
    if not useful_candidates:
        return None
    useful = max(useful_candidates, key=_si_value)
    loss_value = _si_value(loss)
    useful_value = _si_value(useful)
    total = useful_value + loss_value
    if total <= 0 or useful_value < 0 or loss_value < 0:
        return None
    value = useful_value / total * 100.0
    spec = FORMULA_REGISTRY["energy_efficiency_from_loss"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"W_useful": _quantity_trace(useful), "W_loss": _quantity_trace(loss)},
            "constants": {},
            "binding_audit": {"policy": "efficiency_from_useful_energy_and_dissipated_loss"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.78, route_result.confidence),
    )


def _energy_density_result(*, value: float, spec: FormulaSpec, route_result, inputs: dict, policy: str) -> SolverResult:
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": inputs,
            "constants": {"mu0": 4.0 * math.pi * 1e-7},
            "binding_audit": {"policy": policy},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.82, route_result.confidence),
    )


def _solve_solenoid_magnetic_field_or_flux_special(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["solenoid", "coil", "magnetic flux", "flux linkage"]):
        return None
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    if route_result.task_type == "solenoid_magnetic_field" or any(cue in target_text for cue in ["magnetic field", "flux density"]):
        turn_density = _quantity_by_symbol_or_context(front_payload, "turn_density", {"n"}, context_cues={"turn density", "turns per meter"})
        if turn_density is None:
            n_value = _parse_turn_density_text(text)
            if n_value is not None:
                turn_density = _synthetic_quantity("n", n_value, "turns/m", "turn_density", "parsed turn density phrase")
        current = _quantity_by_symbol_or_context(front_payload, "current", {"i"}, context_cues={"current"})
        if turn_density is None or current is None:
            return None
        value = 4.0 * math.pi * 1e-7 * _si_value(turn_density) * _si_value(current)
        spec = FORMULA_REGISTRY["solenoid_magnetic_field"]
        return SolverResult(
            True,
            _format(value, spec.target_unit),
            value,
            spec.target_unit,
            spec.formula_id,
            spec.principle_id,
            [spec.premise],
            {
                "stage": "registry_formula_solver",
                "formula_id": spec.formula_id,
                "expression": spec.expression,
                "target_dimension": spec.target_dimension,
                "inputs": {"n": _quantity_trace(turn_density), "I": _quantity_trace(current)},
                "constants": {"mu0": 4.0 * math.pi * 1e-7},
                "binding_audit": {"policy": "solenoid_field_from_turn_density_current"},
                "attempted_formula_ids": [spec.formula_id],
            },
            min(0.84, route_result.confidence),
        )
    if route_result.task_type == "magnetic_flux" or "flux linkage" in text:
        flux = _quantity_by_symbol_or_context(front_payload, "magnetic_flux", {"phi", "flux"}, context_cues={"flux", "per turn"})
        turns = _largest_count_quantity(front_payload)
        if flux is not None and turns is not None:
            value = _si_value(turns) * _si_value(flux)
            spec = FORMULA_REGISTRY["magnetic_flux_linkage"]
            return SolverResult(
                True,
                _format(value, spec.target_unit),
                value,
                spec.target_unit,
                spec.formula_id,
                spec.principle_id,
                [spec.premise],
                {
                    "stage": "registry_formula_solver",
                    "formula_id": spec.formula_id,
                    "expression": spec.expression,
                    "target_dimension": spec.target_dimension,
                    "inputs": {"N": _quantity_trace(turns), "Phi": _quantity_trace(flux)},
                    "constants": {},
                    "binding_audit": {"policy": "flux_linkage_from_flux_per_turn"},
                    "attempted_formula_ids": [spec.formula_id],
                },
                min(0.84, route_result.confidence),
            )
        area = _quantity_by_symbol_or_context(front_payload, "area", {"a", "s"}, context_cues={"area", "cross-sectional"})
        current = _quantity_by_symbol_or_context(front_payload, "current", {"i"}, context_cues={"current"})
        turn_density = _quantity_by_symbol_or_context(front_payload, "turn_density", {"n"}, context_cues={"turn density", "turns per meter"})
        if turn_density is None:
            n_value = _parse_turn_density_text(text)
            if n_value is not None:
                turn_density = _synthetic_quantity("n", n_value, "turns/m", "turn_density", "parsed turn density phrase")
        if turn_density is None and turns is not None:
            length = _quantity_by_symbol_or_context(front_payload, "length", {"l"}, context_cues={"length", "long", "solenoid"})
            if length is not None and _si_value(length) > 0:
                turn_density = _synthetic_quantity("n", _si_value(turns) / _si_value(length), "turns/m", "turn_density", "turn count divided by solenoid length")
        if area is None or current is None or turn_density is None:
            return None
        value = 4.0 * math.pi * 1e-7 * _si_value(turn_density) * _si_value(current) * _si_value(area)
        spec = FORMULA_REGISTRY["solenoid_flux_one_turn"]
        return SolverResult(
            True,
            _format(value, spec.target_unit),
            value,
            spec.target_unit,
            spec.formula_id,
            spec.principle_id,
            [spec.premise],
            {
                "stage": "registry_formula_solver",
                "formula_id": spec.formula_id,
                "expression": spec.expression,
                "target_dimension": spec.target_dimension,
                "inputs": {"n": _quantity_trace(turn_density), "I": _quantity_trace(current), "A": _quantity_trace(area)},
                "constants": {"mu0": 4.0 * math.pi * 1e-7},
                "binding_audit": {"policy": "solenoid_flux_from_turn_density_current_area"},
                "attempted_formula_ids": [spec.formula_id],
            },
            min(0.84, route_result.confidence),
        )
    return None


def _solve_capacitor_series_voltage(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "series" not in text or "capacitor" not in text:
        return None
    capacitances = _topology_component_quantities(front_payload, "capacitance")
    voltages = _topology_component_quantities(front_payload, "voltage")
    if len(capacitances) < 2 or len(voltages) != 1:
        return None
    c1 = _si_value(capacitances[0])
    c2 = _si_value(capacitances[1])
    total_voltage = _si_value(voltages[0])
    if c1 <= 0 or c2 <= 0:
        return None
    target_index = 1
    target_text = " ".join(str(goal.get("text") or "") for goal in front_payload.get("goals") or [] if isinstance(goal, dict)).lower()
    if re.search(r"\bc1\b|capacitor\s+c1|first capacitor", target_text):
        target_index = 0
    elif re.search(r"\bc2\b|capacitor\s+c2|second capacitor", target_text):
        target_index = 1
    equivalent = c1 * c2 / (c1 + c2)
    charge = equivalent * total_voltage
    value = charge / (c1 if target_index == 0 else c2)
    spec = FORMULA_REGISTRY["capacitor_series_voltage"]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "topology_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {
                "C1": _quantity_trace(capacitances[0]),
                "C2": _quantity_trace(capacitances[1]),
                "U_total": _quantity_trace(voltages[0]),
            },
            "constants": {},
            "binding_audit": {
                "policy": "series_capacitors_same_charge_voltage_division",
                "target_capacitor_index": target_index + 1,
                "equivalent_capacitance_f": equivalent,
                "shared_charge_c": charge,
            },
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.82, route_result.confidence),
    )


def _solve_capacitor_series_unknown_from_final_charge(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text or "series" not in text:
        return None
    if not re.search(r"\bfind\s+c['′]?\b|\bcalculate\s+c['′]?\b|\bunknown\s+capacitor\b", text):
        return None
    capacitances = _topology_component_quantities(front_payload, "capacitance")
    charges = _topology_component_quantities(front_payload, "charge")
    voltages = _topology_component_quantities(front_payload, "voltage")
    if not capacitances or not charges or not voltages:
        return None
    known_c = _si_value(capacitances[0])
    shared_q = abs(_si_value(charges[-1]))
    total_voltage_candidates = [
        quantity
        for quantity in voltages
        if any(cue in f"{quantity.get('raw_text') or ''} {quantity.get('context') or ''}".lower() for cue in ["total", "entire", "circuit"])
    ]
    total_voltage = _si_value(total_voltage_candidates[-1] if total_voltage_candidates else voltages[-1])
    if known_c <= 0 or shared_q <= 0 or total_voltage <= 0:
        return None
    denominator = total_voltage - shared_q / known_c
    if denominator <= 0:
        return None
    value = shared_q / denominator
    spec = FORMULA_REGISTRY["capacitor_series_unknown_from_final_charge"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "topology_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {
                "C_known": _quantity_trace(capacitances[0]),
                "Q_final": _quantity_trace(charges[-1]),
                "U_total": _quantity_trace(total_voltage_candidates[-1] if total_voltage_candidates else voltages[-1]),
            },
            "constants": {},
            "binding_audit": {"policy": "series_capacitors_shared_final_charge_unknown_capacitance"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.8, route_result.confidence),
    )


def _solve_capacitor_charge_sharing_voltage(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text:
        return None
    if not any(cue in text for cue in ["connected together", "plates are connected", "terminals", "joined", "combination"]):
        return None
    capacitances = _topology_component_quantities(front_payload, "capacitance")
    voltages = _topology_component_quantities(front_payload, "voltage")
    if len(capacitances) < 2 or len(voltages) < 2:
        return None
    c1 = _si_value(capacitances[0])
    c2 = _si_value(capacitances[1])
    u1 = _si_value(voltages[0])
    u2 = _si_value(voltages[1])
    if c1 <= 0 or c2 <= 0:
        return None
    like = any(cue in text for cue in ["like", "same polarity", "same-poled", "same-signed", "positive to positive", "negative to negative"])
    unlike = any(cue in text for cue in ["opposite polarity", "oppositely", "positive to negative", "unlike"])
    if not (like or unlike):
        return None
    numerator = c1 * u1 + c2 * u2 if like else c1 * u1 - c2 * u2
    value = numerator / (c1 + c2)
    if unlike:
        value = abs(value)
    spec = FORMULA_REGISTRY["capacitor_charge_sharing_voltage"]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "topology_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {
                "C1": _quantity_trace(capacitances[0]),
                "C2": _quantity_trace(capacitances[1]),
                "U1": _quantity_trace(voltages[0]),
                "U2": _quantity_trace(voltages[1]),
            },
            "constants": {},
            "binding_audit": {
                "policy": "two_capacitor_charge_conservation_after_plate_connection",
                "polarity": "like" if like else "unlike",
                "initial_total_charge_c": numerator,
            },
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.82, route_result.confidence),
    )


def _solve_parallel_capacitor_voltage_from_branch_charge(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text or "parallel" not in text:
        return None
    if not any(cue in text for cue in ["charge", "charged", "has a charge", "carries a charge"]):
        return None
    capacitances = _topology_component_quantities(front_payload, "capacitance")
    charges = _topology_component_quantities(front_payload, "charge")
    if not capacitances or not charges:
        return None
    candidates: list[dict] = []
    for charge in charges:
        q = abs(_si_value(charge))
        if q <= 0:
            continue
        for capacitance in capacitances:
            c = _si_value(capacitance)
            if c <= 0:
                continue
            candidates.append({"value": q / c, "charge": charge, "capacitance": capacitance})
    if not candidates:
        return None
    selected = _select_unique_voltage_candidate(candidates, _voltage_bounds_from_text(front_payload))
    if selected is None:
        return None
    value = float(selected["value"])
    spec = FORMULA_REGISTRY["parallel_capacitor_voltage_from_branch_charge"]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "topology_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {
                "Q_branch": _quantity_trace(selected["charge"]),
                "C_branch": _quantity_trace(selected["capacitance"]),
            },
            "constants": {},
            "binding_audit": {
                "policy": "parallel_branch_voltage_from_branch_charge",
                "candidate_values_v": [round(float(item["value"]), 12) for item in candidates],
                "selection": selected.get("selection_reason"),
            },
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.78, route_result.confidence),
    )


def _solve_lc_capacitor_voltage_from_energy_partition(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "lc" not in text and "oscillation energy" not in text and "electromagnetic energy" not in text:
        return None
    if "voltage" not in text and "potential difference" not in text:
        return None
    capacitance = _first_quantity(front_payload, "capacitance")
    if capacitance is None:
        return None
    c_value = _si_value(capacitance)
    if c_value <= 0:
        return None
    energies = [
        quantity
        for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "energy" and unit_info(quantity.get("unit") or "") is not None
    ]
    if not energies:
        return None
    total_energy = None
    electric_energy = None
    magnetic_energy = None
    for quantity in energies:
        span = quantity.get("span") or (0, 0)
        try:
            left_context = text[max(0, int(span[0]) - 56) : int(span[0])]
        except (TypeError, ValueError):
            left_context = str(quantity.get("context") or "").lower()
        context = f"{left_context} {quantity.get('context') or ''} {quantity.get('raw_text') or ''}".lower()
        value = _si_value(quantity)
        if re.search(r"\btotal\s+(?:oscillation\s+|electromagnetic\s+)?energy\b", left_context):
            total_energy = {"quantity": quantity, "value": value}
        elif "magnetic" in left_context:
            magnetic_energy = {"quantity": quantity, "value": value}
        elif "electric" in left_context or "capacitor" in left_context:
            electric_energy = {"quantity": quantity, "value": value}
        elif "total" in context or "oscillation" in context or "electromagnetic" in context:
            total_energy = {"quantity": quantity, "value": value}
    if electric_energy is not None:
        selected_energy = electric_energy
        policy = "lc_voltage_from_explicit_electric_field_energy"
    elif total_energy is not None and magnetic_energy is not None:
        selected_energy = {
            "quantity": total_energy["quantity"],
            "value": total_energy["value"] - magnetic_energy["value"],
            "complement": magnetic_energy["quantity"],
        }
        policy = "lc_voltage_from_total_minus_magnetic_energy"
    elif total_energy is not None and any(cue in text for cue in ["maximum voltage", "max voltage", "amplitude"]):
        selected_energy = total_energy
        policy = "lc_max_voltage_from_total_energy"
    else:
        return None
    w_e = float(selected_energy["value"])
    if w_e < 0:
        return None
    value = math.sqrt(2.0 * w_e / c_value)
    spec = FORMULA_REGISTRY["capacitor_voltage_energy"]
    inputs = {"C": _quantity_trace(capacitance), "W_electric": {"si_value": w_e, "source": policy}}
    if selected_energy.get("quantity"):
        inputs["energy_fact"] = _quantity_trace(selected_energy["quantity"])
    if selected_energy.get("complement"):
        inputs["complement_energy_fact"] = _quantity_trace(selected_energy["complement"])
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise, "In an ideal LC circuit, total energy is split between electric and magnetic field energy."],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": inputs,
            "constants": {},
            "binding_audit": {"policy": policy},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.82, route_result.confidence),
    )


def _solve_connected_capacitor_voltage_constant(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text:
        return None
    if not any(cue in text for cue in ["still connected", "remains connected", "connected to the source", "connected to a source", "connected to battery", "connected to the battery"]):
        return None
    voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v", "u_source", "v_source"}, context_cues={"source", "battery", "charged to a voltage"})
    if voltage is None:
        return None
    value = _si_value(voltage)
    spec = FORMULA_REGISTRY["capacitor_connected_voltage_constant"]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"U_source": _quantity_trace(voltage)},
            "constants": {},
            "binding_audit": {"policy": "ideal_source_connected_capacitor_voltage_clamp"},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.84, route_result.confidence),
    )


def _solve_isolated_capacitor_dielectric_voltage(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text or "dielectric" not in text:
        return None
    isolated = any(cue in text for cue in ["disconnected", "isolated", "battery removed", "source removed", "after removing"])
    if not isolated:
        return None
    voltage = _quantity_by_symbol_or_context(
        front_payload,
        "voltage",
        {"u", "v"},
        context_cues={"initial", "charged", "potential difference", "voltage"},
    )
    epsilon_r = _dielectric_constant_value(front_payload)
    if voltage is None or epsilon_r is None or epsilon_r <= 0:
        return None
    value = _si_value(voltage) / epsilon_r
    spec = FORMULA_REGISTRY["capacitor_voltage_isolated_dielectric"]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"U_initial": _quantity_trace(voltage)},
            "constants": {"epsilon_r": epsilon_r},
            "binding_audit": {"policy": "isolated_capacitor_charge_conserved_dielectric_scaling"},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.82, route_result.confidence),
    )


def _solve_isolated_capacitor_distance_scaled_voltage(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text:
        return None
    if not any(cue in text for cue in ["disconnected", "isolated", "battery removed", "source removed", "after removing"]):
        return None
    if not any(cue in text for cue in ["distance", "separation", "plate spacing", "moved apart", "brought closer"]):
        return None
    distance_factor = extract_change_factor(text)
    if distance_factor is None or distance_factor <= 0:
        return None
    voltage = _quantity_by_symbol_or_context(
        front_payload,
        "voltage",
        {"u", "v"},
        context_cues={"initial", "charged", "potential difference", "voltage"},
    )
    if voltage is None:
        return None
    value = _si_value(voltage) * distance_factor
    spec = FORMULA_REGISTRY["capacitor_voltage_isolated_distance_scaled"]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"U_initial": _quantity_trace(voltage)},
            "constants": {"distance_factor": distance_factor},
            "binding_audit": {"policy": "isolated_capacitor_voltage_scales_with_plate_separation"},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.8, route_result.confidence),
    )


def _solve_capacitor_geometry_scaled_capacitance(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "capacitor" not in text:
        return None
    if not any(cue in text for cue in ["new capacitance", "final capacitance", "capacitance becomes", "what will be the capacitance"]):
        return None
    capacitance = _quantity_by_symbol_or_context(front_payload, "capacitance", {"c"}, context_cues={"capacitance", "capacitor"})
    if capacitance is None:
        return None
    scale = 1.0
    policies: list[str] = []
    if any(cue in text for cue in ["distance", "separation", "plate spacing", "between the plates"]):
        distance_factor = _explicit_distance_change_factor(front_payload) or extract_change_factor(text)
        if distance_factor is not None and distance_factor > 0:
            scale *= 1.0 / distance_factor
            policies.append("capacitance_inverse_to_plate_separation")
    if any(cue in text for cue in ["plate area", "area of the plates", "plates are split", "split in half", "cut in half"]):
        area_factor = _explicit_area_change_factor(front_payload) or extract_change_factor(text)
        if area_factor is not None and area_factor > 0:
            scale *= area_factor
            policies.append("capacitance_proportional_to_plate_area")
    if "dielectric" in text:
        epsilon_r = _dielectric_constant_value(front_payload)
        if epsilon_r is not None and epsilon_r > 0:
            scale *= epsilon_r
            policies.append("capacitance_proportional_to_dielectric_constant")
    if not policies or scale <= 0:
        return None
    value = _si_value(capacitance) * scale
    spec = FORMULA_REGISTRY["capacitor_geometry_scaled_capacitance"]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"C_initial": _quantity_trace(capacitance)},
            "constants": {"scale_factor": scale},
            "binding_audit": {"policy": "+".join(policies), "scale_factor": scale},
            "attempted_formula_ids": [spec.formula_id],
        },
        confidence=min(0.8, route_result.confidence),
    )


def _solve_rlc_resonance_current_ratio_transform(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "resonan" not in text or "current" not in text:
        return None
    factor = _frequency_change_factor(text) or _frequency_ratio_from_quantities(front_payload)
    if factor is None or factor <= 0 or math.isclose(factor, 1.0):
        return None
    resistance = _quantity_by_symbol_or_context(front_payload, "resistance", {"r"}, reject_symbols={"xl", "x_l", "xc", "x_c", "z"})
    currents = _topology_component_quantities(front_payload, "current")
    if resistance is None or len(currents) < 2:
        return None
    r_value = _si_value(resistance)
    resonance_current, changed_current = _select_resonance_and_changed_currents(currents)
    i_resonance = _si_value(resonance_current)
    i_changed = _si_value(changed_current)
    if r_value <= 0 or i_resonance <= 0 or i_changed <= 0:
        return None
    source_voltage = i_resonance * r_value
    changed_impedance = source_voltage / i_changed
    delta_reactance = math.sqrt(max(0.0, changed_impedance * changed_impedance - r_value * r_value))
    denominator = abs(factor - 1.0 / factor)
    if denominator <= 0:
        return None
    xl_initial = delta_reactance / denominator
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    target_is_changed = bool(re.search(r"\bat\s+(?:the\s+)?(?:new|changed|doubled|tripled|quadrupled|increased|final|\d+\s*hz)\b", target_text + " " + text))
    value = xl_initial * factor if target_is_changed and not re.search(r"\binitial|resonant frequency|at resonance|original", target_text) else xl_initial
    spec = FORMULA_REGISTRY["rlc_resonance_reactance_from_current_ratio"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"R": _quantity_trace(resistance), "I_resonance": _quantity_trace(resonance_current), "I_changed": _quantity_trace(changed_current)},
            "constants": {"frequency_factor": factor},
            "transformed_reactances": {
                "source_voltage": source_voltage,
                "changed_impedance": changed_impedance,
                "XL_initial": xl_initial,
                "XL_target": value,
            },
            "binding_audit": {"policy": "series_rlc_reactance_from_resonance_current_ratio"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.82, route_result.confidence),
    )


def _select_resonance_and_changed_currents(currents: list[dict]) -> tuple[dict, dict]:
    """Bind resonance current from semantics, falling back to the larger current.

    In a series RLC circuit under fixed source voltage, resonance minimizes
    impedance, so the resonant current is the maximum of comparable measured
    currents when explicit context is missing.
    """

    resonance_candidates = [
        quantity
        for quantity in currents
        if re.search(r"\bresonan", f"{quantity.get('raw_text') or ''} {quantity.get('context') or ''}", re.IGNORECASE)
    ]
    if resonance_candidates:
        resonance = max(resonance_candidates, key=_si_value)
        changed = next((quantity for quantity in currents if quantity is not resonance), currents[-1])
        return resonance, changed
    ordered = sorted(currents, key=_si_value, reverse=True)
    return ordered[0], ordered[1]


def _solve_rlc_resonance_frequency_multiplier(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["resonan", "resonate"]) or not any(
        cue in text for cue in ["multiple", "factor", "changed to achieve resonance", "to achieve resonance", "to resonate"]
    ):
        return None
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    target_dimensions = _target_dimensions(front_payload)
    asks_for_multiplier = (
        "frequency" in target_text
        or "ω" in target_text
        or "omega" in target_text
        or "multiple" in text
        or "factor" in text
        or "dimensionless" in target_dimensions
        or "constant" in target_dimensions
        or re.search(r"\b(?:value|coefficient|multiplier)\s+of\s+k\b|\bk\s+for\s+the\s+circuit\s+to\s+resonate\b", f"{target_text} {text}")
    )
    if not asks_for_multiplier:
        return None
    xl = _quantity_by_symbol_or_context(front_payload, "resistance", {"xl", "x_l"}, context_cues={"inductive reactance"})
    xc = _quantity_by_symbol_or_context(front_payload, "resistance", {"xc", "x_c"}, context_cues={"capacitive reactance"})
    if xl is None or xc is None:
        return None
    xl_value = _si_value(xl)
    xc_value = _si_value(xc)
    if xl_value <= 0 or xc_value <= 0:
        return None
    value = math.sqrt(xc_value / xl_value)
    spec = FORMULA_REGISTRY["rlc_resonance_frequency_multiplier"]
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"XL_initial": _quantity_trace(xl), "XC_initial": _quantity_trace(xc)},
            "constants": {},
            "binding_audit": {"policy": "frequency_multiplier_to_make_m_xl_equal_xc_over_m"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.82, route_result.confidence),
    )


def _solve_rlc_frequency_transform(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not (
        any(cue in text for cue in ["rlc", "reactance", "ac circuit", "impedance"])
        or re.search(r"\b(?:x\s*[lc]|x[lc])\s*=", text)
    ):
        return None
    factor = _frequency_change_factor(text) or _frequency_ratio_from_quantities(front_payload)
    if factor is None or factor <= 0:
        return None
    r = _quantity_by_symbol_or_context(front_payload, "resistance", {"r"}, reject_symbols={"xl", "x_l", "xc", "x_c", "z"})
    xl = _quantity_by_symbol_or_context(front_payload, "resistance", {"xl", "x_l"}, context_cues={"inductive reactance"})
    xc = _quantity_by_symbol_or_context(front_payload, "resistance", {"xc", "x_c"}, context_cues={"capacitive reactance"})
    if r is None or xl is None or xc is None:
        if "voltage" in _target_dimensions(front_payload) and (_frequency_change_factor(text) or _frequency_ratio_from_quantities(front_payload)) is not None:
            voltage_at_resonance = _solve_resistor_voltage_when_frequency_reaches_resonance(front_payload, route_result)
            if voltage_at_resonance is not None:
                return voltage_at_resonance
        return None
    resistance = _si_value(r)
    xl_initial = _si_value(xl)
    xc_initial = _si_value(xc)
    if resistance <= 0 or xl_initial < 0 or xc_initial < 0:
        return None
    xl_new = xl_initial * factor
    xc_new = xc_initial / factor
    impedance = math.sqrt(resistance * resistance + (xl_new - xc_new) ** 2)
    target_dimensions = _target_dimensions(front_payload)
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    if "power" in target_dimensions or "power" in target_text:
        voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v", "emf"})
        if voltage is None or impedance <= 0:
            return None
        value = (_si_value(voltage) ** 2) * resistance / (impedance * impedance)
        formula_id = "rlc_power_impedance"
        inputs = {
            "U": _quantity_trace(voltage),
            "R": _quantity_trace(r),
            "XL_initial": _quantity_trace(xl),
            "XC_initial": _quantity_trace(xc),
        }
        target_dimension = "power"
    elif "current" in target_dimensions or "current" in target_text:
        voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v", "emf"})
        if voltage is None or impedance <= 0:
            return None
        value = _si_value(voltage) / impedance
        formula_id = "rlc_current_from_rlcf_voltage"
        inputs = {
            "U": _quantity_trace(voltage),
            "R": _quantity_trace(r),
            "XL_initial": _quantity_trace(xl),
            "XC_initial": _quantity_trace(xc),
        }
        target_dimension = "current"
    elif "phase" in target_text or "angle" in target_text:
        value = math.atan((xl_new - xc_new) / resistance)
        formula_id = "rlc_phase_angle"
        inputs = {"R": _quantity_trace(r), "XL_initial": _quantity_trace(xl), "XC_initial": _quantity_trace(xc)}
        target_dimension = "angle"
    elif "impedance" in target_text or "resistance" in target_dimensions:
        value = impedance
        formula_id = "rlc_impedance"
        inputs = {"R": _quantity_trace(r), "XL_initial": _quantity_trace(xl), "XC_initial": _quantity_trace(xc)}
        target_dimension = "resistance"
    elif "voltage" in target_dimensions and any(cue in target_text for cue in ["resistor", "resistance", "across r"]):
        if math.isclose(xl_new, xc_new, rel_tol=1e-6, abs_tol=1e-9):
            voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v", "emf"})
            if voltage is None:
                return None
            value = _si_value(voltage)
            formula_id = "rlc_resonance_resistor_voltage"
            inputs = {"U": _quantity_trace(voltage), "XL_initial": _quantity_trace(xl), "XC_initial": _quantity_trace(xc)}
            target_dimension = "voltage"
        else:
            return None
    else:
        return None
    spec = FORMULA_REGISTRY[formula_id]
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=value,
        unit=spec.target_unit,
        formula_id=formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise, "Inductive reactance scales with frequency, while capacitive reactance scales inversely with frequency."],
        trace={
            "stage": "registry_formula_solver",
            "formula_id": formula_id,
            "expression": spec.expression,
            "target_dimension": target_dimension,
            "inputs": inputs,
            "constants": {"frequency_factor": factor},
            "transformed_reactances": {"XL": xl_new, "XC": xc_new, "Z": impedance},
            "binding_audit": {
                "policy": "rlc_frequency_transform_from_initial_reactances",
                "factor": factor,
                "XL_rule": "XL' = factor * XL",
                "XC_rule": "XC' = XC / factor",
            },
        },
        confidence=min(0.8, route_result.confidence),
    )


def _solve_resistor_voltage_when_frequency_reaches_resonance(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    factor = _frequency_change_factor(text) or _frequency_ratio_from_quantities(front_payload)
    if factor is None or factor <= 0:
        return None
    xl = _quantity_by_symbol_or_context(front_payload, "resistance", {"xl", "x_l"}, context_cues={"inductive reactance"})
    xc = _quantity_by_symbol_or_context(front_payload, "resistance", {"xc", "x_c"}, context_cues={"capacitive reactance"})
    voltage = _quantity_by_symbol_or_context(front_payload, "voltage", {"u", "v", "emf"}, context_cues={"rms", "source", "applied", "total"})
    if xl is None or xc is None or voltage is None:
        return None
    xl_new = _si_value(xl) * factor
    xc_new = _si_value(xc) / factor
    if not math.isclose(xl_new, xc_new, rel_tol=1e-6, abs_tol=1e-9):
        return None
    spec = FORMULA_REGISTRY["rlc_resonance_resistor_voltage"]
    value = _si_value(voltage)
    return SolverResult(
        True,
        _format(value, spec.target_unit),
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "registry_formula_solver",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"U": _quantity_trace(voltage), "XL_initial": _quantity_trace(xl), "XC_initial": _quantity_trace(xc)},
            "constants": {"frequency_factor": factor},
            "transformed_reactances": {"XL": xl_new, "XC": xc_new},
            "binding_audit": {"policy": "frequency_change_reaches_resonance_resistor_voltage_equals_source"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.8, route_result.confidence),
    )


def _frequency_change_factor(text: str) -> float | None:
    return extract_change_factor(text)


def _frequency_ratio_from_quantities(front_payload: dict) -> float | None:
    frequencies = _topology_component_quantities(front_payload, "frequency")
    if len(frequencies) < 2:
        return None
    initial = _si_value(frequencies[0])
    final = _si_value(frequencies[-1])
    if initial <= 0 or final <= 0:
        return None
    return final / initial


def _explicit_distance_change_factor(front_payload: dict) -> float | None:
    candidates = []
    for quantity in _topology_component_quantities(front_payload, "length"):
        haystack = f"{quantity.get('raw_text') or ''} {quantity.get('context') or ''}".lower()
        symbol = str(quantity.get("symbol") or "").lower()
        if symbol == "d" or any(cue in haystack for cue in ["separation", "plate spacing", "distance between", "between the plates"]):
            candidates.append(quantity)
    if len(candidates) < 2:
        return None
    initial = _si_value(candidates[0])
    final = _si_value(candidates[-1])
    if initial <= 0 or final <= 0:
        return None
    return final / initial


def _explicit_area_change_factor(front_payload: dict) -> float | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if any(cue in text for cue in ["split in half", "cut in half", "divided in half"]):
        return 0.5
    candidates = _topology_component_quantities(front_payload, "area")
    if len(candidates) < 2:
        return None
    initial = _si_value(candidates[0])
    final = _si_value(candidates[-1])
    if initial <= 0 or final <= 0:
        return None
    return final / initial


def _parse_time_dependent_voltage(front_payload: dict) -> dict | None:
    text = str(front_payload.get("canonical_question") or "")
    pattern = re.compile(
        r"\b(?:u|v|voltage(?:\s+at\s+time\s+t)?)\s*(?:\(\s*t\s*\))?\s*(?:=|is)\s*"
        r"(?P<amplitude>\d+(?:\.\d+)?(?:\s*√\s*2)?)\s*"
        r"(?:×|x|\*)?\s*"
        r"(?P<function>cos|sin)\s*\(?\s*(?P<omega>\d+(?:\.\d+)?(?:\s*pi|π)?)\s*t",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None
    amplitude = _parse_math_factor(match.group("amplitude"))
    omega = _parse_math_factor(match.group("omega"))
    if amplitude is None:
        return None
    return {
        "amplitude": amplitude,
        "omega": omega,
        "function": match.group("function").lower(),
        "raw_text": match.group(0),
    }


def _parse_time_dependent_current(front_payload: dict) -> dict | None:
    text = str(front_payload.get("canonical_question") or "")
    pattern = re.compile(
        r"\b(?:i|current(?:\s+at\s+time\s+t)?)\s*(?:\(\s*t\s*\))?\s*(?:=|is)\s*"
        r"(?P<amplitude>\d+(?:\.\d+)?(?:\s*√\s*2)?)\s*"
        r"(?P<function>cos|sin)\s*\(?\s*(?P<omega>\d+(?:\.\d+)?(?:\s*pi|π)?)\s*t",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None
    amplitude = _parse_math_factor(match.group("amplitude"))
    omega = _parse_math_factor(match.group("omega"))
    if amplitude is None or amplitude <= 0:
        return None
    return {
        "amplitude": amplitude,
        "omega": omega,
        "function": match.group("function").lower(),
        "raw_text": match.group(0),
    }


def _change_factor_near_current(text: str) -> float | None:
    for match in re.finditer(r"\bcurrent\b[^.?;]{0,100}", text, flags=re.IGNORECASE):
        factor = extract_change_factor(match.group(0))
        if factor is not None:
            return factor
        body = match.group(0).lower()
        if "half" in body or "halved" in body:
            return 0.5
        if "twice" in body or "doubled" in body:
            return 2.0
    return extract_change_factor(text)


def _symbolic_quantity_by_dimension(front_payload: dict, dimension: str) -> dict | None:
    for quantity in front_payload.get("symbolic_quantities") or []:
        if quantity.get("dimension") == dimension:
            return quantity
    return None


def _charge_replacement_factor(text: str) -> float | None:
    match = re.search(r"\breplaced\s+by\s+[-+]?\s*(?P<factor>\d+(?:\.\d+)?)\s*(?:q|Q)\b", text, re.IGNORECASE)
    if match:
        return float(match.group("factor"))
    if re.search(r"\bcharge\b[^.]{0,50}\b(?:double[ds]?|twice)\b", text):
        return 2.0
    if re.search(r"\bcharge\b[^.]{0,50}\b(?:halve[ds]?|half)\b", text):
        return 0.5
    return None


def _distance_replacement_factor(text: str) -> float | None:
    bodies = [match.group("body") for match in re.finditer(r"\b(?:distance|radius|separation)\b(?P<body>[^.?,;]{0,120})", text, re.IGNORECASE)]
    bodies.append(text)
    for body in reversed(bodies):
        if re.search(r"\b(?:halve[ds]?|half)\b", body, re.IGNORECASE):
            return 0.5
        if re.search(r"\b(?:double[ds]?|twice)\b", body, re.IGNORECASE):
            return 2.0
        if re.search(r"\btriple[ds]?\b", body, re.IGNORECASE):
            return 3.0
        factor = extract_change_factor(body)
        if factor is not None:
            return factor
    return None


def _format_symbolic_multiple(scale: float, symbol: str) -> str:
    if math.isclose(scale, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        return symbol
    if math.isclose(scale, round(scale), rel_tol=1e-9, abs_tol=1e-12):
        return f"{int(round(scale))}{symbol}"
    return f"{scale:.6g}{symbol}"


def _parse_math_factor(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip().lower().replace(" ", "")
    if not text:
        return None
    multiplier = 1.0
    if "√2" in text:
        multiplier *= math.sqrt(2.0)
        text = text.replace("√2", "")
    if "sqrt(2)" in text:
        multiplier *= math.sqrt(2.0)
        text = text.replace("sqrt(2)", "")
    if "π" in text or "pi" in text:
        multiplier *= math.pi
        text = text.replace("π", "").replace("pi", "")
    if text in {"", "*"}:
        return multiplier
    try:
        return float(text) * multiplier
    except ValueError:
        return _parse_safe_math_number(raw)


def _parse_safe_math_number(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip().lower()
    if not text:
        return None
    text = text.replace("π", "pi").replace("−", "-").replace("⁻", "-").replace("^", "**")
    text = text.replace("√2", "*sqrt(2)").replace("sqrt2", "sqrt(2)")
    text = re.sub(r"(?<=\d)\s*[x×]\s*10\s*\*\*\s*([-+]?\d+)", r"e\1", text)
    text = re.sub(r"(?<=\d)\s*[x×]\s*10\s*\^\s*([-+]?\d+)", r"e\1", text)
    text = re.sub(r"(?<=\d)\s*pi\b", "*pi", text)
    text = re.sub(r"(?<=\d)\s*sqrt", "*sqrt", text)
    text = re.sub(r"\)\s*(?=\d|pi|sqrt)", ")*", text)
    if not re.fullmatch(r"[0-9eE+\-*/().\s*pisqrt]+", text):
        return None
    try:
        return _safe_eval(text, {})
    except Exception:
        return None


def _parse_assignment_with_unit(text: str, symbol: str, unit: str) -> float | None:
    unit_pattern = re.escape(unit)
    pattern = re.compile(
        rf"\b{re.escape(symbol)}\s*=\s*(?P<expr>[-+0-9eE\s*/().^πpi√×x⁻]+?)\s*{unit_pattern}\b",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None
    return _parse_safe_math_number(match.group("expr"))


def _parse_turn_density_text(text: str) -> float | None:
    patterns = (
        r"(?:turn density|number of turns per meter|turns per meter)\s*(?:is|=|of)?\s*(?P<value>[-+]?\d+(?:\.\d+)?)",
        r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*turns\s*/\s*m",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group("value"))
    return None


def _synthetic_quantity(symbol: str, value: float, unit: str, dimension: str, source: str) -> dict:
    return {
        "raw_text": f"{symbol} = {value:g} {unit}",
        "value": value,
        "unit": unit,
        "raw_unit": unit,
        "symbol": symbol,
        "dimension": dimension,
        "span": None,
        "context": source,
        "confidence": 0.74,
        "entity_id": None,
        "state_id": "state:derived",
        "role": "given",
    }


def _largest_count_quantity(front_payload: dict) -> dict | None:
    counts = _topology_component_quantities(front_payload, "count")
    if not counts:
        return None
    plausible = [quantity for quantity in counts if abs(_si_value(quantity)) >= 1.0]
    return max(plausible or counts, key=lambda quantity: abs(_si_value(quantity)))


def _voltage_bounds_from_text(front_payload: dict) -> dict[str, float]:
    """Extract simple target voltage bounds used only to disambiguate branches."""

    bounds: dict[str, float] = {}
    text = str(front_payload.get("canonical_question") or "")
    numeric = r"[-+]?\d+(?:\.\d+)?(?:\s*(?:×|x)\s*10\^-?\d+)?"
    for match in re.finditer(rf"\b(?:u|v|voltage)\s*(?:is\s*)?(?P<op><|<=|>|>=)\s*(?P<value>{numeric})\s*(?P<unit>[munpμµkM]?V)\b", text, re.IGNORECASE):
        value = _parse_bound_number(match.group("value"))
        info = unit_info(match.group("unit"))
        if value is None or info is None:
            continue
        si_value = value * info.si_factor
        op = match.group("op")
        if "<" in op:
            bounds["upper"] = si_value
        if ">" in op:
            bounds["lower"] = si_value
    if bounds:
        return bounds
    for quantity in front_payload.get("quantities") or []:
        if quantity.get("dimension") != "voltage" or unit_info(quantity.get("unit") or "") is None:
            continue
        haystack = f"{quantity.get('raw_text') or ''} {quantity.get('context') or ''}".lower()
        value = _si_value(quantity)
        if re.search(r"\b(?:u|v|voltage)\s*(?:is\s*)?(?:<|less than|below|under|not exceed)", haystack):
            bounds["upper"] = value
        if re.search(r"\b(?:u|v|voltage)\s*(?:is\s*)?(?:>|greater than|above|over)", haystack):
            bounds["lower"] = value
    return bounds


def _parse_bound_number(raw: str) -> float | None:
    text = raw.replace(" ", "")
    text = text.replace("×", "x")
    try:
        if "x10^" in text.lower():
            base, exponent = re.split(r"x10\^", text, maxsplit=1, flags=re.IGNORECASE)
            return float(base) * (10 ** int(exponent))
        return float(text)
    except (TypeError, ValueError):
        return None


def _select_unique_voltage_candidate(candidates: list[dict], bounds: dict[str, float]) -> dict | None:
    if len(candidates) == 1:
        selected = dict(candidates[0])
        selected["selection_reason"] = "single_voltage_candidate"
        return selected
    filtered = list(candidates)
    if "upper" in bounds:
        upper = bounds["upper"]
        filtered = [item for item in filtered if float(item["value"]) < upper or math.isclose(float(item["value"]), upper, rel_tol=1e-9, abs_tol=1e-12)]
    if "lower" in bounds:
        lower = bounds["lower"]
        filtered = [item for item in filtered if float(item["value"]) > lower or math.isclose(float(item["value"]), lower, rel_tol=1e-9, abs_tol=1e-12)]
    unique_values = _unique_float_values(filtered)
    if len(unique_values) == 1 and filtered:
        selected = dict(filtered[0])
        selected["selection_reason"] = "unique_candidate_after_explicit_voltage_bound" if bounds else "all_candidates_same_voltage"
        return selected
    return None


def _unique_float_values(items: list[dict], *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> list[float]:
    values: list[float] = []
    for item in items:
        value = float(item["value"])
        if not any(math.isclose(value, existing, rel_tol=rel_tol, abs_tol=abs_tol) for existing in values):
            values.append(value)
    return values


def _quantity_by_symbol_or_context(
    front_payload: dict,
    dimension: str,
    symbols: set[str],
    context_cues: set[str] | None = None,
    reject_symbols: set[str] | None = None,
) -> dict | None:
    context_cues = context_cues or set()
    reject_symbols = reject_symbols or set()
    candidates = [
        quantity
        for quantity in sorted(front_payload.get("quantities", []), key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == dimension and unit_info(quantity.get("unit") or "") is not None
    ]
    for quantity in candidates:
        symbol = str(quantity.get("symbol") or "").lower()
        if symbol in symbols:
            return quantity
    for quantity in candidates:
        context = f"{quantity.get('raw_text') or ''} {quantity.get('context') or ''}".lower()
        if any(cue in context for cue in context_cues):
            return quantity
    for quantity in candidates:
        symbol = str(quantity.get("symbol") or "").lower()
        if symbol not in reject_symbols:
            return quantity
    return None


def _quantity_by_context_strict(front_payload: dict, dimension: str, context_cues: set[str]) -> dict | None:
    candidates = _topology_component_quantities(front_payload, dimension)
    question = str(front_payload.get("canonical_question") or "").lower()
    for quantity in candidates:
        span = quantity.get("span") or (None, None)
        try:
            start = int(span[0])
        except (TypeError, ValueError):
            start = -1
        if start >= 0:
            before = question[max(0, start - 80) : start]
            if any(cue.lower() in before for cue in context_cues):
                return quantity
    for quantity in candidates:
        haystack = f"{quantity.get('symbol') or ''} {quantity.get('raw_text') or ''} {quantity.get('context') or ''}".lower()
        if any(cue.lower() in haystack for cue in context_cues):
            return quantity
    return None


def _quantity_matches_symbol_or_context(quantity: dict, symbols: set[str], context_cues: set[str]) -> bool:
    symbol = str(quantity.get("symbol") or "").lower()
    if symbol in {item.lower() for item in symbols}:
        return True
    haystack = f"{quantity.get('raw_text') or ''} {quantity.get('context') or ''}".lower()
    return any(cue in haystack for cue in context_cues)


def _length_by_context(front_payload: dict, context_cues: set[str]) -> dict | None:
    candidates = _topology_component_quantities(front_payload, "length")
    for quantity in candidates:
        haystack = f"{quantity.get('raw_text') or ''} {quantity.get('context') or ''}".lower()
        if any(cue in haystack for cue in context_cues):
            return quantity
    return candidates[0] if len(candidates) == 1 else None


def _resultant_angle_rad(front_payload: dict) -> float | None:
    for quantity in front_payload.get("quantities", []):
        if quantity.get("dimension") == "angle":
            return _si_value(quantity)
    text = str(front_payload.get("canonical_question") or "").lower()
    if any(cue in text for cue in ["same direction", "same line and same direction"]):
        return 0.0
    if any(cue in text for cue in ["opposite direction", "opposite directions", "opposite sense"]):
        return math.pi
    if any(cue in text for cue in ["perpendicular", "right angle", "90 degrees", "90°"]):
        return math.pi / 2.0
    if "collinear" in text and "opposite" not in text:
        return 0.0
    return None


def _resultant_angle_source(front_payload: dict) -> str:
    for quantity in front_payload.get("quantities", []):
        if quantity.get("dimension") == "angle":
            return str(quantity.get("raw_text") or "angle")
    text = str(front_payload.get("canonical_question") or "").lower()
    for cue in ["same direction", "opposite directions", "opposite direction", "perpendicular", "right angle", "collinear"]:
        if cue in text:
            return cue
    return "inferred force angle"


def _canonical_topology_relation(front_payload: dict) -> str | None:
    topology = front_payload.get("topology_graph") or {}
    canonical = topology.get("canonical_form")
    ambiguity = set(topology.get("ambiguity") or [])
    if ambiguity:
        return None
    if canonical == "series_topology":
        return "series"
    if canonical == "parallel_topology":
        return "parallel"
    return None


def _topology_component_quantities(front_payload: dict, dimension: str) -> list[dict]:
    quantities = [
        quantity
        for quantity in front_payload.get("quantities", [])
        if quantity.get("dimension") == dimension and unit_info(quantity.get("unit") or "") is not None
    ]
    return sorted(quantities, key=lambda quantity: quantity.get("span") or (10**9, 10**9))


def _target_requests_equivalent_resistance(text: str) -> bool:
    if "impedance" in text or "reactance" in text:
        return False
    return any(cue in text for cue in ["equivalent resistance", "total resistance", "combined resistance", "effective resistance"])


def _equivalent_resistance(relation: str, resistors: list[dict]) -> dict:
    values = []
    for quantity in resistors:
        value = _si_value(quantity)
        if value <= 0:
            return {"ok": False, "issue": "non_positive_resistance_in_topology"}
        values.append(value)
    if len(values) < 2:
        return {"ok": False, "issue": "too_few_resistors_for_topology"}
    if relation == "series":
        return {"ok": True, "value": sum(values)}
    if relation == "parallel":
        reciprocal = sum(1.0 / value for value in values)
        if reciprocal <= 0:
            return {"ok": False, "issue": "invalid_parallel_resistance_reciprocal"}
        return {"ok": True, "value": 1.0 / reciprocal}
    return {"ok": False, "issue": "unsupported_topology_relation"}


def _equivalent_capacitance(relation: str, capacitors: list[dict]) -> dict:
    values = []
    for quantity in capacitors:
        value = _si_value(quantity)
        if value <= 0:
            return {"ok": False, "issue": "non_positive_capacitance_in_topology"}
        values.append(value)
    if len(values) < 2:
        return {"ok": False, "issue": "too_few_capacitors_for_topology"}
    if relation == "parallel":
        return {"ok": True, "value": sum(values)}
    if relation == "series":
        reciprocal = sum(1.0 / value for value in values)
        if reciprocal <= 0:
            return {"ok": False, "issue": "invalid_series_capacitance_reciprocal"}
        return {"ok": True, "value": 1.0 / reciprocal}
    return {"ok": False, "issue": "unsupported_topology_relation"}


def _topology_solver_result(
    *,
    formula_id: str,
    relation: str,
    value: float,
    components: list[dict],
    topology: dict,
    target_dimension: str,
    confidence: float,
    extra_trace: dict,
) -> SolverResult:
    spec = FORMULA_REGISTRY[formula_id]
    trace = {
        "stage": "topology_engine",
        "formula_id": formula_id,
        "expression": spec.expression,
        "target_dimension": target_dimension,
        "topology_relation": relation,
        "topology_graph": topology,
        "components": [_quantity_trace(quantity) for quantity in components],
        "constants": {},
        "binding_audit": {
            "policy": "canonical_topology_only",
            "component_count": len(components),
            "canonical_form": topology.get("canonical_form"),
        },
    }
    trace.update(extra_trace)
    return SolverResult(
        solved=True,
        answer=_format(value, spec.target_unit),
        value=float(value),
        unit=spec.target_unit,
        formula_id=formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace=trace,
        confidence=confidence,
    )


def _topology_unsolved_result(reason: str, task_type: str) -> SolverResult:
    return _unsolved(reason, task_type, {"stage": "topology_engine"})


def _quantity_trace(quantity: dict) -> dict:
    return {
        "raw_text": quantity.get("raw_text"),
        "symbol": quantity.get("symbol"),
        "dimension": quantity.get("dimension"),
        "unit": quantity.get("unit"),
        "si_value": _si_value(quantity),
        "entity_id": quantity.get("entity_id"),
        "state_id": quantity.get("state_id"),
    }


def _first_quantity(front_payload: dict, dimension: str) -> dict | None:
    for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9)):
        if quantity.get("dimension") == dimension and unit_info(quantity.get("unit") or "") is not None:
            return quantity
    return None


def solve_candidate_paths(
    front_payload: dict,
    route_result,
    target_dimension: str | None = None,
    exclude_formula_id: str | None = None,
    limit: int = 8,
) -> List[SolverResult]:
    """Execute all registry-owned direct formulas that can solve the same route.

    This is used by the verifier for redundant path checks. It never generates
    equations and never searches outside the code-owned formula registry.
    """

    if front_payload.get("answer_type_hint") in {"conceptual", "yes_no", "multi_output"}:
        return []
    if route_result.task_type in {"conceptual", "unknown", "multi_output"}:
        return []
    dims = _by_dimension(front_payload)
    results: List[SolverResult] = []
    for spec in _select_candidates(front_payload, route_result, dims):
        if exclude_formula_id and spec.formula_id == exclude_formula_id:
            continue
        if target_dimension and spec.target_dimension != target_dimension:
            continue
        executed = _execute_formula_spec(spec, front_payload, dims)
        if not executed["ok"]:
            continue
        value = float(executed["value"])
        results.append(
            SolverResult(
                solved=True,
                answer=_format(value, spec.target_unit),
                value=value,
                unit=spec.target_unit,
                formula_id=spec.formula_id,
                principle_id=spec.principle_id,
                premises=[spec.premise],
                trace={
                    "stage": "registry_formula_redundant_path",
                    "formula_id": spec.formula_id,
                    "expression": spec.expression,
                    "target_dimension": spec.target_dimension,
                    "inputs": executed["inputs"],
                    "constants": executed["constants"],
                    "binding_audit": executed["binding_audit"],
                    "verification_path": True,
                },
                confidence=min(0.88, route_result.confidence),
            )
        )
        if len(results) >= limit:
            break
    return results


def _select_candidates(
    front_payload: dict,
    route_result,
    dims: dict[str, list[dict]],
    allowed_formula_ids: Iterable[str] | None = None,
) -> list[FormulaSpec]:
    available = [dimension for dimension, values in dims.items() for _ in values]
    target_dimensions = _target_dimensions(front_payload)
    allowed = set(allowed_formula_ids or [])
    scored: list[tuple[float, str, FormulaSpec]] = []
    for formula_id, spec in FORMULA_REGISTRY.items():
        if formula_id not in FORMULA_IDS:
            continue
        if allowed and formula_id not in allowed:
            continue
        if spec.task_type != route_result.task_type:
            continue
        if _formula_is_metadata_only(spec):
            continue
        if not _formula_context_allowed(spec, front_payload):
            continue
        missing = _missing_required_dimensions(available, spec.required_dimensions)
        if missing:
            continue
        if not _is_safe_formula_shape(spec.expression):
            continue
        score = 0.0
        if spec.target_dimension in target_dimensions:
            score -= 50.0 - target_dimensions.index(spec.target_dimension)
        score += len(spec.required_dimensions)
        text = str(front_payload.get("canonical_question") or "").lower()
        if spec.formula_id == "magnetic_flux_angle" and "angle" in text:
            score -= 25.0
        if spec.formula_id == "rlc_current_from_rlcf_voltage" and any(cue in text for cue in ["rlc", "impedance", "reactance", "ac circuit"]):
            score -= 35.0
        if spec.formula_id == "ohm_current" and any(cue in text for cue in ["rlc", "impedance", "reactance", "ac circuit"]):
            score += 25.0
        scored.append((score, formula_id, spec))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [spec for _, _, spec in scored]


def _execute_formula_spec(spec: FormulaSpec, front_payload: dict, dims: dict[str, list[dict]]) -> dict:
    sanitized = _sanitize_equation(spec.expression)
    trace = {
        "stage": "registry_formula_execution",
        "formula_id": spec.formula_id,
        "expression": spec.expression,
        "ok": False,
        "issues": [],
    }
    if sanitized is None:
        trace["issues"].append("formula_expression_not_safe")
        return {"ok": False, "trace": trace}
    lhs, rhs = sanitized.split("=", 1)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs.strip()):
        trace["issues"].append("formula_not_explicit_target_assignment")
        return {"ok": False, "trace": trace}
    symbols = _symbols_in_expression(rhs)
    inputs = _bind_symbols(symbols, spec, front_payload, dims)
    if inputs["issues"]:
        trace["issues"].extend(inputs["issues"])
        trace["symbols"] = symbols
        return {"ok": False, "trace": trace}
    try:
        value = _safe_eval(rhs, {**inputs["values"], **inputs["constants"]})
    except Exception as exc:
        trace["issues"].append(f"safe_eval_failed:{type(exc).__name__}")
        trace["error"] = str(exc)[:160]
        return {"ok": False, "trace": trace}
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        trace["issues"].append("non_finite_formula_value")
        return {"ok": False, "trace": trace}
    trace["ok"] = True
    return {
        "ok": True,
        "value": float(value),
        "inputs": inputs["input_trace"],
        "constants": inputs["constant_trace"],
        "binding_audit": inputs["binding_audit"],
        "trace": trace,
    }


def _bind_symbols(symbols: list[str], spec: FormulaSpec, front_payload: dict, dims: dict[str, list[dict]]) -> dict:
    issues: list[str] = []
    values: dict[str, float] = {}
    constants: dict[str, float] = {}
    input_trace: dict[str, dict] = {}
    constant_trace: dict[str, float] = {}
    dimension_counters: dict[str, int] = defaultdict(int)
    binding_audit: dict[str, dict] = {}

    for symbol in symbols:
        if symbol in _SAFE_FUNCTIONS:
            continue
        constant = _constant_value(symbol, front_payload)
        if constant is not None:
            constants[symbol] = constant
            constant_trace[symbol] = constant
            continue
        dimension = _infer_symbol_dimension(symbol, spec)
        if dimension is None:
            issues.append(f"unbound_symbol:{symbol}")
            continue
        values_for_dimension = dims.get(dimension) or []
        selected = _choose_quantity_for_symbol(
            symbol,
            dimension,
            values_for_dimension,
            dimension_counters[dimension],
            allow_ordered_repeated=spec.required_dimensions.count(dimension) > 1,
        )
        binding_audit[symbol] = selected["audit"]
        if selected["issue"]:
            issues.append(selected["issue"])
            continue
        index = selected["index"]
        quantity = values_for_dimension[index]
        try:
            value = _si_value(quantity)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        values[symbol] = value
        input_trace[symbol] = {
            "dimension": dimension,
            "raw_text": quantity.get("raw_text"),
            "unit": quantity.get("unit"),
            "si_value": value,
            "symbol": quantity.get("symbol"),
            "entity_id": quantity.get("entity_id"),
            "state_id": quantity.get("state_id"),
            "binding_policy": selected["audit"].get("policy"),
            "candidate_count": selected["audit"].get("candidate_count"),
        }
        dimension_counters[dimension] = max(dimension_counters[dimension], index + 1)

    return {
        "issues": issues,
        "values": values,
        "constants": constants,
        "input_trace": input_trace,
        "constant_trace": constant_trace,
        "binding_audit": binding_audit,
    }


def _choose_quantity_for_symbol(
    symbol: str,
    dimension: str,
    candidates: list[dict],
    fallback_index: int,
    allow_ordered_repeated: bool = False,
) -> dict:
    audit = {
        "symbol": symbol,
        "dimension": dimension,
        "candidate_count": len(candidates),
        "candidate_symbols": [candidate.get("symbol") for candidate in candidates],
        "policy": None,
    }
    if not candidates:
        return {"index": None, "issue": f"missing_dimension_for_symbol:{symbol}:{dimension}", "audit": audit}

    exact_matches = [
        index
        for index, candidate in enumerate(candidates)
        if str(candidate.get("symbol") or "").lower() == symbol.lower()
    ]
    if len(exact_matches) == 1:
        audit["policy"] = "exact_symbol_match"
        audit["selected_index"] = exact_matches[0]
        return {"index": exact_matches[0], "issue": None, "audit": audit}
    if len(exact_matches) > 1:
        state_ids = [candidates[index].get("state_id") for index in exact_matches]
        if len(set(state_ids)) > 1:
            latest = max(exact_matches, key=lambda index: (candidates[index].get("span") or (0, 0))[0])
            audit["policy"] = "latest_state_exact_symbol_match"
            audit["selected_index"] = latest
            audit["candidate_state_ids"] = state_ids
            return {"index": latest, "issue": None, "audit": audit}
        audit["policy"] = "ambiguous_exact_symbol_match"
        return {"index": None, "issue": f"ambiguous_symbol_binding:{symbol}:{dimension}", "audit": audit}

    explicit_index = _explicit_symbol_index(symbol)
    if explicit_index is not None:
        indexed_matches = [
            index
            for index, candidate in enumerate(candidates)
            if _explicit_symbol_index(str(candidate.get("symbol") or "")) == explicit_index
        ]
        if len(indexed_matches) == 1:
            audit["policy"] = "matching_numeric_suffix"
            audit["selected_index"] = indexed_matches[0]
            return {"index": indexed_matches[0], "issue": None, "audit": audit}
        if not indexed_matches and explicit_index < len(candidates) and not any(candidate.get("symbol") for candidate in candidates):
            audit["policy"] = "ordered_unlabeled_repeated_dimension"
            audit["selected_index"] = explicit_index
            return {"index": explicit_index, "issue": None, "audit": audit}
        if not indexed_matches and explicit_index < len(candidates) and allow_ordered_repeated:
            audit["policy"] = "ordered_labeled_repeated_dimension"
            audit["selected_index"] = explicit_index
            audit["reason"] = "formula_index_bound_to_canonical_question_order"
            return {"index": explicit_index, "issue": None, "audit": audit}
        audit["policy"] = "missing_or_ambiguous_numeric_suffix"
        return {"index": None, "issue": f"missing_indexed_symbol_binding:{symbol}:{dimension}", "audit": audit}

    if len(candidates) == 1:
        audit["policy"] = "single_candidate_dimension"
        audit["selected_index"] = 0
        return {"index": 0, "issue": None, "audit": audit}

    context_rank = _contextual_symbol_matches(symbol, dimension, candidates)
    if len(context_rank) == 1:
        audit["policy"] = "contextual_symbol_role_match"
        audit["selected_index"] = context_rank[0]
        audit["role"] = symbol
        return {"index": context_rank[0], "issue": None, "audit": audit}

    if fallback_index < len(candidates) and (_formula_symbol_is_ordered(symbol) or allow_ordered_repeated):
        audit["policy"] = "ordered_repeated_dimension"
        audit["selected_index"] = fallback_index
        return {"index": fallback_index, "issue": None, "audit": audit}

    audit["policy"] = "ambiguous_dimension_binding"
    return {"index": None, "issue": f"ambiguous_dimension_binding:{symbol}:{dimension}", "audit": audit}


def _contextual_symbol_matches(symbol: str, dimension: str, candidates: list[dict]) -> list[int]:
    lower = symbol.lower()
    role_cues: dict[str, tuple[str, ...]] = {
        "r": ("radius",),
        "radius": ("radius",),
        "d": ("distance", "separation", "plate separation", "between the plates"),
        "z": ("axis", "axial", "from the center", "above the center"),
        "lambda": ("linear charge density", "charge per unit length"),
        "lam": ("linear charge density", "charge per unit length"),
        "sigma": ("surface charge density", "charge density"),
    }
    cues = role_cues.get(lower)
    if not cues:
        return []
    matches = []
    for index, candidate in enumerate(candidates):
        haystack = f"{candidate.get('raw_text') or ''} {candidate.get('context') or ''}".lower()
        if any(cue in haystack for cue in cues):
            matches.append(index)
    return matches


def _explicit_symbol_index(symbol: str) -> int | None:
    match = re.search(r"(\d+)$", symbol or "")
    if match:
        return max(0, int(match.group(1)) - 1)
    return None


def _formula_symbol_is_ordered(symbol: str) -> bool:
    return _explicit_symbol_index(symbol) is not None or any(cue in symbol.lower() for cue in ["initial", "final", "useful", "dissipated"])


def _infer_symbol_dimension(symbol: str, spec: FormulaSpec) -> str | None:
    lower = symbol.lower()
    required = set(spec.required_dimensions)
    if lower in {"epsilon_r", "eps_r", "pi", "k", "epsilon0", "mu0"}:
        return None
    if lower.startswith("delta_phi") or lower in {"phi", "flux"}:
        return "magnetic_flux"
    if lower.startswith("delta_i"):
        return "current"
    if lower.startswith("delta_t"):
        return "time"
    if lower in {"theta", "phi_angle"}:
        return "angle"
    if lower in {"rho", "ρ"}:
        return "resistivity"
    if lower in {"omega", "ω"}:
        return "angular_frequency"
    if re.fullmatch(r"q\d*|q_[a-z0-9]+|q", lower):
        return "charge"
    if re.fullmatch(r"u\d*|v\d*|emf|e", lower):
        if "electric_field" in required and lower == "e":
            return "electric_field"
        if "energy" in required and lower == "e":
            return "energy"
        return "voltage"
    if re.fullmatch(r"i\d*|i_[a-z0-9]+", lower):
        return "current"
    if re.fullmatch(r"p\d*|p_total", lower):
        return "power"
    if lower.startswith("w") or lower in {"energy", "useful", "dissipated", "initial_energy", "final_energy"}:
        return "energy"
    if lower.startswith("f") and "force" in required and "frequency" not in required:
        return "force"
    if re.fullmatch(r"f\d*|freq|frequency", lower):
        return "frequency"
    if lower in {"t", "t1", "t2", "tau"}:
        return "time"
    if lower in {"n", "n1", "n2"}:
        if "number_density" in required:
            return "number_density"
        if "turn_density" in required:
            return "turn_density"
        return "count"
    if lower in {"lambda", "λ", "lam"}:
        return "linear_charge_density"
    if lower.startswith("r") or lower in {"z", "xl", "xc"}:
        if "resistance" in required and lower not in {"r0", "radius"}:
            return "resistance"
        return "length"
    if lower.startswith("c"):
        return "capacitance" if "capacitance" in required or spec.target_dimension == "capacitance" else "charge"
    if symbol == "l" and "length" in required:
        return "length"
    if symbol == "L" and "inductance" in required:
        return "inductance"
    if lower.startswith("l"):
        if "inductance" in required and "length" not in required:
            return "inductance"
        return "length"
    if lower in {"a", "area", "s"}:
        return "area" if "area" in required else "length"
    if lower in {"d", "x", "z", "h", "radius"}:
        return "length"
    if lower in {"m", "mass"}:
        return "mass"
    if lower in {"b"}:
        return "magnetic_field"
    if lower in {"sigma"}:
        return "surface_charge_density"
    if lower in {"v_d", "vd", "velocity", "speed"}:
        return "velocity"
    if lower in {"percent", "fraction"}:
        return "percent"
    if lower in {"turns"}:
        return "count"
    return None


def _symbol_index(symbol: str, fallback_index: int) -> int:
    match = re.search(r"(\d+)$", symbol)
    if match:
        return max(0, int(match.group(1)) - 1)
    named_order = {
        "initial": 0,
        "final": 1,
        "useful": 1,
        "dissipated": 0,
    }
    lowered = symbol.lower()
    for cue, index in named_order.items():
        if cue in lowered:
            return index
    return fallback_index


def _constant_value(symbol: str, front_payload: dict) -> float | None:
    lower = symbol.lower()
    if lower == "pi":
        return math.pi
    if lower == "k":
        return 9e9
    if lower == "epsilon0":
        return 8.8541878128e-12
    if lower == "mu0":
        return 4 * math.pi * 1e-7
    if lower in {"epsilon_r", "eps_r"}:
        for constant in front_payload.get("numeric_constants") or []:
            symbol = str(constant.get("symbol") or "").lower()
            if constant.get("dimension") != "permittivity" and symbol not in {"epsilon_r", "eps_r", "ε_r", "εr"}:
                continue
            try:
                value = float(constant.get("value"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                return value
        text = str(front_payload.get("canonical_question") or "").lower()
        if "dielectric" not in text or "air" in text or "vacuum" in text:
            return 1.0
    if lower == "g":
        return 9.8
    return None


def _dielectric_constant_value(front_payload: dict) -> float | None:
    text = str(front_payload.get("canonical_question") or "")
    patterns = (
        r"(?:dielectric\s+constant|relative\s+permittivity|epsilon_r|eps_r|ε\s*_?\s*r|εr|ε)\s*(?:=|is|of)?\s*(?P<value>\d+(?:\.\d+)?)",
        r"(?:dielectric\s+with|dielectric\s+of)\s+(?P<value>\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        parsed = float(match.group("value"))
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    value = _constant_value("epsilon_r", front_payload)
    if value is not None:
        return value
    return None


def _safe_eval(expression: str, values: dict[str, float]) -> float:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"unsafe expression node {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
                raise ValueError("unsafe function call")
        if isinstance(node, ast.Name) and node.id not in values and node.id not in _SAFE_FUNCTIONS:
            raise ValueError(f"unbound name {node.id}")
    env = {name: getattr(math, name) for name in ("sqrt", "sin", "cos", "tan", "atan", "exp", "log")}
    env["abs"] = abs
    env["pi"] = math.pi
    env.update(values)
    return float(eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, env))


def _sanitize_equation(expression: str) -> str | None:
    if not isinstance(expression, str) or expression.count("=") != 1:
        return None
    if any(cue in expression.lower() for cue in [" or ", " by ", " when ", " at ", "vector", "integral", "sum("]):
        return None
    if "'" in expression or "|" in expression:
        return None
    sanitized = expression.replace("^", "**").replace("π", "pi")
    sanitized = re.sub(r"\s*=\s*", "=", sanitized.strip())
    sanitized = re.sub(r"(?<=\d)(?=[A-Za-z_(])", "*", sanitized)
    sanitized = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=[A-Za-z_(])", "*", sanitized)
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().=\s]+", sanitized):
        return None
    return sanitized


def _is_safe_formula_shape(expression: str) -> bool:
    return _sanitize_equation(expression) is not None


def _symbols_in_expression(expression: str) -> list[str]:
    seen: list[str] = []
    for symbol in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression):
        if symbol not in seen:
            seen.append(symbol)
    return seen


def _formula_is_metadata_only(spec: FormulaSpec) -> bool:
    expression = spec.expression.lower()
    return spec.formula_id in {"conceptual_direct", "yes_no_direct", "multi_output_direct", "measurement_error_direct"} or not spec.required_dimensions or any(
        cue in expression for cue in ["deterministic ", "vector_sum", "vector integral", " by symmetry"]
    )


def _formula_context_allowed(spec: FormulaSpec, front_payload: dict) -> bool:
    text = str(front_payload.get("canonical_question") or "").lower()
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    formula_id = spec.formula_id
    geometry_cues = {
        "right_isosceles": ("right isosceles", "isosceles right"),
        "equilateral": ("equilateral",),
        "triangle": ("triangle", "vertices"),
        "square": ("square",),
        "perpendicular": ("perpendicular", "bisector"),
        "long_wire": ("long wire", "straight wire", "straight conductor", "long straight"),
        "circular_loop": ("loop", "circular", "coil"),
        "loop_current": ("loop", "circular", "coil"),
        "zero_line": ("zero",),
        "ring": ("ring",),
        "semicircular": ("semicircle", "semicircular"),
        "rod": ("rod", "wire"),
        "disk": ("disk",),
        "sheets": ("sheet", "plate"),
        "dielectric": ("dielectric",),
    }
    for marker, cues in geometry_cues.items():
        if marker in formula_id and not any(cue in text for cue in cues):
            if marker == "triangle" and formula_id == "coulomb_force_triangle_sides" and _collinear_two_source_metric_context(text):
                continue
            return False
    if spec.formula_id == "rc_time_constant":
        return "time constant" in target_text or "time constant" in text or "tau" in target_text or "τ" in target_text
    if spec.formula_id == "capacitor_geometry_scaled_capacitance" and not (
        any(cue in text for cue in ["distance", "separation", "dielectric", "plate spacing", "plate area", "split in half", "cut in half"])
        and (has_change_factor_cue(text) or re.search(r"\bd\s*=", text))
    ):
        return False
    if spec.task_type in {"inductive_reactance", "capacitive_reactance"} and "impedance" in text:
        return False
    if spec.formula_id in {"rlc_power_resonance", "rlc_current_resonance", "rlc_voltage_resonance"} and "resonance" not in text and "resonant" not in text:
        return False
    if spec.formula_id.startswith("rlc_") and not (
        any(cue in text for cue in ["rlc", "impedance", "reactance", "resonance", "ac circuit", "lcω", "lcw", "quadrature", "uam", "u_am"])
        or re.search(r"\bz\s*=", text)
        or re.search(r"\b(?:x\s*[lc]|x[lc])\s*=", text)
    ):
        return False
    if spec.formula_id in {"resistance_resistivity", "resistivity_from_resistance"} and not any(cue in text for cue in ["resistivity", "rho", "ρ", "wire", "conductor"]):
        return False
    if spec.formula_id == "coulomb_force_direction_superposition" and "direction" not in text:
        return False
    if spec.formula_id in {"lorentz_force_magnetic", "wire_magnetic_force"} and "angle" in text and "perpendicular" not in text:
        return False
    if spec.formula_id in {"lorentz_force_perpendicular", "wire_magnetic_force_perpendicular"} and "angle" in text and "perpendicular" not in text:
        return False
    return True


def _collinear_two_source_metric_context(text: str) -> bool:
    return any(
        cue in text
        for cue in ["collinear", "same line", "straight line", "opposite sides", "passing through", "line segment", "endpoints"]
    ) and any(cue in text for cue in ["distance", "distances", "from", "separated", "apart", "cm", " m"])


def _target_dimensions(front_payload: dict) -> list[str]:
    ordered: list[str] = []

    def add(dimension: str | None) -> None:
        if dimension and dimension not in ordered:
            ordered.append(dimension)

    for quantity in front_payload.get("symbolic_quantities", []):
        add(quantity.get("dimension"))
    text = " ".join(front_payload.get("target_hints", [])).lower()
    for keyword, dimension in TARGET_DIMENSION_KEYWORDS:
        if keyword in text:
            add(dimension)
    return ordered


def _missing_required_dimensions(available_dimensions: Iterable[str], required_dimensions: tuple[str, ...]) -> list[str]:
    pool = Counter(available_dimensions)
    missing: list[str] = []
    for dimension in required_dimensions:
        if pool[dimension] <= 0:
            missing.append(dimension)
        else:
            pool[dimension] -= 1
    return missing


def _by_dimension(front_payload: dict) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for quantity in front_payload.get("quantities", []):
        dimension = quantity.get("dimension")
        if dimension:
            out.setdefault(dimension, []).append(quantity)
    return out


def _si_value(quantity: dict) -> float:
    info = unit_info(quantity.get("unit") or "")
    if info is None:
        raise ValueError(f"unknown_unit:{quantity.get('unit')}")
    return float(quantity["value"]) * info.si_factor


def _format(value: float, unit: str) -> str:
    return f"{value:.6g} {unit}".strip()


def _unsolved(reason: str, task_type: str, extra: Optional[dict] = None) -> SolverResult:
    trace = {"stage": "registry_formula_solver", "reason": reason, "task_type": task_type}
    if extra:
        trace.update(extra)
    return SolverResult(False, "", None, None, None, None, [], trace, 0.0)


TARGET_DIMENSION_KEYWORDS = (
    ("phase angle", "angle"),
    ("angle", "angle"),
    ("capacitance", "capacitance"),
    ("area", "area"),
    ("plate area", "area"),
    ("charge", "charge"),
    ("turns", "count"),
    ("number of turns", "count"),
    ("current", "current"),
    ("angular frequency", "angular_frequency"),
    ("omega", "angular_frequency"),
    ("voltage", "voltage"),
    ("potential difference", "voltage"),
    ("emf", "voltage"),
    ("resistance", "resistance"),
    ("impedance", "resistance"),
    ("reactance", "resistance"),
    ("power factor", "dimensionless"),
    ("percentage uncertainty", "percent"),
    ("percent uncertainty", "percent"),
    ("percentage error", "percent"),
    ("percent error", "percent"),
    ("relative error", "percent"),
    ("random error", "uncertainty"),
    ("absolute error", "uncertainty"),
    ("measurement error", "uncertainty"),
    ("power", "power"),
    ("energy", "energy"),
    ("work", "energy"),
    ("heat", "energy"),
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
    ("velocity", "velocity"),
    ("inductance", "inductance"),
    ("turn density", "turn_density"),
    ("turns per meter", "turn_density"),
)


_SAFE_FUNCTIONS = {"sqrt", "sin", "cos", "tan", "atan", "exp", "log", "abs", "pi"}
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)
