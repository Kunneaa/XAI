"""Deterministic task router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .registries import TASK_TYPES


@dataclass(frozen=True)
class RouteResult:
    task_type: str
    answer_type: str
    confidence: float
    reasons: List[str]

    def to_dict(self):
        return {
            "task_type": self.task_type,
            "answer_type": self.answer_type,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }


def route(front_payload: dict) -> RouteResult:
    text = front_payload["canonical_question"].lower()
    dims = [q["dimension"] for q in front_payload["quantities"]]
    concepts = set(front_payload["concepts"])
    answer_type = front_payload["answer_type_hint"]
    target_text = " ".join(front_payload.get("target_hints", [])).lower()
    reasons: List[str] = []

    def has(*required: str) -> bool:
        pool = list(dims)
        for dim in required:
            if dim not in pool:
                return False
            pool.remove(dim)
        return True

    def has_symbol(*symbols: str) -> bool:
        wanted = {symbol.lower() for symbol in symbols}
        present = {
            str(quantity.get("symbol") or "").lower().replace("_", "")
            for quantity in front_payload["quantities"]
        }
        return wanted <= present

    def has_particle_name() -> bool:
        return "electron" in text or "proton" in text

    task_type = "unknown"
    confidence = 0.35

    if answer_type == "multi_output":
        task_type, confidence = "multi_output", 0.76
        reasons.append("multi-output target detected; deterministic executor must preserve output order")
    elif _asks_for(target_text, text, "electric field", "field strength") and _asks_for(target_text, text, "force") and has("charge", "charge", "charge", "length"):
        answer_type = "multi_output"
        task_type, confidence = "multi_output", 0.78
        reasons.append("field-and-force target detected; deterministic executor must preserve both outputs")
    elif (
        answer_type == "yes_no"
        and ("resonance" in text or "resonate" in text or "resonant" in text)
        and (has("inductance", "capacitance", "frequency") or "xl = xc" in text or "x_l = x_c" in text)
    ):
        task_type, confidence = "conceptual", 0.82
        reasons.append("yes/no resonance condition should be checked as a condition, not solved as impedance")
    elif _asks_for(target_text, text, "resultant force", "net force", "angle between the two forces") and (
        has("force", "force") or ("two" in text and "each" in text and has("force"))
    ):
        task_type, confidence = "resultant_force", 0.86
        reasons.append("resultant-force wording with two force magnitudes")
    elif _asks_for(target_text, text, "q") and has("force", "length") and ("q1 = q2" in text or "q1 = q2 = q" in text):
        task_type, confidence = "equal_charge_coulomb", 0.86
        reasons.append("equal-charge Coulomb inversion wording")
    elif _asks_for(target_text, text, "charge", "magnitude of charge") and has("force", "charge", "length") and ("point charge" in text or "electric field of" in text or "coulomb" in text):
        answer_type = "numeric"
        task_type, confidence = "coulomb_force", 0.84
        reasons.append("Coulomb-law unknown-charge inversion wording")
    elif _asks_for(target_text, text, "dielectric constant") and has("capacitance", "area", "length"):
        task_type, confidence = "dielectric_constant", 0.88
        reasons.append("parallel-plate dielectric constant wording")
    elif (
        "parallel" in text
        and "capacitor" in text
        and _asks_for(target_text, text, "charge")
        and has("voltage")
        and (has("area", "length") or ("circular" in text and dims.count("length") >= 2))
    ):
        task_type, confidence = "capacitor_charge", 0.86
        reasons.append("parallel-plate charge wording with plate geometry and voltage")
    elif (
        "parallel" in text
        and "capacitor" in text
        and _asks_for(target_text, text, "capacitance")
        and (has("area", "length") or ("circular" in text and dims.count("length") >= 2))
    ):
        task_type, confidence = "capacitance", 0.88
        reasons.append("parallel-plate capacitance wording with plate geometry")
    elif _asks_for(
        target_text,
        text,
        "final voltage",
        "voltage across the combined",
        "voltage across the combination",
        "voltage across the capacitor combination",
        "voltage across them after",
    ) and has("capacitance", "capacitance", "voltage", "voltage"):
        task_type, confidence = "capacitor_final_voltage", 0.86
        reasons.append("capacitor charge-sharing final voltage wording")
    elif (
        "capacitor" in text
        and _asks_for(target_text, text, "potential difference", "voltage")
        and has("voltage")
        and any(cue in text for cue in ["disconnected", "while still connected", "connected to the source", "remains connected", "dielectric", "moved apart", "distance between"])
    ):
        task_type, confidence = "capacitor_final_voltage", 0.84
        reasons.append("capacitor state-change voltage wording")
    elif (
        "capacitor" in text
        and (
            "new capacitance" in target_text
            or "capacitance c" in target_text
            or target_text.strip() in {"capacitance", "the capacitance"}
        )
        and has("capacitance")
        and any(cue in text for cue in ["distance", "plate separation", "split in half", "dielectric", "halved", "doubled", "changed to"])
    ):
        task_type, confidence = "capacitance", 0.84
        reasons.append("capacitor geometry state-change capacitance wording")
    elif "capacitor" in text and "series" in text and _asks_for(target_text, text, "voltage across capacitor", "voltage across c") and has("capacitance", "capacitance", "voltage"):
        task_type, confidence = "capacitor_final_voltage", 0.84
        reasons.append("series capacitor voltage-divider wording")
    elif "capacitor" in text and "series" in text and _asks_for(target_text, text, "electric field", "field strength") and has("capacitance", "capacitance", "voltage", "length"):
        task_type, confidence = "electric_field_point", 0.82
        reasons.append("series capacitor electric-field wording with plate separation")
    elif "capacitor" in text and "series" in text and _asks_for(target_text, text, "c'", "capacitance") and has("capacitance", "voltage", "charge"):
        task_type, confidence = "capacitance", 0.82
        reasons.append("series capacitor unknown capacitance from final charge wording")
    elif ("internal resistance" in text or "terminal voltage" in target_text or "emf" in text) and (
        has("voltage", "current", "resistance") or has("voltage", "resistance", "resistance")
    ):
        task_type, confidence = "internal_resistance", 0.86
        reasons.append("source internal-resistance wording with compatible quantities")
    elif _asks_for(target_text, text, "heat", "thermal energy", "joule heat", "energy dissipated") and has("current", "resistance", "time"):
        task_type, confidence = "joule_heating", 0.9
        reasons.append("Joule-heating wording with current, resistance, and time")
    elif _asks_for(target_text, text, "work", "energy") and "capacitor" not in text and has("charge", "voltage"):
        task_type, confidence = "electric_work", 0.86
        reasons.append("electric-work wording with charge and potential difference")
    elif _asks_for(target_text, text, "mass") and "equilibrium" in text and has("charge", "electric_field", "angle") and any(cue in text for cue in ["thread", "string", "suspended"]):
        task_type, confidence = "force_in_electric_field", 0.82
        reasons.append("suspended charged-particle equilibrium wording with electric field and string angle")
    elif _asks_for(target_text, text, "angle", "deflection") and has("mass", "charge", "electric_field", "acceleration") and any(cue in text for cue in ["thread", "string", "suspended"]):
        task_type, confidence = "force_in_electric_field", 0.82
        reasons.append("suspended charged-particle equilibrium angle wording")
    elif _asks_for(target_text, text, "electric field", "field strength") and has("voltage", "length") and ("uniform" in text or "plates" in text):
        task_type, confidence = "uniform_field", 0.88
        reasons.append("uniform-field wording with voltage and distance")
    elif (
        _asks_for(target_text, text, "electric field", "field strength")
        and "dielectric" in text
        and has("electric_field")
    ):
        task_type, confidence = "electric_field_point", 0.78
        reasons.append("dielectric field-scaling wording with known original field")
    elif (
        _asks_for(target_text, text, "electric field", "field strength")
        and "surface charge densit" in text
        and has("surface_charge_density")
        and any(cue in text for cue in ["wide", "infinite", "sheet", "plate"])
    ):
        task_type, confidence = "electric_field_point", 0.8
        reasons.append("wide sheet/plate field wording with surface charge density")
    elif (
        _asks_for(target_text, text, "electric field", "field strength")
        and (
            has("linear_charge_density", "length")
            or has("surface_charge_density", "length", "length")
        )
        and any(cue in text for cue in ["wire", "rod", "disk", "uniformly charged"])
    ):
        task_type, confidence = "electric_field_point", 0.8
        reasons.append("continuous charge distribution electric-field wording")
    elif (
        _asks_for(target_text, text, "charge", "magnitude of q", "sign and magnitude")
        and "dielectric" in text
        and "capacitor" not in text
        and "breakdown" not in text
        and "plate" not in text
        and has("electric_field", "length")
    ):
        task_type, confidence = "electric_field_point", 0.78
        reasons.append("point-charge magnitude from dielectric field wording")
    elif (
        _asks_for(target_text, text, "electric field", "field strength")
        and "midpoint" in text
        and "same electric field line" in text
        and has("electric_field", "electric_field")
    ):
        task_type, confidence = "electric_field_point", 0.78
        reasons.append("point-charge field-line midpoint wording with two field strengths")
    elif _asks_for(target_text, text, "charge") and "equilibrium" in text and has("mass", "electric_field", "acceleration"):
        task_type, confidence = "force_in_electric_field", 0.82
        reasons.append("charged dust equilibrium wording with mg balanced by qE")
    elif _asks_for(target_text, text, "electric field", "field strength") and "equilibrium" in text and has("mass", "charge", "acceleration"):
        task_type, confidence = "electric_field_point", 0.82
        reasons.append("charged-particle equilibrium wording with mg balanced by qE")
    elif _asks_for(target_text, text, "distance", "speed", "velocity") and has_particle_name() and ("uniform electric field" in text or "electric field" in text):
        task_type, confidence = "charged_particle_motion", 0.82
        reasons.append("charged-particle motion wording in an electric field")
    elif (
        _asks_for(target_text, text, "time", "how long")
        and ("charging" in text or "discharging" in text or "charge" in text or "discharge" in text)
        and has("resistance", "capacitance", "percent")
    ):
        task_type, confidence = "rc_circuit", 0.84
        reasons.append("RC charging/discharging fraction wording with R, C, and target fraction")
    elif _asks_for(target_text, text, "time constant", "tau", "τ") and has("resistance", "capacitance"):
        task_type, confidence = "rc_circuit", 0.9
        reasons.append("RC time-constant wording with R and C")
    elif _asks_for(target_text, text, "induced electromotive force", "emf", "induced emf") and has("inductance", "current", "time"):
        task_type, confidence = "faraday_induction", 0.84
        reasons.append("self-induction emf wording with inductance and current change")
    elif _asks_for(target_text, text, "induced electromotive force", "emf", "induced emf") and has("magnetic_flux", "time"):
        task_type, confidence = "faraday_induction", 0.84
        reasons.append("Faraday induction wording with magnetic-flux change and time")
    elif "transformer" in text and _asks_for(target_text, text, "voltage", "secondary voltage", "primary voltage") and has("voltage", "count", "count"):
        task_type, confidence = "transformer", 0.84
        reasons.append("ideal-transformer voltage ratio wording with turns and voltage")
    elif (
        any(cue in text for cue in ["drift velocity", "carrier density", "number density"])
        and _asks_for(target_text, text, "current")
        and has("number_density", "charge", "area", "velocity")
    ):
        task_type, confidence = "drift_current", 0.84
        reasons.append("microscopic drift-current wording with n, q, area, and drift speed")
    elif "wheatstone" in text and ("balance" in text or "balanced" in text) and _asks_for(target_text, text, "resistance", "r4") and has("resistance", "resistance", "resistance"):
        task_type, confidence = "wheatstone_bridge", 0.82
        reasons.append("balanced Wheatstone bridge wording with three known resistors")
    elif _asks_for(target_text, text, "magnetic force", "lorentz force", "force") and has("charge", "velocity", "magnetic_field") and ("magnetic field" in text or "lorentz" in text):
        task_type, confidence = "lorentz_force", 0.84
        reasons.append("Lorentz-force wording with charge, speed, and magnetic field")
    elif _asks_for(target_text, text, "force") and has("magnetic_field", "current", "length") and ("wire" in text or "conductor" in text):
        task_type, confidence = "wire_magnetic_force", 0.84
        reasons.append("magnetic force on current-carrying wire with B, I, and length")
    elif _asks_for(target_text, text, "energy") and "capacitor" in text and (has("capacitance", "voltage") or has("charge", "capacitance") or has("charge", "voltage")):
        task_type, confidence = "capacitor_energy", 0.95
        reasons.append("capacitor energy wording with compatible capacitor quantities")
    elif _asks_for(target_text, text, "energy", "electrical field energy", "electric field energy") and "capacitor" in text and has("energy", "voltage", "voltage"):
        task_type, confidence = "capacitor_energy", 0.84
        reasons.append("capacitor energy voltage-scaling wording")
    elif "percentage loss" in text and has("energy", "energy"):
        task_type, confidence = "capacitor_energy", 0.78
        reasons.append("energy percentage-loss wording")
    elif "efficiency" in target_text and has("energy", "energy"):
        task_type, confidence = "electric_power", 0.78
        reasons.append("energy efficiency wording")
    elif _asks_for(target_text, text, "capacitance") and (has("charge", "voltage") or has("energy", "voltage")):
        task_type, confidence = "capacitance", 0.92
        reasons.append("capacitance wording with charge and voltage")
    elif "capacitor" in text and _asks_for(target_text, text, "voltage", "potential difference") and (has("charge", "capacitance") or has("energy", "capacitance")):
        task_type, confidence = "capacitor_final_voltage", 0.86
        reasons.append("capacitor voltage wording with stored charge/energy quantities")
    elif _asks_for(target_text, text, "maximum charge", "charge") and "capacitor" in text and "breakdown" in text and has("length", "electric_field"):
        task_type, confidence = "capacitor_charge", 0.84
        reasons.append("parallel-plate breakdown maximum-charge wording")
    elif _asks_for(target_text, text, "charge") and "capacitor" in text and has("capacitance", "voltage"):
        task_type, confidence = "capacitor_charge", 0.9
        reasons.append("capacitor charge wording with capacitance and voltage")
    elif _asks_for(target_text, text, "charge") and "capacitor" in text and has("energy", "voltage"):
        task_type, confidence = "capacitor_charge", 0.86
        reasons.append("capacitor charge wording with stored energy and voltage")
    elif _asks_for(target_text, text, "power") and (
        has("voltage", "current") or has("current", "resistance") or has("voltage", "resistance")
    ):
        task_type, confidence = "electric_power", 0.9
        reasons.append("power wording with compatible circuit quantities")
    elif _asks_for(target_text, text, "total power") and dims.count("power") >= 2:
        task_type, confidence = "electric_power", 0.86
        reasons.append("total circuit power wording with multiple component powers")
    elif _asks_for(target_text, text, "current", "i") and (
        has("power", "resistance") or has("power", "voltage")
    ):
        task_type, confidence = "ohm_law", 0.84
        reasons.append("current requested from deterministic DC power relation")
    elif _asks_for(target_text, text, "voltage", "potential difference", "u") and has("power", "current"):
        task_type, confidence = "ohm_law", 0.84
        reasons.append("voltage requested from deterministic DC power relation")
    elif _asks_for(target_text, text, "resistance", "r") and (
        has("power", "current") or has("voltage", "power")
    ):
        task_type, confidence = "ohm_law", 0.84
        reasons.append("resistance requested from deterministic DC power relation")
    elif ("rlc" in text or "lc circuit" in text or "resonan" in text or "resonat" in text or "f0" in text or "f_0" in text) and (
        _asks_for(target_text, text, "capacitance", "capacitor", "value of c", "calculate c", "determine c")
        or target_text.strip() == "c"
        or target_text.strip().startswith("c for")
    ) and has("inductance", "frequency"):
        task_type, confidence = "capacitance", 0.84
        reasons.append("LC/RLC resonance capacitance wording with L and f")
    elif ("rlc" in text or "lc circuit" in text or "resonan" in text or "resonat" in text or "f0" in text or "f_0" in text) and (
        _asks_for(target_text, text, "inductance", "inductor", "value of l", "calculate l", "determine l", "what l", "l is needed")
        or target_text.strip() == "l"
        or target_text.strip().startswith("l ")
    ) and has("capacitance", "frequency"):
        task_type, confidence = "inductance", 0.84
        reasons.append("LC/RLC resonance inductance wording with C and f")
    elif _asks_for(target_text, text, "inductance", "value of l") and has("energy", "current"):
        task_type, confidence = "inductance", 0.84
        reasons.append("inductor energy inverted for inductance")
    elif ("rlc" in text) and ("resonance" in text or "resonant" in text) and _asks_for(target_text, text, "resistance", "pure resistance") and has("resistance"):
        task_type, confidence = "ohm_law", 0.82
        reasons.append("RLC resonance pure-resistance equals impedance wording")
    elif ("rlc" in text) and _asks_for(target_text, text, "impedance", "total impedance", "z") and (
        has("resistance", "inductance", "capacitance", "frequency") or has("voltage", "current")
    ):
        task_type, confidence = "rlc_impedance", 0.86
        reasons.append("series-RLC impedance wording with circuit parameters")
    elif _is_rlc_frequency_shift_question(text, target_text) and (has("voltage", "resistance", "resistance", "resistance") or has("resistance", "current", "current") or has("voltage", "resistance", "resistance")):
        task_type, confidence = "ohm_law", 0.82
        reasons.append("RLC frequency-shift resonance wording")
    elif _is_rlc_quadrature_resistance_question(text, target_text) and has("voltage", "power", "resistance"):
        task_type, confidence = "ohm_law", 0.82
        reasons.append("RLC quadrature missing-resistance wording")
    elif _is_measurement_error_question(text, target_text):
        task_type, confidence = "measurement_error", 0.78
        reasons.append("measurement-error wording with explicit measured/true/uncertainty data")
    elif _asks_for(target_text, text, "current", "voltage", "resistance") and (
        has("voltage", "resistance") or has("current", "resistance") or has("voltage", "current")
    ):
        task_type, confidence = "ohm_law", 0.84
        reasons.append("ohm-law compatible quantities")
    elif _asks_for(target_text, text, "force") and "opposite sides" in text and "straight line" in text and has("charge", "charge", "length", "length"):
        task_type, confidence = "coulomb_force", 0.78
        reasons.append("collinear opposite-side Coulomb force wording")
    elif _asks_for(target_text, text, "force") and has("charge", "charge", "length"):
        task_type, confidence = "coulomb_force", 0.82
        reasons.append("force wording with two charges and distance")
    elif _asks_for(target_text, text, "force") and "midpoint" in text and "charge" in text and any(cue in text for cue in ["equal magnitude", "same magnitude", "q1 and q2"]):
        task_type, confidence = "coulomb_force", 0.76
        reasons.append("midpoint force symmetry wording")
    elif _asks_for(target_text, text, "distance") and has("charge", "charge", "length") and "electric field" in text and "zero" in text:
        task_type, confidence = "electric_field_point", 0.8
        reasons.append("zero-field point wording with two charges and separation")
    elif "electric field" in text and "zero" in text and "same sign" in text and has("length"):
        task_type, confidence = "electric_field_point", 0.76
        reasons.append("symbolic zero-field point wording with charge ratio and separation")
    elif (
        "electric field" in text
        and "zero" in text
        and "centroid" in text
        and "equilateral triangle" in text
        and _asks_for(target_text, text, "charge", "value")
        and has("charge")
    ):
        task_type, confidence = "electric_field_point", 0.76
        reasons.append("equilateral-centroid zero-field charge symmetry wording")
    elif "square" in text and "electric field" in text and "zero" in text and _asks_for(target_text, text, "charge", "q4") and has("charge"):
        task_type, confidence = "electric_field_point", 0.76
        reasons.append("square-center zero-field unknown charge symmetry wording")
    elif _asks_for(target_text, text, "force") and _is_right_isosceles_identical_charge_pattern(text) and has("charge", "length"):
        task_type, confidence = "coulomb_force", 0.78
        reasons.append("right-isosceles identical-charge force geometry wording")
    elif _asks_for(target_text, text, "force") and "charge" in text and "center" in text and (
        "identical" in text or "same magnitude" in text
    ):
        task_type, confidence = "coulomb_force", 0.76
        reasons.append("symmetric charge-center force wording")
    elif _asks_for(target_text, text, "force") and has("charge", "length") and "equilateral triangle" in text and (
        "q1 = q2 = q3" in text or "three identical charges" in text
    ):
        task_type, confidence = "coulomb_force", 0.8
        reasons.append("equilateral identical-charge Coulomb geometry wording")
    elif _asks_for(target_text, text, "electric field", "field strength") and has("force", "charge"):
        task_type, confidence = "electric_field_force", 0.88
        reasons.append("electric-field wording with force and charge")
    elif _asks_for(target_text, text, "force") and has("charge", "electric_field"):
        task_type, confidence = "force_in_electric_field", 0.88
        reasons.append("electric-force wording with charge and electric field")
    elif _is_square_diagonal_alternating_zero_field(text, target_text):
        task_type, confidence = "electric_field_point", 0.8
        reasons.append("square-center alternating diagonal charges cancel by symmetry")
    elif _is_square_adjacent_alternating_center_field(text, target_text):
        task_type, confidence = "electric_field_point", 0.78
        reasons.append("square-center adjacent alternating charges combine by symmetry")
    elif _asks_for(target_text, text, "voltage", "potential difference") and has("electric_field", "length"):
        task_type, confidence = "uniform_field_voltage", 0.86
        reasons.append("potential-difference wording with uniform field quantities")
    elif _asks_for(target_text, text, "potential", "voltage") and has("charge", "length") and "capacitor" not in text:
        task_type, confidence = "electric_potential_point", 0.78
        reasons.append("electric-potential wording with charge and distance")
    elif _asks_for(target_text, text, "energy") and has("charge", "charge", "length"):
        task_type, confidence = "electrostatic_energy", 0.8
        reasons.append("electrostatic energy wording with two charges and distance")
    elif _asks_for(target_text, text, "electric field", "field strength") and has("charge", "length"):
        task_type, confidence = "electric_field_point", 0.82
        reasons.append("electric-field wording with charge and distance")
    elif (_asks_for(target_text, text, "frequency") or "f0" in text or "f_0" in text) and has("inductance", "capacitance"):
        task_type, confidence = "lc_frequency", 0.86
        reasons.append("frequency/resonance wording with L and C")
    elif _asks_for(target_text, text, "frequency", "oscillation frequency") and has("time") and ("period" in text or "lc circuit" in text):
        task_type, confidence = "lc_frequency", 0.82
        reasons.append("frequency-from-period wording")
    elif _asks_for(target_text, text, "period") and has("inductance", "capacitance"):
        task_type, confidence = "lc_period", 0.86
        reasons.append("period wording with L and C")
    elif ("lc circuit" in text or "ideal lc" in text) and _asks_for(target_text, text, "total energy", "energy") and has("charge", "capacitance"):
        task_type, confidence = "capacitor_energy", 0.86
        reasons.append("LC total energy from maximum capacitor charge")
    elif ("lc circuit" in text or "ideal lc" in text) and _asks_for(target_text, text, "energy", "magnetic field energy", "electric field energy") and has("energy", "energy"):
        task_type, confidence = "lc_energy", 0.82
        reasons.append("ideal-LC complementary energy wording")
    elif _asks_for(target_text, text, "current") and ("inductor" in text or "coil" in text) and has("inductance", "energy"):
        task_type, confidence = "inductor_energy", 0.86
        reasons.append("inductor energy inverted for current")
    elif _asks_for(target_text, text, "energy") and ("inductor" in text or "magnetic field" in text or "coil" in text) and has("inductance", "current"):
        task_type, confidence = "inductor_energy", 0.9
        reasons.append("inductor energy wording with inductance and current")
    elif _asks_for(target_text, text, "inductive reactance") and has("frequency", "inductance"):
        task_type, confidence = "inductive_reactance", 0.9
        reasons.append("inductive-reactance wording with frequency and inductance")
    elif _asks_for(target_text, text, "capacitive reactance") and has("frequency", "capacitance"):
        task_type, confidence = "capacitive_reactance", 0.9
        reasons.append("capacitive-reactance wording with frequency and capacitance")
    elif _asks_for(target_text, text, "impedance") and has("resistance", "resistance", "resistance"):
        task_type, confidence = "rlc_impedance", 0.82 if has_symbol("r", "xl", "xc") else 0.68
        reasons.append("series-RLC impedance wording with resistance/reactance quantities")
    elif _asks_for(target_text, text, "power factor", "cos phi", "cosφ") and has("resistance", "resistance"):
        task_type, confidence = "power_factor", 0.82 if has_symbol("r", "z") else 0.68
        reasons.append("power-factor wording with R and Z quantities")
    elif "solenoid" in text and _asks_for(target_text, text, "turn density", "turns per meter", "turns per unit", "number of turns per") and has("count", "length"):
        task_type, confidence = "turn_density", 0.88
        reasons.append("solenoid turn-density wording with turns and length")
    elif "solenoid" in text and _asks_for(target_text, text, "inductance") and has("count", "area", "length"):
        task_type, confidence = "solenoid_inductance", 0.84
        reasons.append("air-core solenoid inductance wording with turns, area, and length")
    elif "solenoid" in text and _asks_for(target_text, text, "magnetic flux", "flux") and has("area", "turn_density", "current"):
        task_type, confidence = "magnetic_flux", 0.84
        reasons.append("solenoid one-turn magnetic-flux wording with area, turn density, and current")
    elif _asks_for(target_text, text, "magnetic field") and "solenoid" in text and (
        has("turn_density", "current") or has("count", "length", "current")
    ):
        task_type, confidence = "solenoid_magnetic_field", 0.9
        reasons.append("solenoid magnetic-field wording with turn density or turns/length and current")
    elif _asks_for(target_text, text, "magnetic flux", "flux") and has("magnetic_field", "area"):
        task_type, confidence = "magnetic_flux", 0.86
        reasons.append("magnetic-flux wording with magnetic field and area")
    elif answer_type == "symbolic" and "electric field" in target_text and "perpendicular bisector" in text:
        task_type, confidence = "electric_field_point", 0.78
        reasons.append("symbolic perpendicular-bisector electric-field geometry")
    elif concepts or answer_type in {"conceptual", "symbolic", "yes_no"}:
        task_type, confidence = "conceptual", 0.72
        reasons.append("concept or symbolic front signal")

    if task_type not in TASK_TYPES:
        task_type = "unknown"
    return RouteResult(task_type, answer_type, confidence, reasons)


def _asks_for(target_text: str, full_text: str, *keywords: str) -> bool:
    if target_text:
        return any(keyword in target_text for keyword in keywords)
    return any(keyword in full_text for keyword in keywords)


def _is_rlc_frequency_shift_question(text: str, target_text: str) -> bool:
    target_or_text = f"{target_text} {text}"
    return (
        any(
            cue in text
            for cue in [
                "frequency is doubled",
                "frequency doubles",
                "frequency is tripled",
                "frequency is quadrupled",
                "frequency quadrupled",
                "frequency is increased",
                "increased by a factor",
            ]
        )
        and any(cue in text for cue in ["xl", "xc", "resonance", "rlc"])
        and any(cue in target_or_text for cue in ["current", "voltage across r", "voltage across the resistor", "power", "zl", "inductive reactance"])
    )


def _is_rlc_quadrature_resistance_question(text: str, target_text: str) -> bool:
    return (
        ("lcω^2 = 1" in text or "lcω2 = 1" in text or "lcω² = 1" in text)
        and any(cue in text for cue in ["90 degrees out of phase", "90° out of phase", "quadrature"])
        and any(cue in target_text or cue in text for cue in ["r1", "r2", "value of r"])
    )


def _is_measurement_error_question(text: str, target_text: str) -> bool:
    haystack = f"{text} {target_text}"
    return any(
        cue in haystack
        for cue in [
            "least count",
            "uncertainty",
            "absolute error",
            "relative error",
            "relative uncertainty",
            "percentage relative",
            "random error",
            "mean absolute error",
            "actual value",
            "true value",
            "measured value",
            "student measured",
        ]
    )


def _is_right_isosceles_identical_charge_pattern(text: str) -> bool:
    return (
        ("isosceles right" in text or "right isosceles" in text or "right-angled triangle" in text)
        and any(cue in text for cue in ["three identical charges", "3 vertices", "three vertices"])
        and any(cue in text for cue in ["right angle vertex", "right-angle vertex", "right-angled vertex", "right angle"])
        and any(cue in text for cue in ["legs", "equal sides", "leg length", "leg lengths", "sides of length", "side length", "sides of"])
    )


def _is_square_diagonal_alternating_zero_field(text: str, target_text: str) -> bool:
    haystack = f"{text} {target_text}"
    return (
        "square" in haystack
        and "diagonal" in haystack
        and ("electric field" in haystack or "field strength" in haystack)
        and ("same magnitude" in haystack or "magnitude q" in haystack or "equal magnitude" in haystack)
        and ("positive charges" in haystack or "positive charge" in haystack)
        and ("negative charges" in haystack or "negative charge" in haystack)
        and (
            ("a and c" in haystack and "b and d" in haystack)
            or ("a, c" in haystack and "b, d" in haystack)
        )
    )


def _is_square_adjacent_alternating_center_field(text: str, target_text: str) -> bool:
    haystack = f"{text} {target_text}"
    return (
        "square" in haystack
        and "diagonal" in haystack
        and ("intersection" in haystack or "center" in haystack)
        and ("electric field" in haystack or "field strength" in haystack)
        and ("same magnitude" in haystack or "magnitude q" in haystack or "equal magnitude" in haystack)
        and ("positive charges" in haystack or "positive charge" in haystack)
        and ("negative charges" in haystack or "negative charge" in haystack)
        and (
            ("a and d" in haystack and "b and c" in haystack)
            or ("a, d" in haystack and "b, c" in haystack)
        )
    )
