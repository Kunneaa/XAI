import csv
import unittest
from collections import Counter
from pathlib import Path

from xai_pipeline.cache import clear_verified_response_cache, get_verified_response, put_verified_response
from xai_pipeline.pipeline import process_question
from xai_pipeline.verifier import validate_plan


DATA_PATH = Path(__file__).resolve().parents[1] / "Physics_Problems_Text_Only.csv"


def load_rows():
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class DatasetPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clear_verified_response_cache()
        cls.rows = load_rows()
        cls.results = []
        cls.crashes = []
        for row in cls.rows:
            try:
                cls.results.append((row, process_question(row["question"])))
            except Exception as exc:
                cls.crashes.append((row.get("id"), repr(exc)))

    def test_full_pipeline_boundary_runs_on_all_rows(self):
        self.assertEqual(len(self.rows), 1350)
        self.assertEqual(self.crashes, [])
        self.assertEqual(len(self.results), len(self.rows))

    def test_solver_and_verifier_state_are_consistent(self):
        inconsistent = []
        for row, result in self.results:
            if result["verifier"]["ok"] != result["solver"]["solved"]:
                inconsistent.append(row["id"])
            if result["verifier"]["ok"] and result["answer"] == "Uncertain":
                inconsistent.append(row["id"])
            if not result["verifier"]["ok"] and result["answer"] != "Uncertain":
                inconsistent.append(row["id"])
        self.assertEqual(inconsistent, [])

    def test_core_tail_boundaries_are_present_and_consistent(self):
        required_keys = {
            "answer",
            "explanation",
            "cot",
            "premises",
            "metadata",
            "schema_registry_validator",
            "unit_conversion",
            "solver",
            "verifier",
            "answer_checker",
            "cache",
            "trace",
            "telemetry",
            "polish",
        }
        expected_tail = [
            "schema_registry_validator",
            "unit_converter",
            "principle_selector",
            "geometry_template_matcher",
            "deterministic_executor",
            "verifier",
            "trace_explanation",
            "answer_checker",
            "cache_store_if_verified",
            "response",
        ]
        bad = []
        for row, result in self.results:
            if not required_keys <= set(result):
                bad.append((row["id"], "missing_keys"))
                continue
            if not isinstance(result["cot"], list) or not isinstance(result["premises"], list):
                bad.append((row["id"], "bad_public_reasoning_shape"))
            if not isinstance(result["metadata"], dict):
                bad.append((row["id"], "bad_metadata_shape"))
            if "deadline" not in result["trace"]:
                bad.append((row["id"], "missing_deadline_trace"))
            if "target_unit_policy" not in result["metadata"]:
                bad.append((row["id"], "missing_target_unit_policy"))
            if not result.get("telemetry") or "event_type" not in result["telemetry"]:
                bad.append((row["id"], "missing_telemetry"))
            if result["verifier"]["ok"] and result.get("polish") is None:
                bad.append((row["id"], "missing_polish_boundary"))
            stages = result["trace"]["stages"]
            if any(stage not in stages for stage in expected_tail):
                bad.append((row["id"], "missing_trace_stage"))
            if result["verifier"]["ok"] and not result["answer_checker"]["ok"]:
                bad.append((row["id"], "verified_but_answer_check_failed"))
            if result["verifier"]["ok"] and not result["schema_registry_validator"]["ok"]:
                bad.append((row["id"], "verified_but_registry_invalid"))
            if result["verifier"]["ok"] and not result["unit_conversion"]["ok"]:
                bad.append((row["id"], "verified_but_unit_conversion_invalid"))
        self.assertEqual(bad, [])

    def test_pipeline_distribution_matches_current_fast_solver_scope(self):
        solved_counts = Counter("solved" if result["verifier"]["ok"] else "unsolved" for _, result in self.results)
        routes = Counter(result["route"]["task_type"] for _, result in self.results)
        unsolved_reasons = Counter(
            result["solver"]["trace"].get("reason")
            for _, result in self.results
            if not result["verifier"]["ok"]
        )
        planner_reasons = Counter(result["planner"]["reason"] for _, result in self.results)

        self.assertEqual(dict(solved_counts), {"solved": 1193, "unsolved": 157})
        self.assertEqual(
            dict(sorted(routes.items())),
            {
                "capacitance": 101,
                "capacitive_reactance": 3,
                "capacitor_charge": 47,
                "capacitor_energy": 91,
                "capacitor_final_voltage": 31,
                "charged_particle_motion": 1,
                "conceptual": 236,
                "coulomb_force": 188,
                "dielectric_constant": 5,
                "electric_field_point": 197,
                "electric_power": 59,
                "equal_charge_coulomb": 1,
                "faraday_induction": 10,
                "force_in_electric_field": 3,
                "inductance": 38,
                "inductive_reactance": 3,
                "inductor_energy": 39,
                "lc_energy": 4,
                "lc_frequency": 32,
                "lc_period": 2,
                "magnetic_flux": 9,
                "measurement_error": 58,
                "multi_output": 7,
                "ohm_law": 82,
                "power_factor": 8,
                "resultant_force": 60,
                "rlc_impedance": 12,
                "solenoid_inductance": 1,
                "solenoid_magnetic_field": 14,
                "turn_density": 7,
                "unknown": 1,
            },
        )
        self.assertEqual(
            dict(sorted(unsolved_reasons.items())),
            {
                "coulomb_geometry_or_multi_body_not_fast_path": 6,
                "electric_field_geometry_not_fast_path": 2,
                "no_conceptual_rule": 148,
                "no_formula_for_route": 1,
            },
        )
        self.assertEqual(dict(planner_reasons), {"solver_already_solved": 1193, "llm_disabled": 157})

    def test_retrieval_is_used_only_for_unverified_questions_when_enabled(self):
        clear_verified_response_cache()
        checked = []
        for row in self.rows[:120]:
            result = process_question(row["question"], data_path=DATA_PATH)
            checked.append((result["verifier"]["ok"], len(result["retrieval"])))

        bad = [
            (ok, retrieval_count)
            for ok, retrieval_count in checked
            if (ok and retrieval_count != 0) or ((not ok) and retrieval_count == 0)
        ]
        self.assertEqual(bad, [])

    def test_retrieval_metadata_never_leaks_answers_or_cot(self):
        clear_verified_response_cache()
        unsafe_keys = {"answer", "cot", "final_numeric", "final_answer", "unit"}
        bad = []
        for row in self.rows[:120]:
            result = process_question(row["question"], data_path=DATA_PATH)
            for hit in result["retrieval"]:
                metadata = hit["task_metadata"]
                if unsafe_keys & set(metadata):
                    bad.append((row["id"], hit["problem_id"], sorted(unsafe_keys & set(metadata))))
                if not metadata.get("safe_fields_only"):
                    bad.append((row["id"], hit["problem_id"], "missing_safe_marker"))
        self.assertEqual(bad, [])

    def test_plan_validator_rejects_unowned_ids_and_numeric_answers(self):
        plan = {
            "task_type": "ohm_law",
            "answer_type": "numeric",
            "given": [],
            "targets": [{"symbol": "I"}],
            "formula_ids": ["capacitor_energy_voltage"],
            "principle_ids": ["dc_circuit_core"],
            "implicit_rule_ids": ["invented_rule"],
            "numeric_answer": 42,
        }
        result = validate_plan(plan, {})
        self.assertFalse(result.ok)
        self.assertIn("formula_task_mismatch:capacitor_energy_voltage:ohm_law", result.issues)
        self.assertIn("unknown_implicit_rule_id:invented_rule", result.issues)
        self.assertIn("planner_supplied_numeric_answer", result.issues)

        bad_schema_plan = {
            "status": "invented",
            "task_type": "ohm_law",
            "answer_type": "numeric",
            "targets": [],
            "formula_ids": [],
            "principle_ids": [],
            "geometry_template_ids": ["freehand_diagram"],
            "implicit_rule_ids": [],
            "solve_strategy": "freeform",
            "numeric_answer": None,
            "conceptual_answer": "Use Ohm law.",
        }
        schema_result = validate_plan(bad_schema_plan, {})
        self.assertFalse(schema_result.ok)
        self.assertIn("unknown_status", schema_result.issues)
        self.assertIn("unknown_solve_strategy", schema_result.issues)
        self.assertIn("empty_targets", schema_result.issues)
        self.assertIn("unknown_geometry_template_id:freehand_diagram", schema_result.issues)
        self.assertIn("numeric_task_with_conceptual_answer", schema_result.issues)

    def test_new_deterministic_executors_solve_only_whitelisted_patterns(self):
        by_id = {row["id"]: result for row, result in self.results}
        expected = {
            "LD001": "coulomb_force_triangle_sides",
            "LD003": "coulomb_force_triangle_sides",
            "LD006": "coulomb_force",
            "LD008": "resultant_two_forces",
            "LD009": "resultant_two_forces",
            "LD014": "coulomb_equal_charge",
            "LD017": "coulomb_force",
            "LD020": "resultant_two_forces_angle",
            "LD021": "coulomb_force",
            "LD029": "coulomb_force_triangle_sides",
            "LD031": "coulomb_force",
            "LD034": "coulomb_force",
            "LD039": "symmetric_zero_force",
            "LD042": "symmetric_zero_force",
            "LD043": "symmetric_zero_force",
            "LD060": "electric_field_point",
            "LD065": "electric_field_two_charge_isosceles",
            "LD051": "electric_field_two_charge_triangle_sides",
            "LD052": "electric_field_two_charge_triangle_sides",
            "LD053": "electric_field_two_charge_triangle_sides",
            "LD058": "electric_field_two_charge_triangle_sides",
            "LD059": "electric_field_square_diagonal_alternating_zero",
            "LD099": "electric_field_two_charge_isosceles",
            "LD108": "coulomb_force_triangle_sides",
            "LD123": "coulomb_force",
            "LD142": "coulomb_force",
            "LD150": "coulomb_force",
            "LD152": "coulomb_force",
            "DT056": "electric_field_equilateral_vertex",
            "DT053": "electric_field_equilateral_vertex",
            "DT060": "electric_equilibrium_deflection_angle",
            "DT072": "electric_field_ring_axis",
            "DT073": "electric_field_finite_rod_perpendicular_end",
            "DT074": "electric_field_parallel_sheets",
            "DT075": "electric_field_parallel_sheets",
            "DT083": "electric_field_disk_axis",
            "DT089": "electric_field_conducting_plate",
            "DT090": "electric_field_infinite_line",
            "DT091": "electric_field_semicircular_arc_center",
            "DT092": "electric_field_point",
            "DT093": "electric_field_point",
            "DT007": "electric_field_two_charge_isosceles",
            "DT008": "electric_field_two_charge_isosceles",
            "DT019": "electric_field_square_diagonal_alternating_zero",
            "DT020": "electric_field_square_adjacent_alternating_center",
            "DT025": "electric_field_zero_line_two_charges",
            "DT027": "electric_field_zero_line_two_charges",
            "DT028": "electric_field_zero_line_two_charges",
            "DT044": "point_charge_from_field_dielectric",
            "DT059": "electric_field_equilibrium_mg",
            "DT087": "electric_field_point_dielectric",
            "LD206": "coulomb_perpendicular_bisector_opposite_charges",
            "TD003": "capacitor_energy_voltage",
            "TD002": "capacitor_connected_voltage_constant",
            "TD368": "capacitor_charge",
            "TD376": "multi_output_direct",
            "TD374": "multi_output_direct",
            "TD388": "capacitor_energy_shared_identical",
            "TD400": "capacitor_energy_shared_identical",
            "TD387": "capacitor_series_unknown_from_final_charge",
            "TD390": "capacitor_series_field",
            "NL091": "capacitor_energy_voltage_percent",
            "NL092": "energy_loss_percent",
            "NL326": "energy_efficiency",
            "NL022": "capacitor_energy_voltage_scaled",
            "THCB066": "multi_output_direct",
            "THCB088": "measurement_error_direct",
            "THCB084": "ohm_current_power_voltage",
            "DDT340": "multi_output_direct",
            "TD014": "capacitor_charge_sharing_voltage",
            "TD094": "parallel_plate_dielectric_constant",
            "DT043": "charged_particle_stopping_distance_uniform_field",
            "DDT150": "faraday_flux_emf",
            "DDT160": "faraday_flux_emf",
            "DDT376": "faraday_flux_emf",
            "DDT322": "rlc_current_impedance",
            "CH041": "rlc_power_resonance",
            "CH221": "rlc_quadrature_current",
            "CH236": "rlc_quadrature_segment_voltage",
            "CH246": "rlc_quadrature_power_factor",
            "CH251": "rlc_frequency_resonance_resistor_voltage",
            "CH263": "rlc_frequency_resonance_resistor_voltage",
            "CH275": "rlc_frequency_resonance_power",
            "TD012": "parallel_plate_breakdown_charge",
            "TD015": "capacitor_series_voltage",
            "TD016": "capacitor_voltage_charge",
            "TD361": "capacitor_energy_charge_voltage",
            "TD362": "capacitance_from_energy_voltage",
            "TD372": "capacitor_voltage_energy",
            "LD122": "coulomb_right_isosceles_identical_vertex",
            "CH242": "rlc_quadrature_segment_power_same_voltage",
            "CH103": "rlc_initial_reactance_from_doubled_frequency_current",
            "CH365": "rlc_resonance_inductor_voltage",
            "CH066": "lc_resonance_inductance",
            "CH350": "lc_resonance_capacitance",
            "CHLT017": "yes_no_direct",
            "NL320": "conceptual_direct",
            "NL346": "capacitor_charge_energy_voltage",
            "LD345": "electric_field_two_charge_triangle_sides",
            "LD349": "electric_field_two_charge_angle",
        }
        bad = []
        for problem_id, formula_id in expected.items():
            result = by_id[problem_id]
            if not result["verifier"]["ok"] or result["solver"]["formula_id"] != formula_id:
                bad.append((problem_id, result["solver"]["formula_id"], result["verifier"]["issues"]))
        self.assertEqual(bad, [])

        guarded = {}
        guarded_bad = []
        for problem_id, reason in guarded.items():
            result = by_id[problem_id]
            if result["verifier"]["ok"] or result["solver"]["trace"].get("reason") != reason:
                guarded_bad.append((problem_id, result["solver"]["trace"].get("reason"), result["answer"]))
        self.assertEqual(guarded_bad, [])

    def test_extended_domain_smoke_cases_cover_router_and_solver_families(self):
        cases = [
            (
                "A series RLC circuit has resistance 10 ohm, inductive reactance 30 ohm, and capacitive reactance 20 ohm. Calculate impedance.",
                "rlc_impedance",
                "rlc_impedance",
            ),
            (
                "A solenoid has 500 turns, length 0.5 m, and current 2 A. Calculate magnetic field.",
                "solenoid_magnetic_field",
                "solenoid_magnetic_field_turns_length",
            ),
            (
                "An ideal transformer has primary voltage 220 V, primary turns 1100 and secondary turns 100. Calculate secondary voltage.",
                "transformer",
                "ideal_transformer_voltage_ratio",
            ),
            (
                "A wire has carrier density n = 8e28 m^-3, charge q = 1.6e-19 C, area A = 1e-6 m2 and drift speed 0.01 m/s. Calculate current.",
                "drift_current",
                "drift_current",
            ),
            (
                "A balanced Wheatstone bridge has R1 = 2 ohm, R2 = 4 ohm, R3 = 3 ohm. Find R4 resistance.",
                "wheatstone_bridge",
                "wheatstone_balance_resistance",
            ),
            (
                "Yes or no: Does an ideal voltmeter have infinite resistance and draw no current?",
                "conceptual",
                "yes_no_direct",
            ),
            (
                "A parallel-plate air capacitor has an area of each plate of 33.2 cm2 and the distance between the two plates is 1.86 mm. Calculate the capacitance of the capacitor.",
                "capacitance",
                "parallel_plate_capacitance",
            ),
            (
                "An RLC circuit has a resistance R = 20 ohm, an inductance L = 0.5 H, a capacitance C = 100 uF, and a frequency f = 50 Hz. Calculate the total impedance Z of the circuit.",
                "rlc_impedance",
                "rlc_impedance_from_rlcf",
            ),
            (
                "Given an inductor L = 0.5 H, what capacitance C is needed to achieve resonance at a frequency of 60 Hz?",
                "capacitance",
                "lc_resonance_capacitance",
            ),
            (
                "Given a capacitor with C = 50 uF, what inductance L is required to achieve resonance at 200 Hz?",
                "inductance",
                "lc_resonance_inductance",
            ),
            (
                "The electric field strength produced by a point charge at point A is 36 V/m, and at point B is 9 V/m. Points A and B lie on the same electric field line. What is the electric field strength at point C, the midpoint of AB?",
                "electric_field_point",
                "point_charge_field_midpoint_from_two_fields",
            ),
            (
                "A point at a fixed distance from a charge in air has an electric field strength of 4000 V/m. If a dielectric material with a dielectric constant of 2 now completely surrounds the point charge and the point under consideration, what will be the magnitude of the electric field strength at that point?",
                "electric_field_point",
                "dielectric_field_scaled",
            ),
            (
                "Three identical charges q = 2e-7 C are placed at the three vertices of an isosceles right triangle with legs of 15 cm. Calculate the net electric field strength at the right-angle vertex.",
                "electric_field_point",
                "electric_field_right_isosceles_identical_vertex",
            ),
            (
                "Two electric charges q1 = -3e-8 C and q2 = 8e-8 C are placed at two points A and B, 15 cm apart, in air. A charge q3 = -2e-8 C is placed at point C, given that the distance from C to A is 7 cm and to B is 9 cm. Calculate the force acting on q3.",
                "coulomb_force",
                "coulomb_force_triangle_sides",
            ),
            (
                "Two point charges q1 = +2e-6 C and q2 = -2e-6 C are placed at A and B separated by 6 cm. Point M lies on the line connecting A and B, 2 cm from A. Calculate the net electric field at M.",
                "electric_field_point",
                "electric_field_point",
            ),
            (
                "Two point charges q1 = 4e-6 C and q2 = 2e-6 C are separated by 8 cm. Point M lies on the perpendicular bisector of AB, 3 cm from the midpoint. Calculate the resultant electric field at M.",
                "electric_field_point",
                "electric_field_two_charge_isosceles",
            ),
            (
                "Two charges q1 = +2e-6 C and q2 = +2e-6 C are at the ends of a 10 cm line segment. A third charge q3 = -1e-6 C lies on the line connecting q1 and q2, 4 cm from q1. Calculate the net force on q3.",
                "coulomb_force",
                "coulomb_force",
            ),
            (
                "What is the shape of the graph of magnetic field energy versus current I?",
                "conceptual",
                "conceptual_direct",
            ),
            (
                "In an ideal LC circuit, when the magnetic energy is half of the total energy, what is the electric energy?",
                "conceptual",
                "conceptual_direct",
            ),
            (
                "If you double the number of turns of a solenoid, but keep its length and current the same, how does the magnetic field change?",
                "conceptual",
                "conceptual_direct",
            ),
        ]
        bad = []
        for question, task_type, formula_id in cases:
            result = process_question(question)
            if not result["verifier"]["ok"] or result["route"]["task_type"] != task_type or result["solver"]["formula_id"] != formula_id:
                bad.append((task_type, formula_id, result["route"], result["solver"], result["verifier"]["issues"]))
        self.assertEqual(bad, [])

    def test_cache_only_replays_verified_answers(self):
        clear_verified_response_cache()
        solved_question = None
        for row in self.rows:
            if process_question(row["question"])["verifier"]["ok"]:
                solved_question = row["question"]
                break
        clear_verified_response_cache()
        self.assertIsNotNone(solved_question)
        first = process_question(solved_question)
        second = process_question(solved_question)
        self.assertTrue(first["verifier"]["ok"])
        self.assertFalse(first["cache"]["hit"])
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(second["answer"], first["answer"])

        clear_verified_response_cache()
        unsolved_question = None
        for row in self.rows:
            if not process_question(row["question"])["verifier"]["ok"]:
                unsolved_question = row["question"]
                break
        clear_verified_response_cache()
        self.assertIsNotNone(unsolved_question)
        first_unsolved = process_question(unsolved_question)
        second_unsolved = process_question(unsolved_question)
        self.assertFalse(first_unsolved["verifier"]["ok"])
        self.assertFalse(first_unsolved["cache"]["hit"])
        self.assertFalse(second_unsolved["cache"]["hit"])

        clear_verified_response_cache()
        put_verified_response(
            "A guarded cache entry",
            {"answer": "1 A", "verifier": {"ok": True}, "answer_checker": {"ok": False}},
        )
        self.assertIsNone(get_verified_response("A guarded cache entry"))

    def test_cache_key_uses_canonical_raw_question_and_deadline_fallback_is_controlled(self):
        clear_verified_response_cache()
        question = next(row["question"] for row in self.rows if process_question(row["question"])["verifier"]["ok"])
        clear_verified_response_cache()
        first = process_question(question)
        spaced = "  ".join(question.split())
        second = process_question(f"  {spaced}  ")
        self.assertTrue(first["verifier"]["ok"])
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(second["answer"], first["answer"])

        clear_verified_response_cache()
        timed_out = process_question(question, timeout_seconds=0.0)
        self.assertEqual(timed_out["answer"], "Uncertain")
        self.assertEqual(timed_out["confidence"], 0.0)
        self.assertTrue(timed_out["metadata"]["timeout"])
        self.assertIn("deadline_expired", timed_out["verifier"]["issues"])


if __name__ == "__main__":
    unittest.main()
