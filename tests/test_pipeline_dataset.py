import csv
import unittest
from collections import Counter
from pathlib import Path

from xai_pipeline.core.pipeline import process_question
from xai_pipeline.runtime.cache import clear_verified_response_cache, get_verified_response, put_verified_response
from xai_pipeline.verification.verifier import validate_plan


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

    def test_core_pipeline_runs_on_all_rows(self):
        self.assertGreater(len(self.rows), 0)
        self.assertEqual(self.crashes, [])
        self.assertEqual(len(self.results), len(self.rows))

    def test_solver_and_verifier_state_are_consistent(self):
        inconsistent = []
        for row, result in self.results:
            if result["verifier"]["ok"] != result["solver"]["solved"]:
                inconsistent.append((row["id"], "solver_verifier_mismatch"))
            if result["verifier"]["ok"] and result["answer"] == "Uncertain":
                inconsistent.append((row["id"], "verified_uncertain"))
            if not result["verifier"]["ok"] and result["answer"] != "Uncertain":
                inconsistent.append((row["id"], "unverified_answered"))
            if result["verifier"]["ok"] and not result["answer_checker"]["ok"]:
                inconsistent.append((row["id"], "answer_check_failed"))
        self.assertEqual(inconsistent, [])

    def test_core_response_contract_is_present(self):
        required_keys = {
            "answer",
            "explanation",
            "cot",
            "premises",
            "metadata",
            "front",
            "route",
            "constraint_graph",
            "solver",
            "verifier",
            "answer_checker",
            "cache",
            "trace",
            "telemetry",
        }
        expected_stages = {
            "semantic_parser",
            "logic_engine",
            "constraint_graph",
            "equation_engine",
            "spatial_engine",
            "verifier",
            "explanation",
            "answer_check",
            "response",
        }
        bad = []
        for row, result in self.results:
            if not required_keys <= set(result):
                bad.append((row["id"], "missing_keys"))
                continue
            if not isinstance(result["cot"], list) or not isinstance(result["premises"], list):
                bad.append((row["id"], "bad_public_reasoning_shape"))
            if not isinstance(result["metadata"], dict):
                bad.append((row["id"], "bad_metadata_shape"))
            if not expected_stages <= set(result["trace"]["stages"]):
                bad.append((row["id"], "missing_trace_stage"))
            if "proof_dag" not in result["trace"]:
                bad.append((row["id"], "missing_proof_dag"))
            if result["verifier"]["ok"] and not result["trace"]["proof_dag"].get("nodes"):
                bad.append((row["id"], "missing_proof_nodes"))
            if "versions" not in result["metadata"]:
                bad.append((row["id"], "missing_versions"))
            if "route_task_type" not in result["telemetry"]:
                bad.append((row["id"], "missing_telemetry"))
        self.assertEqual(bad, [])

    def test_pipeline_distribution_matches_core_scope(self):
        solved_counts = Counter("solved" if result["verifier"]["ok"] else "unsolved" for _, result in self.results)
        routes = Counter(result["route"]["task_type"] for _, result in self.results)
        unsolved_reasons = Counter(
            result["solver"]["trace"].get("reason")
            for _, result in self.results
            if not result["verifier"]["ok"]
        )

        self.assertGreaterEqual(solved_counts["solved"], int(0.95 * len(self.rows)))
        for task_type in ["ohm_law", "capacitor_energy", "coulomb_force", "electric_field_point", "rlc_impedance"]:
            self.assertGreater(routes[task_type], 0)
        if solved_counts["unsolved"]:
            self.assertGreater(sum(unsolved_reasons.values()), 0)

    def test_plan_validator_rejects_unowned_ids_and_numeric_answers(self):
        plan = {
            "status": "ok",
            "task_type": "ohm_law",
            "answer_type": "numeric",
            "targets": [{"symbol": "I"}],
            "formula_ids": ["capacitor_energy_voltage"],
            "principle_ids": ["dc_circuit_core"],
            "geometry_template_ids": ["freehand_diagram"],
            "implicit_rule_ids": ["invented_rule"],
            "solve_strategy": "freeform",
            "numeric_answer": 42,
            "conceptual_answer": "Use Ohm law.",
        }
        result = validate_plan(plan, {})
        self.assertFalse(result.ok)
        self.assertIn("formula_task_mismatch:capacitor_energy_voltage:ohm_law", result.issues)
        self.assertIn("unknown_implicit_rule_id:invented_rule", result.issues)
        self.assertIn("proposal_supplied_numeric_answer", result.issues)
        self.assertIn("unknown_geometry_template_id:freehand_diagram", result.issues)

    def test_registry_engine_solves_general_formula_cases_without_problem_ids(self):
        cases = [
            ("A resistor R = 10 ohm has voltage U = 20 V. Find current.", "ohm_law", "ohm_current"),
            ("A capacitor C = 100 uF has voltage U = 30 V. Find energy.", "capacitor_energy", "capacitor_energy_voltage"),
            ("Two charges q1 = 2e-6 C and q2 = 3e-6 C are separated by r = 10 cm. Calculate force.", "coulomb_force", "coulomb_force"),
            ("A point charge q = 2 nC is in air. Find electric field at r = 3 cm.", "electric_field_point", "electric_field_point"),
            ("An inductor L = 0.5 H carries current I = 2 A. Find energy.", "inductor_energy", "inductor_energy"),
        ]
        bad = []
        for question, task_type, formula_id in cases:
            result = process_question(question)
            if not result["verifier"]["ok"] or result["route"]["task_type"] != task_type or result["solver"]["formula_id"] != formula_id:
                bad.append((question, result["route"], result["solver"], result["verifier"]["issues"]))
        self.assertEqual(bad, [])

    def test_extended_domain_smoke_cases_cover_core_families(self):
        cases = [
            ("A series RLC circuit has resistance 10 ohm, inductive reactance 30 ohm, and capacitive reactance 20 ohm. Calculate impedance.", "rlc_impedance", "rlc_impedance"),
            ("A solenoid has 500 turns, length 0.5 m, and current 2 A. Calculate magnetic field.", "solenoid_magnetic_field", "solenoid_magnetic_field_turns_length"),
            ("An ideal transformer has primary voltage 220 V, primary turns 1100 and secondary turns 100. Calculate secondary voltage.", "transformer", "ideal_transformer_voltage_ratio"),
            ("A wire has carrier density n = 8e28 m^-3, charge q = 1.6e-19 C, area A = 1e-6 m2 and drift speed 0.01 m/s. Calculate current.", "drift_current", "drift_current"),
            ("A balanced Wheatstone bridge has R1 = 2 ohm, R2 = 4 ohm, R3 = 3 ohm. Find R4 resistance.", "wheatstone_bridge", "wheatstone_balance_resistance"),
            ("A parallel-plate air capacitor has an area of each plate of 33.2 cm2 and the distance between the two plates is 1.86 mm. Calculate the capacitance of the capacitor.", "capacitance", "parallel_plate_capacitance"),
            ("An RLC circuit has a resistance R = 20 ohm, an inductance L = 0.5 H, a capacitance C = 100 uF, and a frequency f = 50 Hz. Calculate the total impedance Z of the circuit.", "rlc_impedance", "rlc_impedance_from_rlcf"),
            ("Given an inductor L = 0.5 H, what capacitance C is needed to achieve resonance at a frequency of 60 Hz?", "capacitance", "lc_resonance_capacitance"),
            ("Given a capacitor with C = 50 uF, what inductance L is required to achieve resonance at 200 Hz?", "inductance", "lc_resonance_inductance"),
        ]
        bad = []
        for question, task_type, formula_id in cases:
            result = process_question(question)
            if not result["verifier"]["ok"] or result["route"]["task_type"] != task_type or result["solver"]["formula_id"] != formula_id:
                bad.append((task_type, formula_id, result["route"], result["solver"], result["verifier"]["issues"]))
        self.assertEqual(bad, [])

    def test_cache_only_replays_verified_answers(self):
        clear_verified_response_cache()
        solved_question = next(row["question"] for row in self.rows if process_question(row["question"])["verifier"]["ok"])
        clear_verified_response_cache()
        first = process_question(solved_question)
        second = process_question(solved_question)
        self.assertTrue(first["verifier"]["ok"])
        self.assertFalse(first["cache"]["hit"])
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(second["answer"], first["answer"])

        clear_verified_response_cache()
        unsolved_question = "What is the favorite color of an ideal resistor?"
        clear_verified_response_cache()
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
        self.assertEqual(timed_out["metadata"]["status"], "timeout")
        self.assertIn("deadline_expired", timed_out["verifier"]["issues"])


if __name__ == "__main__":
    unittest.main()
