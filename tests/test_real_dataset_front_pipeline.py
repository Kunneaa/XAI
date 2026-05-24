import csv
import unittest
from collections import Counter
from pathlib import Path

from xai_pipeline import process_question_front
from xai_pipeline.implicit_kb import allowed_implicit_rule_ids


DATA_PATH = Path(__file__).resolve().parents[1] / "Physics_Problems_Text_Only.csv"
ANSWER_TYPE_IDS = {"numeric", "symbolic", "conceptual", "yes_no", "multi_output", "unknown"}


def load_rows():
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class RealDatasetFrontPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_rows()
        cls.payloads = []
        cls.crashes = []
        for row in cls.rows:
            try:
                cls.payloads.append((row, process_question_front(row["question"])))
            except Exception as exc:
                cls.crashes.append((row.get("id"), repr(exc)))

    def test_processes_every_real_row_without_crashing(self):
        self.assertEqual(len(self.rows), 1350)
        self.assertEqual(self.crashes, [])
        self.assertEqual(len(self.payloads), len(self.rows))

    def test_every_payload_preserves_raw_question_and_stage_trace(self):
        for row, payload in self.payloads:
            with self.subTest(problem_id=row["id"]):
                self.assertEqual(payload["raw_question"], row["question"])
                self.assertEqual(payload["trace"]["stages"], ["normalize", "implicit_kb"])
                self.assertFalse(payload["trace"]["normalize"]["llm_used"])
                self.assertFalse(payload["trace"]["implicit_kb"]["llm_used"])
                self.assertEqual(payload["answer_type_hint"], payload["trace"]["normalize"]["answer_type_hint"])
                self.assertIn(payload["answer_type_hint"], ANSWER_TYPE_IDS)

    def test_every_row_has_some_deterministic_front_signal(self):
        unresolved = []
        for row, payload in self.payloads:
            has_front_signal = any(
                [
                    payload["quantities"],
                    payload["symbolic_quantities"],
                    payload["symbolic_relations"],
                    payload["numeric_constants"],
                    payload["concepts"],
                    payload["target_hints"],
                    payload["implicit_facts"],
                ]
            )
            if not has_front_signal:
                unresolved.append(row["id"])
        self.assertEqual(unresolved, [])

    def test_no_warnings_remain_across_real_dataset(self):
        warnings = Counter(
            warning
            for _, payload in self.payloads
            for warning in payload["warnings"]
        )
        self.assertEqual(warnings, Counter())

    def test_all_implicit_rules_are_from_allowlist(self):
        allowed = set(allowed_implicit_rule_ids())
        unknown = []
        for row, payload in self.payloads:
            applied = set(payload["trace"]["implicit_kb"]["rules_applied"])
            if not applied <= allowed:
                unknown.append((row["id"], sorted(applied - allowed)))
        self.assertEqual(unknown, [])

    def test_all_extracted_spans_are_valid_and_auditable(self):
        bad_spans = []
        bad_raw_text = []
        collection_names = [
            "quantities",
            "symbolic_quantities",
            "symbolic_relations",
            "numeric_constants",
            "implicit_facts",
        ]
        for row, payload in self.payloads:
            text = payload["canonical_question"]
            for collection_name in collection_names:
                for item in payload[collection_name]:
                    span = item.get("span")
                    if span is None:
                        continue
                    start, end = span
                    if not (0 <= start <= end <= len(text)):
                        bad_spans.append((row["id"], collection_name, span))
                        continue
                    expected = item.get("raw_text") or item.get("trigger_text")
                    if expected is not None and text[start:end] != expected:
                        bad_raw_text.append((row["id"], collection_name, expected, text[start:end]))
        self.assertEqual(bad_spans, [])
        self.assertEqual(bad_raw_text, [])

    def test_dataset_wide_extraction_distribution_matches_current_data(self):
        numeric_counts = Counter(len(payload["quantities"]) for _, payload in self.payloads)
        symbolic_counts = Counter(len(payload["symbolic_quantities"]) for _, payload in self.payloads)
        relation_counts = Counter(len(payload["symbolic_relations"]) for _, payload in self.payloads)
        constant_counts = Counter(len(payload["numeric_constants"]) for _, payload in self.payloads)

        self.assertEqual(dict(sorted(numeric_counts.items())), {0: 86, 1: 82, 2: 649, 3: 227, 4: 213, 5: 58, 6: 35})
        self.assertEqual(dict(sorted(symbolic_counts.items())), {0: 1060, 1: 174, 2: 44, 3: 46, 4: 11, 5: 3, 6: 8, 7: 3, 9: 1})
        self.assertEqual(dict(sorted(relation_counts.items())), {0: 1217, 1: 112, 2: 11, 3: 10})
        self.assertEqual(dict(sorted(constant_counts.items())), {0: 1280, 1: 58, 2: 12})

    def test_multi_output_real_rows_are_not_solved_as_single_answer(self):
        multi_ids = [
            row["id"]
            for row, payload in self.payloads
            if payload["answer_type_hint"] == "multi_output"
        ]
        self.assertEqual(multi_ids, ["TD374", "TD376", "THCB066", "DDT340"])

    def test_dataset_wide_concept_and_implicit_coverage_matches_current_data(self):
        concepts = Counter(
            concept
            for _, payload in self.payloads
            for concept in payload["concepts"]
        )
        implicit_rules = Counter(
            fact["rule_id"]
            for _, payload in self.payloads
            for fact in payload["implicit_facts"]
        )

        self.assertEqual(
            dict(concepts),
            {
                "brightness": 12,
                "electric_field_energy": 106,
                "graph_shape": 6,
                "ideal_lc_circuit": 17,
                "impedance": 49,
                "induced_emf": 17,
                "inductance": 98,
                "lc_circuit": 48,
                "magnetic_field_energy": 82,
                "magnetic_flux": 18,
                "parallel_circuit": 146,
                "power_factor": 15,
                "proportionality": 6,
                "qualitative_change": 69,
                "reactance": 78,
                "resonance": 179,
                "rlc_circuit": 141,
                "si_unit": 4,
                "solenoid": 70,
                "total_energy": 17,
                "uniform_electric_field": 2,
            },
        )
        self.assertEqual(
            dict(implicit_rules),
            {
                "electron": 1,
                "fully_charged_capacitor": 2,
                "ideal_lc_no_loss": 18,
                "magnetic_constant": 70,
                "school_coulomb_constant": 289,
                "series_rlc_resonance": 92,
                "vacuum_permittivity": 98,
            },
        )


if __name__ == "__main__":
    unittest.main()
