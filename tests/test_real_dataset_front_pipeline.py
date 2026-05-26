import csv
import unittest
from collections import Counter
from pathlib import Path

from xai_pipeline import process_question_front
from xai_pipeline.engines.logic_engine import allowed_implicit_rule_ids


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
        self.assertGreater(len(self.rows), 0)
        self.assertEqual(self.crashes, [])
        self.assertEqual(len(self.payloads), len(self.rows))

    def test_every_payload_preserves_raw_question_and_stage_trace(self):
        for row, payload in self.payloads:
            with self.subTest(problem_id=row["id"]):
                self.assertEqual(payload["raw_question"], row["question"])
                self.assertEqual(payload["trace"]["stages"], ["semantic_parser", "logic_engine"])
                self.assertFalse(payload["trace"]["semantic_parser"]["llm_used"])
                self.assertFalse(payload["trace"]["logic_engine"]["llm_used"])
                self.assertEqual(payload["answer_type_hint"], payload["trace"]["semantic_parser"]["answer_type_hint"])
                self.assertIn(payload["answer_type_hint"], ANSWER_TYPE_IDS)
                self.assertIn("entities", payload)
                self.assertIn("relations", payload)
                self.assertIn("constraints", payload)
                self.assertIn("goals", payload)

    def test_every_row_has_some_deterministic_front_signal(self):
        unresolved = []
        for row, payload in self.payloads:
            has_front_signal = any(
                [
                    payload["quantities"],
                    payload["symbolic_quantities"],
                    payload["symbolic_relations"],
                    payload["numeric_constants"],
                    payload["entities"],
                    payload["relations"],
                    payload["constraints"],
                    payload["goals"],
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
            applied = set(payload["trace"]["logic_engine"]["rules_applied"])
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
            "entities",
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

    def test_dataset_wide_extraction_distribution_is_stable_enough_for_core(self):
        numeric_counts = Counter(len(payload["quantities"]) for _, payload in self.payloads)
        symbolic_counts = Counter(len(payload["symbolic_quantities"]) for _, payload in self.payloads)
        relation_counts = Counter(len(payload["symbolic_relations"]) for _, payload in self.payloads)
        constant_counts = Counter(len(payload["numeric_constants"]) for _, payload in self.payloads)

        row_count = len(self.payloads)
        self.assertEqual(sum(numeric_counts.values()), row_count)
        self.assertEqual(sum(symbolic_counts.values()), row_count)
        self.assertEqual(sum(relation_counts.values()), row_count)
        self.assertEqual(sum(constant_counts.values()), row_count)
        self.assertGreater(sum(count * frequency for count, frequency in numeric_counts.items()), 2500)
        self.assertGreater(sum(count * frequency for count, frequency in symbolic_counts.items()), 300)
        self.assertGreater(sum(count * frequency for count, frequency in relation_counts.items()), 100)

    def test_multi_output_real_rows_are_not_solved_as_single_answer(self):
        multi_rows = [
            (row, payload)
            for row, payload in self.payloads
            if payload["answer_type_hint"] == "multi_output"
        ]
        self.assertGreater(len(multi_rows), 0)
        bad = []
        target_terms = {
            "charge",
            "current",
            "voltage",
            "resistance",
            "capacitance",
            "power",
            "energy",
            "frequency",
            "field",
            "force",
            "flux",
            "percent",
        }
        for row, payload in multi_rows:
            target_text = " ".join(payload["target_hints"]).lower()
            signal_count = sum(term in target_text for term in target_terms)
            has_multi_marker = any(marker in target_text for marker in [" and ", "respectively", ";", ","])
            if signal_count < 2 and not has_multi_marker:
                bad.append((row["id"], payload["target_hints"]))
        self.assertEqual(bad, [])

    def test_dataset_wide_concept_and_implicit_coverage_tracks_general_families(self):
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

        expected_families = {
            "brightness",
            "electric_field_energy",
            "graph_shape",
            "ideal_lc_circuit",
            "impedance",
            "induced_emf",
            "inductance",
            "lc_circuit",
            "magnetic_field_energy",
            "magnetic_flux",
            "measurement_uncertainty",
            "parallel_circuit",
            "power_factor",
            "proportionality",
            "qualitative_change",
            "reactance",
            "resonance",
            "rlc_circuit",
            "si_unit",
            "solenoid",
            "total_energy",
            "uniform_electric_field",
        }
        missing = sorted(family for family in expected_families if concepts[family] == 0)
        self.assertEqual(missing, [])
        self.assertGreater(concepts["qualitative_change"], concepts["parallel_circuit"])
        self.assertGreaterEqual(implicit_rules["school_coulomb_constant"], 280)
        self.assertGreaterEqual(implicit_rules["vacuum_permittivity"], 90)
        self.assertGreaterEqual(implicit_rules["magnetic_constant"], 60)
        self.assertEqual(set(implicit_rules), set(implicit_rules) & set(allowed_implicit_rule_ids()))


if __name__ == "__main__":
    unittest.main()
