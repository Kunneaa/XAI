import unittest
from unittest.mock import patch
from types import SimpleNamespace

from xai_pipeline.api import handle_request
from xai_pipeline.adaptive_planning import choose_planning_mode
from xai_pipeline.answer_check import check_answer as facade_check_answer
from xai_pipeline.constrained_decoding import planner_json_schema
from xai_pipeline.deadlines import start_deadline as facade_start_deadline
from xai_pipeline.explanation import build_trace_explanation as facade_build_explanation
from xai_pipeline.front_pipeline import process_question_front
from xai_pipeline.geometry import build_template_coordinates, execute_coulomb_force_superposition, execute_coulomb_force_triangle_sides, geometry_recoverability
from xai_pipeline.guarded_polish import guarded_polish_boundary
from xai_pipeline.implicit_classifier import qwen_implicit_classifier_boundary, semantic_match_implicit_rules
from xai_pipeline.json_repair import parse_or_repair_json
from xai_pipeline.llm_budget import LlmBudgetState
from xai_pipeline.numerical_solver import solve_numerically_bounded
from xai_pipeline.planner_schema import validate_planner_schema
from xai_pipeline.planner_executor import execute_validated_plan
from xai_pipeline.principles import select_minimal_equation_subset
from xai_pipeline.qwen_config import check_qwen_model_readiness, resolve_qwen_runtime_config
from xai_pipeline.qwen_runtime import QwenGenerationResult
from xai_pipeline.qwen_planner import plan_with_qwen_if_needed, validate_planner_json
from xai_pipeline.qwen_prompt import build_qwen_planner_prompt
from xai_pipeline.root_filter import filter_roots
from xai_pipeline.router import route
from xai_pipeline.structured_output import select_structured_output_backend
from xai_pipeline.simultaneous_solver import solve_simultaneous_targets
from xai_pipeline.symbolic_executor import execute_symbolic_expression
from xai_pipeline.target_units import convert_si_to_target
from xai_pipeline.telemetry import TelemetryEvent, persist_telemetry_event
from xai_pipeline.unit_converter import convert_front_quantities_to_si
from xai_pipeline.worker_pool import get_default_sympy_pool, solve_symbolic_supervised
from xai_pipeline.verifier import validate_plan, verify_solver


class CoreBoundaryTests(unittest.TestCase):
    def test_api_wrapper_returns_controlled_invalid_request(self):
        response = handle_request({"not_question": "x"})
        self.assertEqual(response["answer"], "Uncertain")
        self.assertEqual(response["confidence"], 0.0)
        self.assertEqual(response["metadata"]["status"], "invalid_request")

    def test_json_repair_extracts_object_without_fabricating_fields(self):
        repaired = parse_or_repair_json('noise {"status":"ok", "targets": [],} tail')
        self.assertTrue(repaired.ok)
        self.assertEqual(repaired.value["status"], "ok")
        self.assertIn("strict_parse_failed_repaired_by_object_extraction", repaired.issues)

    def test_planner_json_validation_rejects_numeric_answer(self):
        front = process_question_front("A resistor R = 10 Ω has voltage U = 20 V. Find current.")
        payload = '{"status":"ok","task_type":"ohm_law","answer_type":"numeric","targets":[{"symbol":"I"}],"formula_ids":["ohm_current"],"principle_ids":["dc_circuit_core"],"geometry_template_ids":[],"implicit_rule_ids":[],"solve_strategy":"direct","numeric_answer":2}'
        result = validate_planner_json(payload, front)
        self.assertTrue(result.used_llm)
        self.assertFalse(result.validation["ok"])
        self.assertIn("planner_supplied_numeric_answer", result.validation["issues"])
        self.assertEqual(result.budget["calls_used"], 1)

    def test_validator_rejects_untriggered_implicit_and_unsupported_conceptual_answer(self):
        front = process_question_front("A resistor R = 10 Ω has voltage U = 20 V. Find current.")
        plan = {
            "status": "ok",
            "task_type": "ohm_law",
            "answer_type": "numeric",
            "given": front["quantities"],
            "targets": [{"symbol": "I"}],
            "formula_ids": ["ohm_current"],
            "principle_ids": ["dc_circuit_core"],
            "geometry_template_ids": [],
            "implicit_rule_ids": ["electron"],
            "decision_notes": ["Given voltage and resistance."],
            "solve_steps": ["Use Ohm law."],
            "solve_strategy": "direct",
            "conceptual_answer": None,
            "confidence": 0.8,
            "numeric_answer": None,
        }
        result = validate_plan(plan, front)
        self.assertFalse(result.ok)
        self.assertIn("implicit_rule_not_triggered:electron", result.issues)

        conceptual = dict(plan)
        conceptual.update(
            {
                "task_type": "conceptual",
                "answer_type": "conceptual",
                "formula_ids": [],
                "principle_ids": [],
                "implicit_rule_ids": [],
                "conceptual_answer": "Yes.",
            }
        )
        conceptual_result = validate_plan(conceptual, front)
        self.assertFalse(conceptual_result.ok)
        self.assertIn("conceptual_answer_without_principle", conceptual_result.issues)

    def test_planner_schema_requires_production_fields(self):
        result = validate_planner_schema({"status": "ok"})
        self.assertFalse(result.ok)
        self.assertIn("missing_field:task_type", result.issues)

    def test_principle_selector_and_geometry_matcher_are_deterministic(self):
        front = process_question_front("Two equal charges are separated by 10 cm. Find the field at the midpoint.")
        route_result = route(front)
        selected = select_minimal_equation_subset(front, route_result)
        geometry = geometry_recoverability(front)
        self.assertIn("stage", selected.trace)
        self.assertTrue(geometry["recoverable"])
        self.assertIn("point_on_midpoint", [match["template_id"] for match in geometry["matches"]])

    def test_sympy_and_numerical_engines_are_fail_closed_or_solve_when_available(self):
        symbolic = solve_symbolic_supervised(equations=["x+1=2"], targets=["x"])
        pool = get_default_sympy_pool()
        pooled = pool.solve(equations=["x+1=2"], targets=["x"])
        self.assertEqual(pooled.trace["pool"], "persistent_warm_worker")
        self.assertIn("worker_pid", pooled.trace)
        numerical = solve_numerically_bounded(family_id="test_family", bounds=(0.0, 1.0))
        if symbolic.ok:
            self.assertEqual(symbolic.value, [{"x": 1.0}])
        else:
            self.assertTrue(
                "sympy_timeout" in symbolic.issues
                or any(issue.startswith("sympy_not_available") for issue in symbolic.issues)
                or any(issue.startswith("sympy_error") for issue in symbolic.issues)
            )
        if pooled.ok:
            self.assertEqual(pooled.value, [{"x": 1.0}])
        else:
            self.assertTrue(
                "sympy_timeout" in pooled.issues
                or any(issue.startswith("sympy_not_available") for issue in pooled.issues)
                or any(issue.startswith("sympy_error") for issue in pooled.issues)
            )
        self.assertFalse(numerical.ok)
        self.assertIn("numerical_family_not_whitelisted:test_family", numerical.issues)

        numerical_real = solve_numerically_bounded(
            family_id="rc_charge_fraction",
            bounds=(0.0, 10.0),
            parameters={"tau": 2.0, "fraction": 0.5},
        )
        self.assertTrue(numerical_real.ok)
        self.assertAlmostEqual(numerical_real.value, 1.386294361, places=6)

        response = handle_request({"question": "In an RC charging circuit, R = 10 Ω and C = 5 μF. How long does it take the capacitor voltage to reach 50% of its final value?"})
        self.assertTrue(response["verifier"]["ok"])
        self.assertEqual(response["solver"]["formula_id"], "rc_charge_fraction_time")
        self.assertIn("numerical_fallback", response["solver"]["trace"])
        self.assertEqual(response["confidence"], 0.75)

    def test_remaining_core_boundaries_are_guarded(self):
        semantic = semantic_match_implicit_rules("Two point charges are in air.")
        self.assertTrue(semantic.ok)
        self.assertTrue(any(match["rule_id"] == "school_coulomb_constant" for match in semantic.matches))

        implicit_llm = qwen_implicit_classifier_boundary("question", ["school_coulomb_constant"])
        self.assertFalse(implicit_llm.ok)
        self.assertIn("qwen_implicit_classifier_disabled", implicit_llm.issues)
        readiness = SimpleNamespace(ready=True, issues=[], to_dict=lambda: {"ready": True, "issues": []})
        qwen_config = SimpleNamespace(enabled=True, readiness=readiness, to_dict=lambda: {"enabled": True, "readiness": readiness.to_dict()})
        with patch.dict("os.environ", {"XAI_ENABLE_QWEN_IMPLICIT": "1"}, clear=False):
            with patch(
                "xai_pipeline.implicit_classifier.generate_planner_text",
                return_value=QwenGenerationResult(True, '{"matches":[{"rule_id":"school_coulomb_constant","trigger_span":"point charges","confidence":0.9}]}', [], {"stage": "fake_qwen"}),
            ):
                implicit_real = qwen_implicit_classifier_boundary("Two point charges are in oil.", ["school_coulomb_constant"], runtime_config=qwen_config)
        self.assertTrue(implicit_real.ok)
        self.assertEqual(implicit_real.matches[0]["rule_id"], "school_coulomb_constant")

        budget = LlmBudgetState()
        self.assertTrue(budget.record_call("combined_normalize_and_plan"))
        self.assertTrue(budget.record_call("repair"))
        self.assertFalse(budget.can_call("guarded_polish"))

        polish = guarded_polish_boundary("Explanation 2 A", "2 A", 10.0, budget)
        self.assertFalse(polish.accepted)
        self.assertIn("llm_budget_disallows_polish", polish.issues)

        allowed_budget = LlmBudgetState()
        config = SimpleNamespace(enabled=True, readiness=readiness, to_dict=lambda: {"enabled": True, "readiness": readiness.to_dict()})
        with patch.dict("os.environ", {"XAI_ENABLE_QWEN_POLISH": "1"}, clear=False):
            with patch(
                "xai_pipeline.guarded_polish.generate_planner_text",
                return_value=QwenGenerationResult(True, "Polished trace keeps the final result as 2 A.", [], {"stage": "fake_qwen"}),
            ):
                accepted_polish = guarded_polish_boundary("Verified result is 2 A.", "2 A", 10.0, allowed_budget, runtime_config=config)
        self.assertTrue(accepted_polish.accepted)
        self.assertIn("2 A", accepted_polish.explanation)

        equivalent_budget = LlmBudgetState()
        with patch.dict("os.environ", {"XAI_ENABLE_QWEN_POLISH": "1"}, clear=False):
            with patch(
                "xai_pipeline.guarded_polish.generate_planner_text",
                return_value=QwenGenerationResult(True, "The verified final result is 0.045 J.", [], {"stage": "fake_qwen"}),
            ):
                equivalent_polish = guarded_polish_boundary("Verified result is 45 mJ.", "45 mJ", 10.0, equivalent_budget, runtime_config=config)
        self.assertTrue(equivalent_polish.accepted)

        converted = convert_si_to_target(0.002, "length", "cm")
        self.assertTrue(converted.ok)
        self.assertAlmostEqual(converted.value, 0.2)

        roots = filter_roots([-1, 2], target_dimension="time", elapsed_time=True)
        self.assertEqual(roots["valid_roots"], [2.0])

        symbolic = execute_symbolic_expression({"formula_ids": ["ohm_current"]})
        simultaneous = solve_simultaneous_targets({"targets": [{"symbol": "x"}]}, [])
        self.assertFalse(symbolic["ok"])
        self.assertIn("symbolic_family_not_whitelisted:None", symbolic["issues"])
        symbolic_real = execute_symbolic_expression(
            {
                "symbolic_family": "linear_single_equation",
                "equations": ["x+1=2"],
                "targets": [{"symbol": "x", "dimension": "dimensionless"}],
            }
        )
        if symbolic_real["ok"]:
            self.assertEqual(symbolic_real["value"], 1.0)
        else:
            self.assertTrue(any(issue.startswith("sympy_not_available") for issue in symbolic_real["issues"]) or "sympy_timeout" in symbolic_real["issues"])
        registry_graph = execute_symbolic_expression(
            {
                "symbolic_family": "registry_equation_graph",
                "equations": ["40.0=I**2*10.0"],
                "targets": [{"symbol": "I", "dimension": "current", "unit": "A"}],
                "non_negative_target": True,
            }
        )
        self.assertTrue(registry_graph["ok"])
        self.assertAlmostEqual(float(registry_graph["value"]), 2.0, places=6)
        self.assertFalse(simultaneous["ok"])
        self.assertIn("simultaneous_family_not_whitelisted:None", simultaneous["issues"])
        unsafe_simultaneous = solve_simultaneous_targets(
            {
                "simultaneous_family": "linear_system",
                "equations": ["__import__(os)=x"],
                "targets": [{"symbol": "x"}],
            },
            [],
        )
        self.assertFalse(unsafe_simultaneous["ok"])
        self.assertIn("unknown_equation_symbol:0:__import__,os", unsafe_simultaneous["issues"])
        simultaneous_real = solve_simultaneous_targets(
            {
                "simultaneous_family": "linear_system",
                "equations": ["x+y=3", "x-y=1"],
                "targets": [{"symbol": "x"}, {"symbol": "y"}],
            },
            [],
        )
        if simultaneous_real["ok"]:
            self.assertEqual(simultaneous_real["partial_results"], [{"symbol": "x", "value": 2.0, "unit": None}, {"symbol": "y", "value": 1.0, "unit": None}])
        else:
            self.assertTrue(any(issue.startswith("sympy_not_available") for issue in simultaneous_real["issues"]) or "sympy_timeout" in simultaneous_real["issues"])

        telemetry_path = __import__("pathlib").Path(__import__("tempfile").gettempdir()) / "xai_core_boundary_telemetry.jsonl"
        if telemetry_path.exists():
            telemetry_path.unlink()
        stored = persist_telemetry_event(TelemetryEvent("unit_test", {"ok": True}), telemetry_path)
        self.assertTrue(stored["written"])
        self.assertIn('"event_type": "unit_test"', telemetry_path.read_text(encoding="utf-8"))

    def test_core_facades_and_planning_boundaries_are_available(self):
        self.assertGreaterEqual(facade_start_deadline().remaining_seconds(), 0.0)
        self.assertTrue(callable(facade_check_answer))
        self.assertTrue(callable(facade_build_explanation))
        with patch.dict("os.environ", {"XAI_QWEN_STRUCTURED_ENDPOINT": ""}, clear=False):
            backend = select_structured_output_backend("vllm_guided_json")
        self.assertFalse(backend.available)
        self.assertIn("structured_output_backend_not_connected", backend.issues)
        combined = choose_planning_mode([])
        split = choose_planning_mode([{"schema_failure": True}, {"schema_failure": True}, {"planner_failure": True}])
        self.assertEqual(combined["mode"], "combined_normalize_and_plan")
        self.assertEqual(split["mode"], "split_extract_then_plan")
        schema = planner_json_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("numeric_answer", schema["required"])

    def test_geometry_engine_and_target_unit_conversion_are_real(self):
        geometry = execute_coulomb_force_superposition(
            "right_isosceles_triangle_vertex",
            {"leg": 0.12},
            [{"point": "B", "charge_c": 4e-6}, {"point": "C", "charge_c": 4e-6}],
            {"point": "A", "charge_c": 4e-6},
        )
        self.assertTrue(geometry.ok)
        self.assertAlmostEqual(geometry.value, 14.1421356, places=5)
        self.assertIn("coordinates", geometry.trace)

        triangle_geometry = execute_coulomb_force_triangle_sides(
            ab=0.20,
            ac=0.12,
            bc=0.16,
            q_a=-3e-6,
            q_b=8e-6,
            q_c=2e-6,
        )
        self.assertTrue(triangle_geometry.ok)
        self.assertAlmostEqual(triangle_geometry.value, 6.7604086, places=5)
        triangle_coordinates = build_template_coordinates("triangle_sides", {"ab": 0.20, "ac": 0.12, "bc": 0.16})
        self.assertIn("C", triangle_coordinates)
        triangle_recoverability = geometry_recoverability(
            {
                "canonical_question": "Points A and B are separated by 20 cm. q3 is at C with AC = 12 cm and BC = 16 cm."
            }
        )
        self.assertTrue(triangle_recoverability["recoverable"])
        self.assertIn("triangle_sides", [match["template_id"] for match in triangle_recoverability["matches"]])

        response = handle_request({"question": "Calculate the capacitance of a capacitor that has 200 μJ of stored energy when the voltage across it is 10 V. Answer in μF."})
        self.assertTrue(response["verifier"]["ok"])
        self.assertEqual(response["answer"], "4 μF")
        self.assertTrue(response["trace"]["target_unit_conversion"]["applied"])

    def test_verifier_accepts_geometry_components_trace(self):
        route_result = SimpleNamespace(task_type="electric_field_point", confidence=0.9)
        front = {"answer_type_hint": "numeric"}
        solver_result = SimpleNamespace(
            solved=True,
            answer="5 V/m",
            value=5.0,
            unit="V/m",
            formula_id="electric_field_point",
            principle_id="electric_field_core",
            trace={"geometry_engine": {"components": {"x": 3.0, "y": 4.0, "magnitude": 5.0}, "value": 5.0}},
            confidence=0.9,
        )
        verified = verify_solver(front, route_result, solver_result)
        self.assertTrue(verified.ok, verified.issues)

        mismatched = SimpleNamespace(
            **{
                **solver_result.__dict__,
                "trace": {"geometry_engine": {"components": {"x": 3.0, "y": 4.0, "magnitude": 6.0}, "value": 5.0}},
            }
        )
        rejected = verify_solver(front, route_result, mismatched)
        self.assertFalse(rejected.ok)
        self.assertIn("vector_component_magnitude_mismatch", rejected.issues)

    def test_symbolic_perpendicular_bisector_field_is_deterministic(self):
        question = (
            "Two charges, q1 = q2 = q (where q > 0, in Coulombs), are placed at points A and B, "
            "with the distance AB = 2a (meters). Point M is located on the perpendicular bisector "
            "of the line segment AB, at a distance h from AB. Determine the magnitude of the electric field vector at point M. "
            "Given k = 9 × 10^9."
        )
        response = handle_request({"question": question}, enable_llm=False)
        self.assertTrue(response["verifier"]["ok"])
        self.assertEqual(response["route"]["task_type"], "electric_field_point")
        self.assertEqual(response["route"]["answer_type"], "symbolic")
        self.assertEqual(response["solver"]["formula_id"], "electric_field_two_charge_isosceles")
        self.assertEqual(response["answer"], "2*k*q*h/(a^2 + h^2)^(3/2) V/m")

    def test_local_qwen_is_configured_but_guarded_by_default(self):
        readiness = check_qwen_model_readiness()
        self.assertTrue(readiness.ready)
        self.assertEqual(readiness.files["safetensor_shards"], 4)

        with patch.dict("os.environ", {"XAI_ENABLE_LOCAL_QWEN": "0"}, clear=False):
            config = resolve_qwen_runtime_config()
        self.assertFalse(config.enabled)
        self.assertTrue(config.local_files_only)
        self.assertTrue(config.readiness.ready)

        front = process_question_front("A point charge q = 2 nC is in air. Find the electric field at 3 cm.")
        route_result = route(front)
        prompt = build_qwen_planner_prompt(
            front,
            route_result,
            [{"problem_id": "row_1", "score": 0.5, "task_metadata": {"answer": "forbidden", "task_type": "electric_field_point"}}],
            {"name": "vllm_guided_json", "available": False},
        )
        self.assertIn("numeric_answer must always be null", prompt)
        self.assertIn("retrieval_metadata_only", prompt)
        self.assertNotIn("forbidden", prompt)

        result = plan_with_qwen_if_needed(
            front,
            route_result,
            SimpleNamespace(solved=False),
            [],
            enable_llm=True,
            runtime_config=config,
        )
        self.assertFalse(result.used_llm)
        self.assertEqual(result.reason, "local_qwen_disabled")
        self.assertTrue(result.validation["qwen_runtime"]["config"]["readiness"]["ready"])

        strict_config = SimpleNamespace(
            enabled=True,
            readiness=SimpleNamespace(ready=True, issues=[], to_dict=lambda: {"ready": True, "issues": []}),
            structured_backend="local_transformers_json_guarded",
            to_dict=lambda: {"enabled": True, "require_constrained_decoding": True},
            require_constrained_decoding=True,
        )
        strict = plan_with_qwen_if_needed(
            front,
            route_result,
            SimpleNamespace(solved=False),
            [],
            enable_llm=True,
            runtime_config=strict_config,
        )
        self.assertEqual(strict.reason, "constrained_decoding_unavailable")
        self.assertIn("true_constrained_decoding_required_but_unavailable", strict.validation["issues"])

    def test_validated_qwen_plan_can_only_drive_deterministic_execution(self):
        front = process_question_front("A resistor R = 10 Ω has voltage U = 20 V. Find current.")
        plan_json = """
        {
          "status": "ok",
          "task_type": "ohm_law",
          "answer_type": "numeric",
          "given": [],
          "targets": [{"symbol": "I", "name": "current"}],
          "formula_ids": ["ohm_current"],
          "principle_ids": ["dc_circuit_core"],
          "geometry_template_ids": [],
          "implicit_rule_ids": [],
          "decision_notes": ["Given voltage and resistance."],
          "solve_steps": ["Use the whitelisted Ohm-law current relation."],
          "solve_strategy": "direct",
          "conceptual_answer": null,
          "confidence": 0.8,
          "numeric_answer": null
        }
        """
        planner = validate_planner_json(plan_json, front)
        self.assertTrue(planner.validation["ok"])
        executed = execute_validated_plan(front, planner, convert_front_quantities_to_si(front))
        self.assertIsNotNone(executed)
        _, solver_result = executed
        self.assertTrue(solver_result.solved)
        self.assertEqual(solver_result.answer, "2 A")
        self.assertTrue(solver_result.trace["planner_execution"]["used_validated_plan"])

    def test_power_inverse_routes_and_new_registry_formulas_are_deterministic(self):
        cases = [
            ("A resistor dissipates power P = 40 W with resistance R = 10 Ω. Find current.", "ohm_current_power_resistance", "2 A"),
            ("A circuit has voltage U = 20 V and power P = 100 W. Find current I.", "ohm_current_power_voltage", "5 A"),
            ("A resistor has power P = 80 W and current I = 2 A. Find resistance R.", "ohm_resistance_power_current", "20 Ω"),
            ("A capacitor stores energy W = 2 J at voltage U = 4 V. Find charge Q.", "capacitor_charge_energy_voltage", "1 C"),
        ]
        for question, formula_id, answer in cases:
            response = handle_request({"question": question})
            self.assertTrue(response["verifier"]["ok"], response["verifier"]["issues"])
            self.assertEqual(response["solver"]["formula_id"], formula_id)
            self.assertEqual(response["answer"], answer)


if __name__ == "__main__":
    unittest.main()
