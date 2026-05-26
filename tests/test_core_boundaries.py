import json
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from xai_pipeline.core import pipeline as core_pipeline
from xai_pipeline.core.api import handle_request
from xai_pipeline.engines.algebraic_engine import solve_algebraic_plan
from xai_pipeline.engines.equation_engine import SolverResult, solve_fast
from xai_pipeline.engines.logic_engine import allowed_implicit_rule_ids
from xai_pipeline.engines.spatial_engine import build_template_coordinates, execute_coulomb_force_superposition, geometry_recoverability
from xai_pipeline.frontend.semantic_parser import normalize_question, process_question_front
from xai_pipeline.knowledge.constraint_graph import build_constraint_graph, infer_target_dimensions, route, select_minimal_equation_subset
from xai_pipeline.knowledge.formula_catalog import formula_catalog_for_prompt, formula_prompt_pack
from xai_pipeline.knowledge.language import extract_change_factor, has_frequency_transform_cue
from xai_pipeline.knowledge.registries import FORMULA_IDS, audit_registry_generalization, formula_logic_catalog
from xai_pipeline.knowledge.units import convert_si_to_target
from xai_pipeline.planning import local_llm
from xai_pipeline.planning.local_llm import (
    _apply_front_repair_patch,
    _complete_llm_solve_plan_payload,
    _extract_first_safe_json_object,
    _front_repair_enabled,
    _solve_plan_prompt,
    check_local_llm_readiness,
    repair_front_ir_once,
    repair_solve_plan_once,
)
from xai_pipeline.planning.plan_compiler import build_plan_error_packet, compile_solve_plan, validate_structured_solve_plan
from xai_pipeline.planning.solve_plan import build_deterministic_solve_plan
from xai_pipeline.runtime.telemetry import build_pipeline_telemetry, persist_telemetry_event
from xai_pipeline.verification.verifier import validate_plan, verify_solver
from xai_pipeline.xai.explanation import build_explanation


class CoreBoundaryTests(unittest.TestCase):
    def test_api_wrapper_returns_controlled_invalid_request(self):
        response = handle_request({"not_question": "x"})
        self.assertEqual(response["answer"], "Uncertain")
        self.assertEqual(response["confidence"], 0.0)
        self.assertEqual(response["metadata"]["status"], "invalid_request")

    def test_llm_required_planning_fails_closed_without_deterministic_fallback(self):
        question = "A resistor R = 10 ohm has voltage U = 20 V. Find current."
        with patch(
            "xai_pipeline.core.pipeline.propose_solve_plan_if_enabled",
            return_value=(
                None,
                {"stage": "local_llm_solve_plan", "used": True, "reason": "no_json_object"},
            ),
        ), patch(
            "xai_pipeline.core.pipeline.repair_solve_plan_once",
            return_value=(
                None,
                {"stage": "local_llm_solve_plan_repair", "used": True, "applied": False, "reason": "no_json_object"},
            ),
        ):
            response = handle_request(
                {"question": question},
                enable_llm=True,
                planning_mode="llm_required",
                timeout_seconds=-1,
            )

        self.assertEqual(response["answer"], "Uncertain")
        self.assertEqual(response["metadata"]["status"], "llm_plan_unavailable")
        self.assertEqual(response["metadata"]["planning_mode"], "llm_required")
        self.assertEqual(response["solve_plan"]["plan"]["source"], "local_llm_required")
        self.assertIn("llm_plan_missing_or_invalid_json", response["solve_plan"]["issues"])
        self.assertFalse(response["solver"]["solved"])
        self.assertEqual(response["solver"]["trace"]["reason"], "plan_compilation_failed")

    def test_executor_result_outside_compiled_plan_is_rejected_at_dispatch(self):
        question = "A resistor R = 10 ohm has voltage U = 20 V. Find current."
        front = process_question_front(question)
        route_result = route(front)
        graph_selection = select_minimal_equation_subset(front, route_result)
        plan = {
            "status": "ok",
            "task_type": "ohm_law",
            "answer_type": "numeric",
            "targets": [{"id": "goal:1", "quantity": "current", "symbol": "I", "text": "current", "unit": "A"}],
            "assumptions": [],
            "steps": [
                {
                    "step_id": "s1",
                    "operation": "apply_formula",
                    "formula_id": "ohm_current",
                    "principle_id": "dc_circuit_core",
                    "inputs": {"dimensions": ["voltage", "resistance"]},
                    "output": "goal:1",
                    "depends_on": [],
                    "public_cot": "Apply the selected registry relation to accepted facts.",
                }
            ],
            "output_format": {"format_kind": "numeric_scalar", "ordered_targets": ["goal:1"], "preferred_unit": "A", "target_count": 1},
            "source": "local_llm",
        }
        compiled = compile_solve_plan(plan, front, route_result, graph_selection)
        self.assertTrue(compiled.ok, compiled.issues)
        off_plan_result = SolverResult(
            True,
            "999 J",
            999.0,
            "J",
            "capacitor_energy_voltage",
            "capacitor_core",
            [],
            {"stage": "fake_fast_solver"},
            0.9,
        )
        unsolved = SolverResult(False, "", None, None, None, None, [], {"stage": "fake_unsolved"}, 0.0)

        with patch("xai_pipeline.core.pipeline.solve_conceptual", return_value=unsolved), patch(
            "xai_pipeline.core.pipeline.solve_fast", return_value=off_plan_result
        ), patch("xai_pipeline.core.pipeline.solve_spatial_from_front", return_value=unsolved), patch(
            "xai_pipeline.core.pipeline.solve_algebraic_plan", return_value=unsolved
        ):
            result = core_pipeline._dispatch_solver(front, route_result, graph_selection, compiled)

        self.assertFalse(result.solved)
        rejected = result.trace["engine_attempts"]["fast_formula"]["off_plan_result_rejected"]
        self.assertEqual(rejected["solver_formula_id"], "capacitor_energy_voltage")
        self.assertEqual(rejected["compiled_formula_ids"], ["ohm_current"])

    def test_semantic_frontend_applies_hidden_si_units_without_llm(self):
        normalized = normalize_question("A capacitor has C = 100 and U = 50. Find charge.")
        by_symbol = {quantity.symbol: quantity for quantity in normalized.quantities}
        self.assertEqual(by_symbol["C"].unit, "F")
        self.assertEqual(by_symbol["U"].unit, "V")
        self.assertEqual(normalized.answer_type_hint, "numeric")
        self.assertTrue(any(entity.entity_type == "capacitor" for entity in normalized.entities))
        self.assertTrue(any(goal.dimension == "charge" for goal in normalized.goals))

        front = process_question_front("Two point charges are in air. q1 = 2 μC, q2 = 3 μC, r = 10 cm. Find force.")
        self.assertIn("school_coulomb_constant", {fact["rule_id"] for fact in front["implicit_facts"]})
        self.assertTrue(front["entities"])
        self.assertTrue(any(constraint["constraint_id"] == "medium_air_or_vacuum" for constraint in front["constraints"]))
        self.assertFalse(front["trace"]["semantic_parser"]["llm_used"])
        self.assertFalse(front["trace"]["logic_engine"]["llm_used"])

        side = process_question_front("A triangle has side length a = 3. Find electric field.")
        side_by_symbol = {quantity["symbol"]: quantity for quantity in side["quantities"]}
        self.assertEqual(side_by_symbol["a"]["dimension"], "length")
        self.assertEqual(side_by_symbol["a"]["unit"], "m")
        self.assertEqual(side_by_symbol["a"]["raw_unit"], "implicit_base_SI")
        self.assertEqual(
            process_question_front("Point P is on line segment AB. Find the electric field at P.")["topology_graph"]["canonical_form"],
            "no_circuit_topology",
        )

        area = process_question_front("A plate has area A = 3. Find capacitance.")
        area_by_symbol = {quantity["symbol"]: quantity for quantity in area["quantities"]}
        self.assertEqual(area_by_symbol["A"]["dimension"], "area")
        self.assertEqual(area_by_symbol["A"]["unit"], "m^2")

    def test_semantic_frontend_keeps_physics_language_structural_not_pattern_specific(self):
        qualitative = process_question_front(
            "An isolated parallel-plate capacitor has its plate separation increased by a factor of 3. "
            "How do capacitance, voltage, and stored energy change?"
        )
        self.assertEqual(qualitative["answer_type_hint"], "conceptual")
        self.assertIn("qualitative_change", qualitative["concepts"])
        self.assertNotIn("parallel_circuit", qualitative["concepts"])
        self.assertFalse(any(relation["qualifier"] == "parallel" for relation in qualitative["relations"]))
        self.assertEqual(qualitative["topology_graph"]["canonical_form"], "single_component_or_global_circuit")

        circuit = process_question_front("Two capacitors are connected in parallel. Find equivalent capacitance.")
        self.assertIn("parallel_circuit", circuit["concepts"])
        self.assertTrue(any(relation["qualifier"] == "parallel" for relation in circuit["relations"]))

    def test_semantic_frontend_generalizes_symbols_and_geometry_labels(self):
        prime = process_question_front("A charge q′ = -1 μC is placed at the remaining vertex of an equilateral triangle.")
        self.assertEqual(prime["quantities"][0]["symbol"], "q′")
        self.assertEqual(prime["quantities"][0]["dimension"], "charge")
        self.assertEqual([entity["symbol"] for entity in prime["entities"]], ["q′"])

        right_isosceles = process_question_front(
            "At the three vertices of a right-angled isosceles triangle PQR, with PQ = PR = a, "
            "charges qP = qR = q are placed."
        )
        self.assertTrue(any(relation["qualifier"] == "right_isosceles_triangle" for relation in right_isosceles["relations"]))
        triangle = right_isosceles["canonical_structures"]["geometry"]["triangles"][0]
        self.assertEqual(triangle["right_angle_at"], "P")
        self.assertEqual(triangle["canonical_right_angle_at"], "A")

        rlc_factor = process_question_front(
            "An RLC circuit has inductive reactance XL = 20 Ω and capacitive reactance XC = 80 Ω. "
            "How many times must the frequency change to be resonant?"
        )
        self.assertEqual(rlc_factor["answer_type_hint"], "numeric")

        vertex_labels = process_question_front("Triangle has vertices P, Q, and R. Find the electric field at P.")
        self.assertEqual(vertex_labels["answer_type_hint"], "symbolic")
        self.assertFalse(any(entity["entity_type"] == "charge" for entity in vertex_labels["entities"]))
        self.assertEqual(vertex_labels["canonical_structures"]["geometry"]["triangles"][0]["labels"], ["P", "Q", "R"])

        square_labels = process_question_front(
            "Given a square WXYZ with side length a. Charges qW = qY = q are placed at W and Y. "
            "What charge must be placed at X so that the electric field at Z is zero?"
        )
        self.assertFalse(any(item["symbol"] == "W" and item["dimension"] == "energy" for item in square_labels["symbolic_quantities"]))
        self.assertEqual(square_labels["symbolic_relations"][0]["rhs"], "qY = q")
        self.assertEqual(square_labels["goals"][0]["symbol"], "qX")
        square = square_labels["canonical_structures"]["geometry"]["squares"][0]
        self.assertEqual(square["labels"], ["W", "X", "Y", "Z"])
        self.assertEqual(square["canonical_by_original"]["W"], "A")

    def test_constraint_graph_is_registry_owned(self):
        front = process_question_front("A resistor R = 10 Ω has voltage U = 20 V. Find current.")
        routed = route(front)
        selected = select_minimal_equation_subset(front, routed)
        plan = build_deterministic_solve_plan(front, routed, selected)
        compiled = compile_solve_plan(plan.to_dict(), front, routed, selected)
        self.assertEqual(routed.task_type, "ohm_law")
        self.assertTrue(selected.ok)
        self.assertTrue(compiled.ok, compiled.issues)
        self.assertEqual(plan.steps[0].operation, "apply_formula")
        self.assertEqual(compiled.plan["steps"][0]["formula_id"], "ohm_current")
        self.assertEqual(compiled.plan["output_format"]["format_kind"], "numeric_scalar")
        self.assertEqual(compiled.plan["output_format"]["preferred_unit"], "A")
        self.assertIn("ohm_current", selected.formula_ids)
        self.assertIn("graph", selected.trace)
        self.assertTrue(selected.trace["graph"]["edges"])
        self.assertIn("ohm_current", selected.trace["graph"]["reachable_formula_ids"])
        self.assertTrue(selected.trace["graph"]["selected_formula_ids"])
        self.assertIn("coulomb_force_triangle_sides", FORMULA_IDS)

    def test_knowledge_routing_uses_structural_cues_not_problem_specific_symbols(self):
        rlc_front = process_question_front(
            "An RLC circuit has inductive reactance XL = 20 Ω and capacitive reactance XC = 80 Ω. "
            "How many times must the frequency change to be resonant?"
        )
        rlc_route = route(rlc_front)
        self.assertEqual(rlc_route.task_type, "ohm_law")
        self.assertEqual(rlc_route.answer_type, "numeric")
        self.assertIn("frequency multiplier", rlc_route.reasons[0])

        distance_front = process_question_front(
            "A test charge q0 = 1 μC is placed at a point whose distances to two source charges "
            "q1 = 2 μC and q2 = 3 μC are 3 cm and 4 cm, respectively. "
            "The two source charges are separated by 5 cm. Find the magnitude of the net electric "
            "force acting on q0."
        )
        distance_graph = build_constraint_graph(distance_front, route(distance_front))
        self.assertIn("coulomb_force_triangle_sides", distance_graph.reachable_formula_ids)

        structural_triangle = process_question_front(
            "Three equal charges q1 = q2 = q3 = 1 C are arranged on a regular triangle "
            "with side length a = 1 m. Find the net force on q3."
        )
        structural_triangle["canonical_question"] = "Find the net force on q3."
        triangle_route = route(structural_triangle)
        triangle_graph = build_constraint_graph(structural_triangle, triangle_route)
        self.assertEqual(triangle_route.task_type, "coulomb_force")
        self.assertIn("coulomb_force_triangle_sides", triangle_graph.selected_formula_ids)

        structural_square = process_question_front(
            "At the center of square WXYZ, equal charges qW = qY = q are placed at opposite corners W and Y. "
            "What charge must be placed at X so that the electric field at the center is zero?"
        )
        structural_square["canonical_question"] = "What charge makes the electric field zero at the observation point?"
        square_route = route(structural_square)
        square_selected = select_minimal_equation_subset(structural_square, square_route)
        self.assertEqual(square_route.task_type, "electric_field_point")
        self.assertIn("electric_field_square_center_cancel_charge", square_selected.formula_ids)

    def test_formula_catalog_exposes_dataset_synthesized_registry_without_answers(self):
        catalog = formula_catalog_for_prompt(route_task_type="ohm_law", candidate_formula_ids=["ohm_current"])
        self.assertEqual(catalog["source"]["file"], "Physics_Problems_Text_Only.csv")
        self.assertEqual(catalog["formula_count"], len(FORMULA_IDS))
        self.assertIn("ohm_current", catalog["all_formula_ids"])
        self.assertIn("ohm_current", catalog["candidate_formula_ids"])
        self.assertIn("ohm_law", catalog["family_index"])
        self.assertTrue(any(card["formula_id"] == "ohm_current" for card in catalog["detailed_formula_cards"]))
        self.assertIn("no dataset answers", catalog["source"]["inference_leakage_policy"])

        front = process_question_front("A resistor R = 10 Ω has voltage U = 20 V. Find current.")
        restricted = solve_fast(front, route(front), allowed_formula_ids=["ohm_voltage"])
        self.assertFalse(restricted.solved)
        self.assertEqual(restricted.trace["reason"], "no_registry_formula_executed")

    def test_llm_formula_prompt_pack_is_route_local_not_full_registry(self):
        front = process_question_front("A resistor R = 10 Ω has voltage U = 20 V. Find current.")
        routed = route(front)
        selected = select_minimal_equation_subset(front, routed)
        prompt = _solve_plan_prompt(front, routed, selected)
        self.assertLess(len(prompt), 5200)
        self.assertNotIn("PLAN_SHAPE:", prompt)
        self.assertIn("DATA:", prompt)
        self.assertIn("formula_menu", prompt)
        self.assertIn("plan_cards", prompt)
        self.assertIn("direct_formula", prompt)
        self.assertIn("plan_card_id", prompt)
        self.assertIn("ohm_current", prompt)
        self.assertNotIn("coulomb_force_triangle_sides", prompt)
        self.assertNotIn("hidden_registry_audit", prompt)
        self.assertNotIn("allowed_template_ids", prompt)
        self.assertNotIn("Copy the skeleton", prompt)

        valid_plan = {
            "status": "ok",
            "task_type": "ohm_law",
            "answer_type": "numeric",
            "targets": [{"id": "goal:1", "quantity": "current", "unit": "A"}],
            "steps": [
                {
                    "step_id": "s1",
                    "operation": "apply_formula",
                    "formula_id": "ohm_current",
                    "principle_id": "dc_circuit_core",
                    "inputs": {},
                    "output": "goal:1",
                    "depends_on": [],
                    "public_cot": "Apply the selected registry formula to SI-normalized facts.",
                }
            ],
            "output_format": {"format_kind": "numeric_scalar", "ordered_targets": ["goal:1"], "target_count": 1},
        }
        recovered = _extract_first_safe_json_object('{"status":"bad"}\n' + json.dumps(valid_plan), schema="solve_plan")
        self.assertEqual(recovered["status"], "ok")
        self.assertEqual(recovered["steps"][0]["formula_id"], "ohm_current")
        unsafe_then_valid = {
            **valid_plan,
            "steps": [{**valid_plan["steps"][0], "numeric_answer": 2}],
        }
        recovered_after_unsafe = _extract_first_safe_json_object(
            json.dumps(unsafe_then_valid) + "\n" + json.dumps(valid_plan),
            schema="solve_plan",
        )
        self.assertEqual(recovered_after_unsafe["steps"][0]["formula_id"], "ohm_current")
        self.assertNotIn("numeric_answer", recovered_after_unsafe["steps"][0])

        pack = formula_prompt_pack(
            route_task_type="ohm_law",
            candidate_formula_ids=["ohm_current"],
            available_dimensions=["resistance", "voltage"],
            target_dimensions=["current"],
            route_reasons=["registry formula ohm_current connects ['voltage', 'resistance'] -> current"],
        )
        self.assertEqual(pack["scope"], "route_local_compact")
        self.assertLess(len(pack["allowed_formula_ids"]), len(FORMULA_IDS))
        self.assertEqual(pack["allowed_formula_ids"][0], "ohm_current")
        self.assertIn("ohm_current", pack["candidate_formula_ids"])
        self.assertTrue(any(card["id"] == "ohm_current" for card in pack["cards"]))
        evidence = pack["decision_evidence"]["candidates"][0]
        self.assertEqual(evidence["formula_id"], "ohm_current")
        self.assertTrue(evidence["selected_by_graph"])
        self.assertTrue(evidence["input_match"])
        self.assertTrue(evidence["target_match"])
        self.assertEqual(evidence["missing_dimensions"], [])
        self.assertEqual(evidence["branch"], "scalar_equation")
        self.assertEqual(pack["cards"][0]["branch"], "scalar_equation")
        self.assertNotIn("coulomb_force_triangle_sides", pack["allowed_formula_ids"])
        self.assertIn("full registry is hidden", pack["policy"].lower())

        compact_plan = {
            "status": "ok",
            "steps": [
                {
                    "step_id": "s1",
                    "operation": "apply_formula",
                    "formula_id": "ohm_current",
                    "principle_id": "dc_circuit_core",
                    "inputs": {},
                    "output": "goal:1",
                    "depends_on": [],
                    "public_cot": "Apply the selected registry formula to SI-normalized facts.",
                }
            ],
            "notes": ["compact"],
        }
        recovered_compact = _extract_first_safe_json_object(json.dumps(compact_plan), schema="solve_plan")
        completed = _complete_llm_solve_plan_payload(recovered_compact, front, routed, selected)
        self.assertEqual(completed["task_type"], "ohm_law")
        self.assertEqual(completed["answer_type"], "numeric")
        self.assertEqual(completed["targets"][0]["quantity"], "current")
        self.assertEqual(completed["output_format"]["format_kind"], "numeric_scalar")
        self.assertTrue(compile_solve_plan(completed, front, routed, selected).ok)

        compact_card_plan = {
            "status": "ok",
            "plan_template_id": "direct_formula",
            "formula_id": "ohm_current",
            "public_cot": ["Apply the selected registry relation to accepted facts."],
        }
        recovered_card = _extract_first_safe_json_object(json.dumps(compact_card_plan), schema="solve_plan")
        completed_card = _complete_llm_solve_plan_payload(recovered_card, front, routed, selected)
        self.assertEqual(completed_card["steps"][0]["operation"], "apply_formula")
        self.assertEqual(completed_card["steps"][0]["formula_id"], "ohm_current")
        self.assertTrue(compile_solve_plan(completed_card, front, routed, selected).ok)

        tiny_card_plan = {"status": "ok", "plan_card_id": "p1"}
        recovered_tiny_card = _extract_first_safe_json_object(json.dumps(tiny_card_plan), schema="solve_plan")
        completed_tiny_card = _complete_llm_solve_plan_payload(recovered_tiny_card, front, routed, selected)
        self.assertEqual(completed_tiny_card["steps"][0]["operation"], "apply_formula")
        self.assertEqual(completed_tiny_card["steps"][0]["formula_id"], "ohm_current")
        self.assertTrue(compile_solve_plan(completed_tiny_card, front, routed, selected).ok)

        spatial_front = process_question_front(
            "Three electric charges, q1 = q2 = q3 = 1.6e-19 C, are placed at the vertices "
            "of an equilateral triangle ABC with side length 16 cm. Determine the net force on q3."
        )
        spatial_route = route(spatial_front)
        spatial_selected = select_minimal_equation_subset(spatial_front, spatial_route)
        minimal_spatial_plan = {
            "status": "ok",
            "steps": [
                {
                    "step_id": "s1",
                    "operation": "construct_geometry",
                    "geometry_constructor_id": "equilateral_triangle",
                    "inputs": [],
                    "output": "geom",
                    "depends_on": [],
                    "public_cot": "Construct accepted geometry.",
                },
                {
                    "step_id": "s2",
                    "operation": "compute_pairwise_force",
                    "formula_id": "coulomb_force_triangle_sides",
                    "principle_id": "coulomb_core",
                    "inputs": {},
                    "output": "goal:1",
                    "depends_on": ["s1"],
                    "public_cot": "Resolve vector contributions.",
                },
            ],
        }
        completed_spatial = _complete_llm_solve_plan_payload(
            minimal_spatial_plan,
            spatial_front,
            spatial_route,
            spatial_selected,
        )
        self.assertEqual(completed_spatial["steps"][0]["geometry_constructor_id"], "equilateral_triangle_vertex")
        self.assertEqual(completed_spatial["steps"][0]["inputs"], {})
        self.assertTrue(compile_solve_plan(completed_spatial, spatial_front, spatial_route, spatial_selected).ok)

        compact_spatial_card = {
            "status": "ok",
            "plan_template_id": "spatial_pairwise_force",
            "formula_id": "coulomb_force_triangle_sides",
            "geometry_template_id": "equilateral_triangle",
            "public_cot": ["Construct accepted geometry.", "Resolve vector contributions."],
        }
        recovered_spatial_card = _extract_first_safe_json_object(json.dumps(compact_spatial_card), schema="solve_plan")
        completed_spatial_card = _complete_llm_solve_plan_payload(
            recovered_spatial_card,
            spatial_front,
            spatial_route,
            spatial_selected,
        )
        self.assertEqual(
            [step["operation"] for step in completed_spatial_card["steps"]],
            ["construct_geometry", "compute_pairwise_force"],
        )
        self.assertEqual(completed_spatial_card["steps"][0]["geometry_constructor_id"], "equilateral_triangle_vertex")
        self.assertTrue(compile_solve_plan(completed_spatial_card, spatial_front, spatial_route, spatial_selected).ok)

        truncated_spatial = (
            '{"status":"ok","plan_template_id":"spatial_pairwise_force",'
            '"formula_id":"coulomb_force_triangle_sides",'
            '"geometry_template_id":"equilateral_triangle",'
            '"public_cot":"Step 1: Identify values q1 = q2 = q3 and r = 16 cm. Step 2'
        )
        recovered_truncated = _extract_first_safe_json_object(truncated_spatial, schema="solve_plan")
        completed_truncated = _complete_llm_solve_plan_payload(
            recovered_truncated,
            spatial_front,
            spatial_route,
            spatial_selected,
        )
        self.assertEqual(
            [step["operation"] for step in completed_truncated["steps"]],
            ["construct_geometry", "compute_pairwise_force"],
        )
        self.assertEqual(completed_truncated["steps"][0]["public_cot"], "Construct accepted geometry.")
        self.assertTrue(compile_solve_plan(completed_truncated, spatial_front, spatial_route, spatial_selected).ok)

    def test_llm_readiness_cache_tracks_runtime_env(self):
        old_timeout = os.environ.get("XAI_LLM_HARD_TIMEOUT_SECONDS")
        try:
            local_llm._READINESS_CACHE.clear()
            os.environ["XAI_LLM_HARD_TIMEOUT_SECONDS"] = "3"
            first = check_local_llm_readiness()
            os.environ["XAI_LLM_HARD_TIMEOUT_SECONDS"] = "7"
            second = check_local_llm_readiness()
            self.assertEqual(first["runtime_config"]["hard_timeout_seconds"], 3.0)
            self.assertEqual(second["runtime_config"]["hard_timeout_seconds"], 7.0)
        finally:
            if old_timeout is None:
                os.environ.pop("XAI_LLM_HARD_TIMEOUT_SECONDS", None)
            else:
                os.environ["XAI_LLM_HARD_TIMEOUT_SECONDS"] = old_timeout
            local_llm._READINESS_CACHE.clear()

    def test_front_repair_is_opt_in_to_preserve_single_llm_budget(self):
        old_flag = os.environ.get("XAI_LLM_ENABLE_FRONT_REPAIR")
        try:
            os.environ.pop("XAI_LLM_ENABLE_FRONT_REPAIR", None)
            self.assertFalse(_front_repair_enabled())
            os.environ["XAI_LLM_ENABLE_FRONT_REPAIR"] = "1"
            self.assertTrue(_front_repair_enabled())
            os.environ["XAI_LLM_ENABLE_FRONT_REPAIR"] = "0"
            self.assertFalse(_front_repair_enabled())
        finally:
            if old_flag is None:
                os.environ.pop("XAI_LLM_ENABLE_FRONT_REPAIR", None)
            else:
                os.environ["XAI_LLM_ENABLE_FRONT_REPAIR"] = old_flag

    def test_front_repair_patch_is_validated_and_general(self):
        front = process_question_front(
            "At point P, what is the direction of the net electric force on a test charge?"
        )
        patch = {
            "status": "ok",
            "target_overrides": [
                {
                    "goal_id": "goal:1",
                    "text": "the direction of the net electric force on a test charge",
                    "dimension": "force",
                    "evidence": "direction of the net electric force",
                }
            ],
            "relation_hints": [
                {
                    "relation_type": "geometry",
                    "qualifier": "collinear",
                    "evidence": "At point P",
                }
            ],
            "symbol_dimension_overrides": [],
            "notes": ["front patch"],
        }
        patched, trace = _apply_front_repair_patch(front, patch)
        self.assertTrue(trace["applied"], trace)
        self.assertTrue(any(item["kind"] == "target" for item in trace["accepted"]))
        self.assertTrue(any(relation["qualifier"] == "collinear" for relation in patched["relations"]))
        self.assertTrue(patched["trace"]["semantic_parser"]["llm_front_patch_applied"])

        unsafe_patch = {
            "status": "ok",
            "target_overrides": [{"dimension": "force", "evidence": "force", "value": 2}],
        }
        unsafe_front, unsafe_trace = _apply_front_repair_patch(front, unsafe_patch)
        self.assertIs(unsafe_front, front)
        self.assertFalse(unsafe_trace["applied"])
        self.assertEqual(unsafe_trace["reason"], "forbidden_payload_in_front_patch")

    def test_registry_does_not_keep_dataset_shaped_pattern_cards(self):
        audit = audit_registry_generalization()
        self.assertTrue(audit["ok"], audit["issues"])

    def test_runtime_source_does_not_embed_dataset_row_or_problem_instances(self):
        root = Path(__file__).resolve().parents[1]
        paths = list((root / "src" / "xai_pipeline").rglob("*.py")) + [
            root / "manual_question_test.py",
            root / "run_50_dataset_llm_cot.py",
        ]
        row_id_pattern = re.compile(r"\b(?:TD|DDT|THCB|LD|CH|NL|DT|CHLT)\d{3,}\b")
        forbidden_phrases = (
            "TA charge q = -1 μC",
            "q1 = q2 = q3 = 1.6 × 10^-19 C",
            "Three electric charges, q1 = q2 = q3 = 1.6e-19 C",
        )
        bad = []
        for path in paths:
            text = path.read_text(encoding="utf-8")
            if row_id_pattern.search(text):
                bad.append((str(path.relative_to(root)), "dataset_row_id"))
            for phrase in forbidden_phrases:
                if phrase in text:
                    bad.append((str(path.relative_to(root)), f"embedded_problem:{phrase[:24]}"))
        self.assertEqual(bad, [])

    def test_knowledge_graph_generalizes_repeated_charge_geometry(self):
        front = process_question_front(
            "Two equal charges q = 2 μC occupy two corners of a regular triangle of side a = 0.1 m, "
            "and a charge q0 = -1 μC is at the last corner. Find the resultant force on q0."
        )
        routed = route(front)
        graph = build_constraint_graph(front, routed)
        selected = select_minimal_equation_subset(front, routed)
        self.assertEqual(routed.task_type, "coulomb_force")
        self.assertEqual(infer_target_dimensions(front), ["force"])
        self.assertIn("coulomb_force_triangle_sides", graph.selected_formula_ids)
        self.assertEqual(selected.formula_ids, ["coulomb_force_triangle_sides"])
        self.assertNotIn("coulomb_force", selected.formula_ids)
        self.assertTrue(
            any(variable.get("derived_by") == "multiplicity:repeated_equal_charges" for variable in graph.variables)
        )

    def test_knowledge_target_dimensions_ignore_role_objects(self):
        front = process_question_front(
            "A test charge is placed at a point whose distances to charges q1 = +2 μC and q2 = -3 μC "
            "are 3 cm and 4 cm, respectively. The two charges are fixed and separated by 7 cm. "
            "What is the direction of the net electric force acting on the test charge?"
        )
        routed = route(front)
        selected = select_minimal_equation_subset(front, routed)
        self.assertEqual(infer_target_dimensions(front), ["force"])
        self.assertEqual(routed.task_type, "coulomb_force")
        self.assertIn("coulomb_force_direction_superposition", selected.formula_ids)

    def test_planning_dispatches_graph_selected_spatial_formulas_structurally(self):
        front = process_question_front(
            "Two equal charges q = 2 μC occupy two corners of a regular triangle of side a = 0.1 m, "
            "and a charge q0 = -1 μC is at the last corner. Find the resultant force on q0."
        )
        routed = route(front)
        selected = select_minimal_equation_subset(front, routed)
        plan = build_deterministic_solve_plan(front, routed, selected)
        compiled = compile_solve_plan(plan.to_dict(), front, routed, selected)
        steps = plan.to_dict()["steps"]
        self.assertEqual([step["operation"] for step in steps[:2]], ["construct_geometry", "compute_pairwise_force"])
        self.assertEqual(steps[0]["geometry_constructor_id"], "equilateral_triangle_vertex")
        self.assertEqual(steps[1]["formula_id"], "coulomb_force_triangle_sides")
        self.assertEqual(compiled.preferred_engine_order[0], "spatial")

    def test_plan_compiler_prefers_spatial_by_formula_branch_not_wording(self):
        front = process_question_front(
            "A probe charge is placed at a point whose distances to charges q1 = +2 μC and q2 = -3 μC "
            "are 3 cm and 4 cm, respectively. The two charges are separated by 7 cm. Determine the "
            "direction of the resultant force on the probe charge."
        )
        routed = route(front)
        selected = select_minimal_equation_subset(front, routed)
        plan = {
            "status": "ok",
            "task_type": "coulomb_force",
            "answer_type": "conceptual",
            "targets": [{"id": "goal:1", "quantity": "force", "text": "direction of the resultant force"}],
            "assumptions": [],
            "steps": [
                {
                    "step_id": "s1",
                    "operation": "apply_formula",
                    "formula_id": "coulomb_force_direction_superposition",
                    "principle_id": "coulomb_core",
                    "inputs": {"facts": "formal_ir"},
                    "output": "goal:1",
                    "depends_on": [],
                    "public_cot": "Resolve accepted vector relation with deterministic geometry.",
                }
            ],
            "output_format": {"format_kind": "conceptual_text", "ordered_targets": ["goal:1"]},
            "source": "test",
            "notes": [],
        }
        compiled = compile_solve_plan(plan, front, routed, selected)
        self.assertTrue(compiled.ok, compiled.issues)
        self.assertEqual(compiled.preferred_engine_order[0], "spatial")

    def test_knowledge_registry_covers_general_inverse_families(self):
        required = {
            "coulomb_distance_from_force",
            "electric_field_point_charge",
            "electric_field_point_distance",
            "electric_field_symbolic_superposition",
            "parallel_plate_area_from_capacitance",
            "parallel_plate_distance_from_capacitance",
            "inductance_from_inductive_reactance",
            "capacitance_from_capacitive_reactance",
            "angular_frequency_lc",
            "magnetic_field_from_flux_area",
            "solenoid_current_from_field",
            "magnetic_field_long_wire_current",
            "magnetic_field_loop_current",
            "faraday_flux_change",
            "self_inductance_from_emf",
            "wire_current_from_force",
            "lorentz_velocity_from_force",
        }
        self.assertTrue(required <= set(FORMULA_IDS))
        catalog = formula_logic_catalog()
        self.assertEqual(catalog["uncovered_formula_ids"], [])
        self.assertIn("magnetism_induction", catalog["families"])
        self.assertIn("coulomb_and_fields", catalog["families"])
        self.assertIn("lc_rlc", catalog["families"])

    def test_shared_language_factors_are_structural_not_phrase_specific(self):
        self.assertEqual(extract_change_factor("frequency is doubled"), 2.0)
        self.assertEqual(extract_change_factor("frequency is reduced by a factor of 4"), 0.25)
        self.assertEqual(extract_change_factor("distance changes to 3 times its initial value"), 3.0)
        self.assertTrue(has_frequency_transform_cue("angular frequency is scaled by a factor of 2"))
        front = process_question_front(
            "Two equal charges are at two vertices of a regular triangle with side length a = 0.1 m."
        )
        self.assertTrue(any(relation["qualifier"] == "equilateral_triangle" for relation in front["relations"]))

    def test_pipeline_solves_core_numeric_cases_and_converts_target_units(self):
        cases = [
            ("A resistor R = 10 ohm has voltage U = 20 V. Find current.", "2 A", "ohm_current"),
            ("A capacitor C = 100 uF has voltage U = 30 V. Find energy.", "0.045 J", "capacitor_energy_voltage"),
            ("Two charges q1 = 2e-6 C and q2 = 3e-6 C are separated by r = 10 cm. Calculate force.", "5.4 N", "coulomb_force"),
            ("Two point charges qM = 2 μC and qN = 3 μC are separated by MN = 10 cm. Calculate force.", "5.4 N", "coulomb_force"),
            ("A point charge q = 2 nC is in air. Find electric field at r = 3 cm.", "20000 V/m", "electric_field_point"),
        ]
        bad = []
        for question, answer, formula_id in cases:
            result = handle_request({"question": question})
            if not result["verifier"]["ok"] or result["answer"] != answer or result["solver"]["formula_id"] != formula_id:
                bad.append((question, result["answer"], result["solver"], result["verifier"]))
        self.assertEqual(bad, [])

        arbitrary_labels = handle_request(
            {"question": "Two point charges qM = 2 μC and qN = 3 μC are separated by MN = 10 cm. Calculate force."}
        )
        self.assertEqual(
            arbitrary_labels["solver"]["trace"]["binding_audit"]["q1"]["policy"],
            "ordered_labeled_repeated_dimension",
        )
        self.assertEqual(
            arbitrary_labels["front"]["canonical_structures"]["component_groups"]["charge"][0]["canonical_id"],
            "charge:1",
        )

        converted = handle_request({"question": "Calculate the capacitance of a capacitor that has 200 μJ of stored energy when the voltage across it is 10 V. Answer in μF."})
        self.assertTrue(converted["verifier"]["ok"], converted["verifier"]["issues"])
        self.assertEqual(converted["answer"], "4 μF")
        self.assertTrue(converted["solve_plan"]["ok"], converted["solve_plan"]["issues"])
        self.assertEqual(converted["solve_plan"]["plan"]["source"], "deterministic")
        self.assertEqual(converted["solve_plan"]["plan"]["output_format"]["format_kind"], "numeric_scalar")
        self.assertIn("plan_step", {node["type"] for node in converted["trace"]["proof_dag"]["nodes"]})
        self.assertTrue(converted["trace"]["target_unit_conversion"]["applied"])
        self.assertTrue(converted["trace"]["proof_dag"]["nodes"])
        self.assertTrue(converted["trace"]["proof_dag"]["edges"])
        self.assertTrue(converted["trace"]["proof_dag"]["certificate"]["certified"])
        self.assertEqual(converted["trace"]["proof_dag"]["certificate"]["algorithm"], "sha256-json-v1")
        proof_nodes = converted["trace"]["proof_dag"]["nodes"]
        self.assertIn("goal", {node["type"] for node in proof_nodes})
        self.assertIn("constraint", {node["type"] for node in proof_nodes})
        self.assertEqual(
            next(node for node in proof_nodes if node["type"] == "result")["answer"],
            converted["answer"],
        )
        self.assertIn("constraint graph", converted["explanation"])
        self.assertIn("proof graph", converted["explanation"])
        self.assertIn("versions", converted["metadata"])
        self.assertEqual(
            converted["metadata"]["xai_policy"]["explanation_source"],
            "proof_dag_and_execution_trace",
        )
        self.assertFalse(converted["metadata"]["xai_policy"]["llm_free_form_reasoning_used"])
        self.assertEqual(
            converted["metadata"]["versions"]["proof_dag_version"],
            "proof-dag-v3-structural-constraint-certificate",
        )
        self.assertEqual(
            converted["metadata"]["versions"]["explanation_version"],
            "trace-explanation-v2-structural-proof",
        )
        self.assertTrue(any("Apply registry formula" in step for step in converted["cot"]))
        proof_step = next(node for node in proof_nodes if node["type"] == "plan_step")
        self.assertIn("public_cot", proof_step)

    def test_pipeline_solves_general_inverse_coverage_cases(self):
        cases = [
            (
                "At distance r = 0.3 m from a point charge, the electric field E = 2e5 N/C. Find the charge.",
                "2e-06 C",
                "electric_field_point_charge",
            ),
            (
                "An inductor has inductive reactance XL = 31.4 Ω at frequency f = 50 Hz. Find inductance.",
                "0.0999493 H",
                "inductance_from_inductive_reactance",
            ),
            (
                "A long solenoid has magnetic field B = 0.004 T and turn density n = 1000 turns/m. Find current.",
                "3.1831 A",
                "solenoid_current_from_field",
            ),
            (
                "A loop has magnetic flux Phi = 0.02 Wb through area A = 0.5 m^2. Find magnetic field.",
                "0.04 T",
                "magnetic_field_from_flux_area",
            ),
            (
                "Two point charges q1 = 2 μC and q2 = 3 μC exert a force F = 5.4 N. Find their separation distance.",
                "0.1 m",
                "coulomb_distance_from_force",
            ),
            (
                "A long straight wire produces magnetic field B = 2e-5 T at distance r = 0.1 m. Find current.",
                "10 A",
                "magnetic_field_long_wire_current",
            ),
            (
                "A circular loop has magnetic field B = 2e-5 T at radius r = 0.1 m. Find current.",
                "3.1831 A",
                "magnetic_field_loop_current",
            ),
        ]
        bad = []
        for question, answer, formula_id in cases:
            result = handle_request({"question": question}, timeout_seconds=-1)
            if not result["verifier"]["ok"] or result["answer"] != answer or result["solver"]["formula_id"] != formula_id:
                bad.append((question, result["answer"], result["solver"]["formula_id"], result["verifier"]["issues"]))
        self.assertEqual(bad, [])

    def test_verifier_rejects_direct_contradictory_quantity_values(self):
        response = handle_request({"question": "A resistor has R = 5 Ω and R = 10 Ω. Find current if U = 20 V."})
        self.assertFalse(response["verifier"]["ok"])
        self.assertEqual(response["answer"], "Uncertain")
        self.assertTrue(any(issue.startswith("contradictory_quantity_value:r") for issue in response["verifier"]["issues"]))
        self.assertEqual(response["verifier"]["conflicts"][0]["type"], "contradictory_quantity_value")

    def test_semantic_frontend_tracks_state_event_and_entity_grounding(self):
        normalized = normalize_question(
            "A resistor initially has R = 5 Ω. After heating, R = 10 Ω. Find current if U = 20 V."
        )
        by_raw = {quantity.raw_text: quantity for quantity in normalized.quantities}
        self.assertEqual(by_raw["R = 5 Ω"].entity_id, "resistor:r")
        self.assertEqual(by_raw["R = 5 Ω"].state_id, "state:initial")
        self.assertEqual(by_raw["R = 10 Ω"].state_id, "state:final")

        response = handle_request(
            {"question": "A resistor initially has R = 5 Ω. After heating, R = 10 Ω. Find current if U = 20 V."}
        )
        self.assertTrue(response["verifier"]["ok"], response["verifier"]["issues"])
        self.assertEqual(response["verifier"]["conflicts"], [])

    def test_logic_engine_forward_chains_structured_derived_facts(self):
        front = process_question_front(
            "A series RLC circuit is at resonance. The capacitor is disconnected from the battery. Find current."
        )
        fact_ids = {fact["fact_id"] for fact in front["derived_facts"]}
        self.assertIn("topology.series_current_equal", fact_ids)
        self.assertIn("state.rlc_resonance_xl_equals_xc", fact_ids)
        self.assertIn("conservation.isolated_capacitor_charge", fact_ids)
        self.assertIn("derived_fact_ids", front["trace"]["logic_engine"])

    def test_multi_path_verification_audits_redundant_formulas(self):
        response = handle_request(
            {"question": "A resistor R = 10 Ω has voltage U = 20 V and current I = 2 A. Find power."}
        )
        self.assertTrue(response["verifier"]["ok"], response["verifier"]["issues"])
        audit = response["verifier"]["audit"]["multi_path"]
        self.assertEqual(audit["status"], "agreement")
        self.assertGreaterEqual(audit["path_count"], 3)

    def test_compound_relations_are_not_converted_to_hidden_unit_quantities(self):
        front = process_question_front(
            "An RLC circuit satisfies LCω² = 1. R1 = 20 Ω, U = 100 V, P = 142.86 W. Find R2."
        )
        self.assertFalse(any(quantity["symbol"] == "LCω2" for quantity in front["quantities"]))
        self.assertFalse(any(constant["symbol"] == "LCω2" for constant in front["numeric_constants"]))
        self.assertTrue(any(relation["lhs"] == "LCω2" for relation in front["symbolic_relations"]))

    def test_ambiguous_circuit_binding_abstains_without_topology(self):
        response = handle_request(
            {"question": "Two resistors R1 = 10 Ω and R2 = 20 Ω are in a circuit. Find current if U = 30 V."}
        )
        self.assertFalse(response["verifier"]["ok"])
        self.assertEqual(response["answer"], "Uncertain")
        self.assertFalse(response["solver"]["solved"])
        self.assertTrue(response["front"]["topology_graph"]["is_complex"])

    def test_canonical_topology_solver_handles_series_parallel_without_guessing(self):
        series = handle_request(
            {"question": "Two resistors R1 = 10 Ω and R2 = 20 Ω are connected in series. Find equivalent resistance."}
        )
        self.assertTrue(series["verifier"]["ok"], series["verifier"]["issues"])
        self.assertEqual(series["answer"], "30 Ω")
        self.assertEqual(series["solver"]["formula_id"], "series_resistance_equivalent")
        self.assertEqual(series["front"]["topology_graph"]["canonical_form"], "series_topology")

        parallel = handle_request(
            {"question": "Two resistors R1 = 10 Ω and R2 = 20 Ω are connected in parallel. Find equivalent resistance."}
        )
        self.assertTrue(parallel["verifier"]["ok"], parallel["verifier"]["issues"])
        self.assertEqual(parallel["answer"], "6.66667 Ω")
        self.assertEqual(parallel["solver"]["formula_id"], "parallel_resistance_equivalent")

        current = handle_request(
            {"question": "Two resistors R1 = 10 Ω and R2 = 20 Ω are connected in series. Voltage U = 30 V. Find current."}
        )
        self.assertTrue(current["verifier"]["ok"], current["verifier"]["issues"])
        self.assertEqual(current["answer"], "1 A")
        self.assertEqual(current["solver"]["formula_id"], "topology_ohm_current_series_resistance")
        self.assertEqual(current["solver"]["trace"]["binding_audit"]["policy"], "canonical_topology_only")
        self.assertTrue(any(node["type"] == "component_fact" for node in current["trace"]["proof_dag"]["nodes"]))

        capacitance = handle_request(
            {"question": "Two capacitors C1 = 2 μF and C2 = 3 μF are connected in parallel. Find equivalent capacitance in μF."}
        )
        self.assertTrue(capacitance["verifier"]["ok"], capacitance["verifier"]["issues"])
        self.assertEqual(capacitance["answer"], "5 μF")
        self.assertEqual(capacitance["solver"]["formula_id"], "parallel_capacitance_equivalent")

    def test_conceptual_logic_engine_answers_only_grounded_facts(self):
        same_current = handle_request(
            {"question": "In a series circuit, is the current through each resistor the same?"}
        )
        self.assertTrue(same_current["verifier"]["ok"], same_current["verifier"]["issues"])
        self.assertEqual(same_current["answer"], "Yes")
        self.assertEqual(same_current["solver"]["formula_id"], "yes_no_direct")
        self.assertEqual(same_current["solver"]["trace"]["fact_id"], "topology.series_current_equal")

        si_unit = handle_request({"question": "What is the SI unit of capacitance?"})
        self.assertTrue(si_unit["verifier"]["ok"], si_unit["verifier"]["issues"])
        self.assertEqual(si_unit["answer"], "The SI unit of capacitance is farad (F).")
        self.assertEqual(si_unit["solver"]["trace"]["rule"], "si_unit_lookup")

    def test_ordered_binding_is_allowed_for_registry_defined_repeated_dimensions(self):
        response = handle_request(
            {
                "question": (
                    "A series RLC circuit has resistance 10 ohm, inductive reactance 30 ohm, "
                    "and capacitive reactance 20 ohm. Calculate impedance."
                )
            }
        )
        self.assertTrue(response["verifier"]["ok"], response["verifier"]["issues"])
        self.assertEqual(response["solver"]["formula_id"], "rlc_impedance")
        self.assertEqual(response["solver"]["trace"]["binding_audit"]["XL"]["policy"], "ordered_repeated_dimension")

        hidden_si_response = handle_request(
            {"question": "A series RLC circuit uses SI values R = 10, XL = 30, and XC = 20. Calculate impedance."}
        )
        self.assertTrue(hidden_si_response["verifier"]["ok"], hidden_si_response["verifier"]["issues"])
        self.assertEqual(hidden_si_response["solver"]["formula_id"], "rlc_impedance")
        self.assertEqual(hidden_si_response["answer"], "14.1421 Ω")
        self.assertTrue(
            all(
                quantity["raw_unit"] == "implicit_base_SI"
                for quantity in hidden_si_response["front"]["quantities"]
                if quantity["symbol"] in {"R", "XL", "XC"}
            )
        )

    def test_spatial_engine_is_deterministic_and_symbolic_geometry_generalizes(self):
        geometry = execute_coulomb_force_superposition(
            "right_isosceles_triangle_vertex",
            {"leg": 0.12},
            [{"point": "B", "charge_c": 4e-6}, {"point": "C", "charge_c": 4e-6}],
            {"point": "A", "charge_c": 4e-6},
        )
        self.assertTrue(geometry.ok)
        self.assertAlmostEqual(geometry.value, 14.1421356, places=5)
        self.assertIn("coordinates", geometry.trace)
        self.assertIn("C", build_template_coordinates("triangle_sides", {"ab": 0.20, "ac": 0.12, "bc": 0.16}))

        question = (
            "Two charges, q1 = q2 = q, are placed at points A and B with AB = 2a. "
            "Point M is on the perpendicular bisector at distance h. Determine the electric field at M."
        )
        response = handle_request({"question": question})
        self.assertTrue(response["verifier"]["ok"], response["verifier"]["issues"])
        self.assertEqual(response["solver"]["formula_id"], "electric_field_symbolic_superposition")
        self.assertIn("E_M", response["answer"])
        self.assertTrue(response["solver"]["trace"]["geometry"]["recoverable"])

        field = handle_request(
            {
                "question": (
                    "Two charges q1 = 2 uC and q2 = 3 uC are placed at the endpoints "
                    "of a segment AB = 10 cm. Find the electric field at the midpoint."
                )
            }
        )
        self.assertTrue(field["verifier"]["ok"], field["verifier"]["issues"])
        self.assertEqual(field["solver"]["formula_id"], "electric_field_two_charge_superposition")
        self.assertEqual(field["solver"]["trace"]["stage"], "spatial_vector_engine")
        self.assertIn("geometry_engine", field["solver"]["trace"])

        equilateral_force = handle_request(
            {
                "question": (
                    "Two identical charges q = +2 μC are placed at two vertices of an equilateral triangle "
                    "with side length a = 0.1 m. A charge q′ = -1 μC is placed at the remaining vertex. "
                    "Calculate the net electric force acting on q′."
                )
            }
        )
        self.assertTrue(equilateral_force["verifier"]["ok"], equilateral_force["verifier"]["issues"])
        self.assertEqual(equilateral_force["solver"]["formula_id"], "coulomb_force_triangle_sides")
        self.assertEqual(equilateral_force["solver"]["trace"]["stage"], "spatial_vector_engine")
        self.assertIn("coulomb_force_triangle_sides", equilateral_force["constraint_graph"]["formula_ids"])
        self.assertAlmostEqual(equilateral_force["solver"]["value"], 3.1176914536, places=10)
        self.assertEqual(
            equilateral_force["solver"]["trace"]["geometry_audit"]["status"],
            "two_identical_sources_at_other_vertices",
        )

        collinear_equal_sources = handle_request(
            {
                "question": (
                    "Two identical charges q = 2 μC are fixed at endpoints A and B of a line segment AB = 10 cm. "
                    "A test charge q0 = -1 μC is placed on the line, 2 cm from A. Find the net electric force on q0."
                )
            }
        )
        self.assertTrue(collinear_equal_sources["verifier"]["ok"], collinear_equal_sources["verifier"]["issues"])
        self.assertEqual(collinear_equal_sources["solver"]["formula_id"], "coulomb_force_triangle_sides")
        self.assertAlmostEqual(collinear_equal_sources["solver"]["value"], 42.1875, places=10)
        self.assertEqual(
            collinear_equal_sources["solver"]["trace"]["geometry_audit"]["charge_binding_policy"],
            "one_target_charge_plus_two_equal_source_charges",
        )
        self.assertEqual(collinear_equal_sources["solver"]["trace"]["geometry_audit"]["source_charge_index"], 0)
        self.assertEqual(collinear_equal_sources["solver"]["trace"]["geometry_audit"]["target_charge_index"], 1)

        opposite_side_sources_question = (
            "A charge q = -1 μC is attracted by two +1 μC charges. These two positive charges are located "
            "on opposite sides of q, along the same straight line passing through q, at distances of 5 cm "
            "and 12 cm respectively from q. Calculate the magnitude of the net electric force acting on q."
        )
        front = process_question_front(opposite_side_sources_question)
        self.assertTrue(any(relation["qualifier"] == "collinear" for relation in front["relations"]))
        graph_route = route(front)
        graph_selection = select_minimal_equation_subset(front, graph_route)
        self.assertEqual(graph_selection.formula_ids[0], "coulomb_force_triangle_sides")
        plan_cards = local_llm._plan_cards_for_context(front, graph_route, graph_selection)
        self.assertEqual(plan_cards[0]["plan_template_id"], "spatial_pairwise_force")
        self.assertEqual(plan_cards[0]["geometry_template_id"], "two_charges_collinear")
        llm_plan = local_llm._complete_llm_solve_plan_payload(
            {"status": "ok", "plan_card_id": "p1"},
            front,
            graph_route,
            graph_selection,
        )
        self.assertEqual(llm_plan["steps"][0]["geometry_constructor_id"], "two_charges_collinear")
        self.assertEqual(llm_plan["steps"][1]["formula_id"], "coulomb_force_triangle_sides")

        square_center_question = (
            "Four identical positive charges are placed at the vertices of a square. "
            "What is the electric field at the center of the square?"
        )
        square_front = process_question_front(square_center_question)
        square_route = route(square_front)
        square_selection = select_minimal_equation_subset(square_front, square_route)
        square_cards = local_llm._plan_cards_for_context(square_front, square_route, square_selection)
        self.assertEqual(square_cards[0]["plan_template_id"], "spatial_vector_resolution")
        self.assertEqual(square_cards[0]["formula_id"], "electric_field_symmetric_zero")
        self.assertEqual(square_cards[0]["geometry_template_id"], "square_vertex_field")
        square_plan = local_llm._complete_llm_solve_plan_payload(
            {"status": "ok", "plan_card_id": "p1"},
            square_front,
            square_route,
            square_selection,
        )
        self.assertEqual(
            [step["operation"] for step in square_plan["steps"]],
            ["construct_geometry", "resolve_vector_components"],
        )

        structural_square_front = process_question_front(
            "At the center of square WXYZ, equal charges qW = qY = q are placed at opposite corners W and Y. "
            "What charge must be placed at X so that the electric field at the center is zero?"
        )
        structural_square_front["canonical_question"] = "What charge makes the electric field zero at the observation point?"
        structural_square_route = route(structural_square_front)
        structural_square_selection = select_minimal_equation_subset(structural_square_front, structural_square_route)
        structural_geometry = local_llm._compact_geometry(structural_square_front)
        self.assertTrue(structural_geometry["squares"])
        structural_cards = local_llm._plan_cards_for_context(
            structural_square_front,
            structural_square_route,
            structural_square_selection,
        )
        self.assertEqual(structural_cards[0]["formula_id"], "electric_field_square_center_cancel_charge")
        self.assertEqual(structural_cards[0]["geometry_template_id"], "square_vertex_field")

        def fake_plan_card_choice(front_payload, route_result=None, graph_selection=None, enable_llm=False):
            raw = {"status": "ok", "plan_card_id": "p1"}
            plan = local_llm._complete_llm_solve_plan_payload(raw, front_payload, route_result, graph_selection)
            return plan, {
                "stage": "local_llm_solve_plan",
                "used": True,
                "applied": False,
                "reason": "proposal_generated_unvalidated",
                "generation": {"ok": True, "json": raw, "raw_text": json.dumps(raw)},
                "completion": local_llm._completion_summary(raw, plan),
                "plan_summary": local_llm.plan_summary(plan),
            }

        with patch("xai_pipeline.core.pipeline.propose_solve_plan_if_enabled", side_effect=fake_plan_card_choice):
            opposite_side_sources = handle_request(
                {"question": opposite_side_sources_question},
                enable_llm=True,
                planning_mode="llm_required",
                timeout_seconds=-1,
            )
        self.assertTrue(opposite_side_sources["verifier"]["ok"], opposite_side_sources["verifier"]["issues"])
        self.assertEqual(opposite_side_sources["solve_plan"]["plan"]["source"], "local_llm")
        self.assertEqual(opposite_side_sources["solver"]["formula_id"], "coulomb_force_triangle_sides")
        self.assertAlmostEqual(opposite_side_sources["solver"]["value"], 2.975, places=10)

        right_triangle_force = handle_request(
            {
                "question": (
                    "Three electric charges are placed at three fixed points, forming a right-angled triangle ABC "
                    "(right-angled at A), where AB = 4 m and BC = 5 m. The charges are qA = 5.0 μC, "
                    "qB = -5.0 μC, and qC = 4.0 μC, respectively. Find the net electric force acting on the charge at A."
                )
            }
        )
        self.assertTrue(right_triangle_force["verifier"]["ok"], right_triangle_force["verifier"]["issues"])
        self.assertEqual(right_triangle_force["solver"]["formula_id"], "coulomb_force_triangle_sides")
        self.assertIn("coulomb_force_triangle_sides", right_triangle_force["constraint_graph"]["formula_ids"])
        self.assertAlmostEqual(right_triangle_force["solver"]["value"], 0.0244490062, places=10)
        self.assertEqual(
            right_triangle_force["solver"]["trace"]["geometry_audit"]["status"],
            "derived_missing_leg",
        )
        self.assertEqual(
            right_triangle_force["solver"]["trace"]["geometry_audit"]["derived_sides"]["ac"],
            3.0,
        )

        renamed_same_structure = handle_request(
            {
                "question": (
                    "Three electric charges are placed at three fixed points, forming a right-angled triangle MNQ "
                    "(right-angled at M), where MN = 4 m and NQ = 5 m. The charges are qM = 5.0 μC, "
                    "qN = -5.0 μC, and qQ = 4.0 μC, respectively. Find the net electric force acting on the charge at M."
                )
            }
        )
        self.assertTrue(renamed_same_structure["verifier"]["ok"], renamed_same_structure["verifier"]["issues"])
        self.assertEqual(renamed_same_structure["solver"]["formula_id"], "coulomb_force_triangle_sides")
        self.assertIn("coulomb_force_triangle_sides", renamed_same_structure["constraint_graph"]["formula_ids"])
        self.assertAlmostEqual(renamed_same_structure["solver"]["value"], 0.0244490062, places=10)
        self.assertEqual(
            renamed_same_structure["solver"]["trace"]["geometry_audit"]["point_label_mapping"],
            {"M": "A", "N": "B", "Q": "C"},
        )

        renamed_leg_leg = handle_request(
            {
                "question": (
                    "Three electric charges are placed at three fixed points, forming a right-angled triangle MNQ "
                    "(right-angled at M), where MN = 4 m and MQ = 5 m. The charges are qM = 5.0 μC, "
                    "qN = -5.0 μC, and qQ = 4.0 μC, respectively. Find the net electric force acting on the charge at M."
                )
            }
        )
        self.assertTrue(renamed_leg_leg["verifier"]["ok"], renamed_leg_leg["verifier"]["issues"])
        self.assertEqual(renamed_leg_leg["solver"]["formula_id"], "coulomb_force_triangle_sides")
        self.assertAlmostEqual(renamed_leg_leg["solver"]["value"], 0.0157985413, places=10)
        self.assertEqual(
            renamed_leg_leg["solver"]["trace"]["geometry_audit"]["status"],
            "derived_hypotenuse",
        )

        renamed_prefix_collision = handle_request(
            {
                "question": (
                    "Three electric charges are placed at three fixed points, forming a right-angled triangle PQR "
                    "(right-angled at P), where PQ = 4 m and QR = 5 m. The charges are qP = 5.0 μC, "
                    "qQ = -5.0 μC, and qR = 4.0 μC, respectively. Find the net electric force acting on the charge at P."
                )
            }
        )
        self.assertTrue(renamed_prefix_collision["verifier"]["ok"], renamed_prefix_collision["verifier"]["issues"])
        self.assertAlmostEqual(renamed_prefix_collision["solver"]["value"], 0.0244490062, places=10)
        length_facts = [
            quantity for quantity in renamed_prefix_collision["front"]["quantities"] if quantity["dimension"] == "length"
        ]
        self.assertTrue(all(quantity["entity_id"] is None for quantity in length_facts))

        symbolic_field = handle_request(
            {
                "question": (
                    "At the three vertices of a right isosceles triangle ABC, with AB = AC = a, "
                    "three positive charges qA = qB = q and qC = 2q are placed in a vacuum. "
                    "What is the expression for the electric field intensity at H, which is the foot "
                    "of the altitude dropped from the right-angle vertex A to the hypotenuse BC?"
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(symbolic_field["verifier"]["ok"], symbolic_field["verifier"]["issues"])
        self.assertEqual(symbolic_field["solver"]["formula_id"], "electric_field_symbolic_superposition")
        self.assertEqual(symbolic_field["solver"]["trace"]["stage"], "symbolic_spatial_vector_engine")
        self.assertEqual(
            symbolic_field["solver"]["trace"]["compiled_geometry_case"],
            "right_isosceles_altitude_to_hypotenuse",
        )
        self.assertEqual(
            symbolic_field["answer"],
            "E_H = 2√2 k q/a^2, directed parallel to AB, from A toward B",
        )
        self.assertEqual(symbolic_field["answer_checker"]["trace"]["mode"], "verified_symbolic")
        self.assertEqual(
            symbolic_field["front"]["canonical_structures"]["geometry"]["triangles"][0]["canonical_right_angle_at"],
            "A",
        )
        self.assertIn(
            "electric_field_symbolic_superposition",
            symbolic_field["solve_plan"]["selected_formula_ids"],
        )

        symbolic_field_right_angle_not_first = handle_request(
            {
                "question": (
                    "At the three vertices of a right isosceles triangle ABC, right-angled at B, "
                    "with BA = BC = a, three positive charges qB = qA = q and qC = 2q are placed in a vacuum. "
                    "What is the expression for the electric field intensity at H, the foot of the altitude "
                    "dropped from B to the hypotenuse AC?"
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(
            symbolic_field_right_angle_not_first["verifier"]["ok"],
            symbolic_field_right_angle_not_first["verifier"]["issues"],
        )
        self.assertEqual(
            symbolic_field_right_angle_not_first["answer"],
            "E_H = 2√2 k q/a^2, directed parallel to BA, from B toward A",
        )
        self.assertEqual(
            symbolic_field_right_angle_not_first["front"]["canonical_structures"]["geometry"]["triangles"][0]["canonical_by_original"],
            {"B": "A", "A": "B", "C": "C"},
        )

        direction_only = handle_request(
            {
                "question": (
                    "A test charge is placed at a point whose distances to the two charges "
                    "q1 = +2 μC and q2 = -3 μC are 3 cm and 4 cm, respectively. "
                    "The two charges are fixed and separated by 7 cm. What is the direction "
                    "of the net electric force acting on the test charge?"
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(direction_only["verifier"]["ok"], direction_only["verifier"]["issues"])
        self.assertEqual(direction_only["answer"], "Toward q2")
        self.assertEqual(direction_only["front"]["answer_type_hint"], "conceptual")
        self.assertEqual(direction_only["solver"]["formula_id"], "coulomb_force_direction_superposition")
        self.assertEqual(
            direction_only["solver"]["trace"]["geometry_audit"]["status"],
            "proved_by_signed_1d_superposition",
        )
        self.assertEqual(
            direction_only["solve_plan"]["plan"]["output_format"]["format_kind"],
            "conceptual_text",
        )
        self.assertIsNone(direction_only["solve_plan"]["plan"]["targets"][0]["unit"])
        self.assertEqual(direction_only["solver"]["unit"], "-")
        self.assertEqual(direction_only["trace"]["proof_dag"]["nodes"][0]["target_unit"], "-")

        probe_direction = handle_request(
            {
                "question": (
                    "A probe charge is placed at a point whose distances to the two charges "
                    "q1 = +2 μC and q2 = -3 μC are 3 cm and 4 cm, respectively. "
                    "The two charges are fixed and separated by 7 cm. What is the direction "
                    "of the net electric force acting on the probe charge?"
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(probe_direction["verifier"]["ok"], probe_direction["verifier"]["issues"])
        self.assertEqual(probe_direction["answer"], "Toward q2")
        self.assertEqual(probe_direction["solver"]["formula_id"], "coulomb_force_direction_superposition")

        field_target_not_c = handle_request(
            {
                "question": (
                    "Two point charges qA = 2 μC and qC = 3 μC are placed at vertices A and C "
                    "of triangle ABC, where AB = 3 m, AC = 4 m, and BC = 5 m. Find the electric field at B."
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(field_target_not_c["verifier"]["ok"], field_target_not_c["verifier"]["issues"])
        self.assertEqual(field_target_not_c["solver"]["formula_id"], "electric_field_two_charge_triangle_sides")
        self.assertEqual(field_target_not_c["solver"]["trace"]["geometry_audit"]["target_point"], "B")
        self.assertEqual(
            field_target_not_c["solver"]["trace"]["geometry_audit"]["target_point_policy"],
            "field_target_from_goal_or_uncharged_vertex",
        )

        generic_triangle = handle_request(
            {
                "question": (
                    "Two charges, q1 = 6 × 10^-8 C and q2 = -6 × 10^-8 C, are placed at points A and B "
                    "in air, 8 cm apart. A third charge, q3 = 6 × 10^-8 C, is placed at point C, "
                    "with CA = 5 cm and CB = 3 cm. Determine the force acting on q3."
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(generic_triangle["verifier"]["ok"], generic_triangle["verifier"]["issues"])
        self.assertEqual(generic_triangle["solver"]["formula_id"], "coulomb_force_triangle_sides")
        self.assertAlmostEqual(generic_triangle["solver"]["value"], 0.04896, places=8)

        perpendicular_bisector = handle_request(
            {
                "question": (
                    "Two point charges q1 = 10^-8 C and q2 = -3×10^-8 C are placed in air at two points A and B, "
                    "8 cm apart. A point charge q = 10^-8 C is placed at point M, which is on the perpendicular "
                    "bisector of the line segment AB and 3 cm away from AB. What is the magnitude of the net "
                    "electric force exerted by q1 and q2 on q?"
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(perpendicular_bisector["verifier"]["ok"], perpendicular_bisector["verifier"]["issues"])
        self.assertEqual(perpendicular_bisector["solver"]["trace"]["geometry_audit"]["stage"], "perpendicular_bisector_force_completion")

        perpendicular_reordered = handle_request(
            {
                "question": (
                    "A point charge q = 10^-8 C is placed at point M on the perpendicular bisector of segment AB "
                    "and 3 cm away from AB. Two point charges q1 = 10^-8 C and q2 = -3×10^-8 C are placed "
                    "in air at points A and B, 8 cm apart. What is the magnitude of the net electric force "
                    "exerted by q1 and q2 on q?"
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(perpendicular_reordered["verifier"]["ok"], perpendicular_reordered["verifier"]["issues"])
        reordered_audit = perpendicular_reordered["solver"]["trace"]["geometry_audit"]
        self.assertEqual(reordered_audit["binding_policy"], "explicit_charge_point_roles_for_line_template")
        self.assertAlmostEqual(reordered_audit["separation_m"], 0.08)
        self.assertAlmostEqual(reordered_audit["height_m"], 0.03)

        square_symmetry = handle_request(
            {
                "question": (
                    "Four identical positive charges are placed at the vertices of a square. "
                    "What is the electric field at the center?"
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(square_symmetry["verifier"]["ok"], square_symmetry["verifier"]["issues"])
        self.assertEqual(square_symmetry["answer"], "0 V/m")
        self.assertEqual(square_symmetry["solver"]["formula_id"], "electric_field_symmetric_zero")
        self.assertEqual(square_symmetry["solver"]["trace"]["stage"], "symmetry_reduction_engine")

        square_vertex_balance = handle_request(
            {
                "question": (
                    "Given a square ABCD with side length a. Charges q1 = q3 = q are placed at A and C. "
                    "What charge must be placed at B so that the electric field at D is zero?"
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(square_vertex_balance["verifier"]["ok"], square_vertex_balance["verifier"]["issues"])
        self.assertEqual(square_vertex_balance["answer"], "qB = -2√2 q")
        vertex_audit = square_vertex_balance["solver"]["trace"]["geometry_audit"]
        self.assertEqual(vertex_audit["target_field_point"], "D")
        self.assertEqual(vertex_audit["unknown_charge_point"], "B")

        equilateral_symmetry = handle_request(
            {
                "question": (
                    "Three identical positive charges are placed at the vertices of an equilateral triangle. "
                    "What is the electric field at the centroid?"
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(equilateral_symmetry["verifier"]["ok"], equilateral_symmetry["verifier"]["issues"])
        self.assertEqual(equilateral_symmetry["answer"], "0 V/m")
        self.assertEqual(
            equilateral_symmetry["solver"]["trace"]["geometry_audit"]["symmetry_group"],
            "C3_equilateral_centroid",
        )

    def test_multi_output_cas_lite_and_uncertainty_are_bounded(self):
        multi = handle_request({"question": "A resistor R = 10 ohm has voltage U = 20 V. Find current and power."})
        self.assertTrue(multi["verifier"]["ok"], multi["verifier"]["issues"])
        self.assertEqual(multi["answer"], "2 A; 40 W")
        self.assertEqual(multi["solver"]["trace"]["stage"], "multi_output_orchestrator")
        self.assertEqual(len(multi["solver"]["value"]), 2)
        self.assertIn("uncertainty", multi["verifier"]["audit"])

        resultant = handle_request(
            {"question": "Two electric forces, each with a magnitude of 5 N, act at an angle of 60° to each other. What is the resultant force?"},
            timeout_seconds=-1,
        )
        self.assertTrue(resultant["verifier"]["ok"], resultant["verifier"]["issues"])
        self.assertEqual(resultant["solver"]["formula_id"], "resultant_two_forces")
        self.assertEqual(resultant["answer"], "8.66025 N")

        plan = {
            "symbolic_family": "registry_equation_graph",
            "formula_ids": ["capacitor_energy_voltage"],
            "principle_ids": ["capacitor_core"],
            "equations": ["W=0.5*0.0001*30**2"],
            "targets": [{"symbol": "W", "dimension": "energy", "unit": "J"}],
            "non_negative_target": True,
            "trace": {"stage": "unit_test_graph"},
        }
        solved = solve_algebraic_plan(plan, SimpleNamespace(task_type="capacitor_energy", confidence=0.8))
        self.assertTrue(solved.solved, solved.trace)
        self.assertEqual(solved.answer, "0.045 J")
        self.assertEqual(solved.trace["stage"], "algebraic_constraint_engine")
        self.assertTrue(all(item["ok"] for item in solved.trace["residuals"]))

        measurement = handle_request(
            {
                "question": (
                    "A current is measured as I = 0.25 A with uncertainty dI = 0.01 A. "
                    "What is the percentage uncertainty?"
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(measurement["verifier"]["ok"], measurement["verifier"]["issues"])
        self.assertEqual(measurement["answer"], "4 %")
        self.assertEqual(measurement["solver"]["formula_id"], "measurement_error_direct")

        rlc_changed = handle_request(
            {
                "question": (
                    "A series RLC circuit has resistance R = 10 Ω, inductive reactance XL = 30 Ω, "
                    "and capacitive reactance XC = 20 Ω at a frequency f. The voltage is U = 100 V. "
                    "If the frequency is doubled, find the current."
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(rlc_changed["verifier"]["ok"], rlc_changed["verifier"]["issues"])
        self.assertEqual(rlc_changed["solver"]["formula_id"], "rlc_current_from_rlcf_voltage")
        self.assertAlmostEqual(rlc_changed["solver"]["value"], 1.9611613514, places=10)
        self.assertEqual(rlc_changed["front"]["answer_type_hint"], "numeric")

        proportional = handle_request(
            {
                "question": "How does the Coulomb force change if the distance is doubled while the charges stay fixed?"
            },
            timeout_seconds=-1,
        )
        self.assertTrue(proportional["verifier"]["ok"], proportional["verifier"]["issues"])
        self.assertIn("one fourth", proportional["answer"])
        self.assertEqual(proportional["solver"]["trace"]["rule"], "coulomb_inverse_square_distance_factor")

        proportional_tripled = handle_request(
            {
                "question": "How does the Coulomb force change if the separation is tripled while the charges stay fixed?"
            },
            timeout_seconds=-1,
        )
        self.assertTrue(proportional_tripled["verifier"]["ok"], proportional_tripled["verifier"]["issues"])
        self.assertIn("one ninth", proportional_tripled["answer"])
        self.assertEqual(proportional_tripled["solver"]["trace"]["rule"], "coulomb_inverse_square_distance_factor")

        capacitor_factor = handle_request(
            {
                "question": (
                    "An isolated parallel-plate capacitor has its plate separation increased by a factor of 3. "
                    "Describe how capacitance, voltage, and stored energy change."
                )
            },
            timeout_seconds=-1,
        )
        self.assertTrue(capacitor_factor["verifier"]["ok"], capacitor_factor["verifier"]["issues"])
        self.assertIn("factor of 3", capacitor_factor["answer"])
        self.assertEqual(capacitor_factor["solver"]["trace"]["rule"], "isolated_capacitor_distance_factor")

    def test_local_llm_repair_boundary_fails_closed_without_runtime(self):
        readiness = check_local_llm_readiness()
        self.assertIn("ready", readiness)
        front = process_question_front("A resistor R = 10 Ω has voltage U = 20 V. Find current.")
        repaired, trace = repair_front_ir_once(front, SimpleNamespace(issues=["unit_test_issue"], conflicts=[]), enable_llm=False)
        self.assertEqual(repaired, front)
        self.assertFalse(trace["used"])
        self.assertEqual(trace["reason"], "disabled")

    def test_plan_and_solver_verifier_reject_untrusted_or_inconsistent_artifacts(self):
        front = process_question_front("A resistor R = 10 Ω has voltage U = 20 V. Find current.")
        plan = {
            "status": "ok",
            "task_type": "ohm_law",
            "answer_type": "numeric",
            "targets": [{"symbol": "I"}],
            "formula_ids": ["capacitor_energy_voltage"],
            "principle_ids": ["dc_circuit_core"],
            "geometry_template_ids": ["freehand_diagram"],
            "implicit_rule_ids": ["electron"],
            "solve_strategy": "freeform",
            "numeric_answer": 42,
            "conceptual_answer": "2 A",
        }
        result = validate_plan(plan, front)
        self.assertFalse(result.ok)
        self.assertIn("formula_task_mismatch:capacitor_energy_voltage:ohm_law", result.issues)
        self.assertIn("unknown_geometry_template_id:freehand_diagram", result.issues)
        self.assertIn("implicit_rule_not_triggered:electron", result.issues)
        self.assertIn("proposal_supplied_numeric_answer", result.issues)

        structured = {
            "status": "ok",
            "task_type": "ohm_law",
            "answer_type": "numeric",
            "targets": [{"id": "goal:1", "quantity": "current", "unit": "A"}],
            "steps": [
                {
                    "step_id": "s1",
                    "operation": "apply_formula",
                    "formula_id": "ohm_current",
                    "principle_id": "dc_circuit_core",
                    "inputs": {"dimensions": ["voltage", "resistance"]},
                    "output": "goal:1",
                    "depends_on": [],
                    "public_cot": "Apply registry formula ohm_current using SI-normalized inputs.",
                }
            ],
            "output_format": {
                "format_kind": "numeric_scalar",
                "ordered_targets": ["goal:1"],
                "preferred_unit": "A",
                "target_count": 1,
                "requires_unit": True,
            },
        }
        structured_result = validate_structured_solve_plan(structured, front, route(front))
        self.assertTrue(structured_result.ok, structured_result.issues)
        compiled_with_untrusted_graph = compile_solve_plan(
            structured,
            front,
            route(front),
            SimpleNamespace(formula_ids=["invented_graph_formula", "ohm_voltage"]),
        )
        self.assertTrue(compiled_with_untrusted_graph.ok, compiled_with_untrusted_graph.issues)
        self.assertIn("ohm_current", compiled_with_untrusted_graph.selected_formula_ids)
        self.assertIn("ohm_voltage", compiled_with_untrusted_graph.selected_formula_ids)
        self.assertNotIn("invented_graph_formula", compiled_with_untrusted_graph.selected_formula_ids)
        compiled_llm_plan = compile_solve_plan(
            {**structured, "source": "local_llm"},
            front,
            route(front),
            SimpleNamespace(formula_ids=["ohm_voltage"]),
        )
        self.assertTrue(compiled_llm_plan.ok, compiled_llm_plan.issues)
        self.assertEqual(compiled_llm_plan.selected_formula_ids, ["ohm_current"])
        self.assertEqual(
            structured_result.audit["forbidden_payload_policy"],
            "no_numeric_answers_no_code_no_coordinates_no_free_form_cot",
        )
        bad_output_format = {**structured, "output_format": {"format_kind": "conceptual_text", "ordered_targets": ["goal:1"]}}
        bad_output_format_result = validate_structured_solve_plan(bad_output_format, front, route(front))
        self.assertFalse(bad_output_format_result.ok)
        self.assertIn("answer_type_output_format_mismatch:numeric:conceptual_text:numeric_scalar", bad_output_format_result.issues)
        missing_cot = {**structured, "steps": [{key: value for key, value in structured["steps"][0].items() if key != "public_cot"}]}
        missing_cot_result = validate_structured_solve_plan(missing_cot, front, route(front))
        self.assertFalse(missing_cot_result.ok)
        self.assertIn("missing_public_cot:s1", missing_cot_result.issues)
        leaked_cot = {**structured, "steps": [{**structured["steps"][0], "public_cot": "The final answer is 2 A."}]}
        leaked_cot_result = validate_structured_solve_plan(leaked_cot, front, route(front))
        self.assertFalse(leaked_cot_result.ok)
        self.assertTrue(any(issue.startswith("invalid_public_cot:s1") for issue in leaked_cot_result.issues))
        leaked_note = {**structured, "notes": ["The final answer is 2 A."]}
        leaked_note_result = validate_structured_solve_plan(leaked_note, front, route(front))
        self.assertFalse(leaked_note_result.ok)
        self.assertIn("invalid_note:0:forbidden_text", leaked_note_result.issues)
        equation_note = {**structured, "notes": ["Use I = U/R before returning."]}
        equation_note_result = validate_structured_solve_plan(equation_note, front, route(front))
        self.assertFalse(equation_note_result.ok)
        self.assertIn("invalid_note:0:equation_text", equation_note_result.issues)
        mismatched_formula = {**structured, "steps": [{**structured["steps"][0], "formula_id": "capacitor_energy_voltage", "principle_id": "capacitor_core"}]}
        mismatched_result = validate_structured_solve_plan(mismatched_formula, front, route(front))
        self.assertFalse(mismatched_result.ok)
        self.assertIn("formula_task_mismatch:capacitor_energy_voltage:capacitor_energy:ohm_law", mismatched_result.issues)
        bad_structured = {**structured, "steps": [{**structured["steps"][0], "operation": "freeform_cot", "numeric_answer": 2}]}
        bad_result = validate_structured_solve_plan(bad_structured, front, route(front))
        self.assertFalse(bad_result.ok)
        self.assertIn("unknown_operation:freeform_cot", bad_result.issues)
        self.assertIn("forbidden_payload_in_step:s1", bad_result.issues)
        routed = route(front)
        selected = select_minimal_equation_subset(front, routed)
        bad_compiled = compile_solve_plan(bad_structured, front, routed, selected)
        packet = build_plan_error_packet(bad_compiled, front, routed, selected)
        self.assertEqual(packet["stage"], "plan_compiler")
        self.assertIn("free_form_cot", packet["forbidden_outputs"])
        repaired_plan, repair_trace = repair_solve_plan_once(front, packet, routed, selected, enable_llm=False)
        self.assertIsNone(repaired_plan)
        self.assertFalse(repair_trace["used"])
        self.assertEqual(repair_trace["stage"], "local_llm_solve_plan_repair")
        prompt = _solve_plan_prompt(front, routed, selected)
        self.assertIn("plan_contract", prompt)
        self.assertIn("You own the executable step DAG", prompt)
        self.assertIn("check_condition", prompt)

        route_result = SimpleNamespace(task_type="electric_field_point", confidence=0.9)
        solver_result = SimpleNamespace(
            solved=True,
            answer="5 V/m",
            value=5.0,
            unit="V/m",
            formula_id="electric_field_point",
            principle_id="electric_field_core",
            trace={"geometry_engine": {"components": {"x": 3.0, "y": 4.0, "magnitude": 6.0}, "value": 5.0}},
            confidence=0.9,
        )
        verified = verify_solver({"answer_type_hint": "numeric"}, route_result, solver_result)
        self.assertFalse(verified.ok)
        self.assertIn("vector_component_magnitude_mismatch", verified.issues)

    def test_trace_explanation_units_and_telemetry_are_core_owned(self):
        converted = convert_si_to_target(0.002, "length", "cm")
        self.assertTrue(converted.ok)
        self.assertAlmostEqual(converted.value, 0.2)

        solver = SolverResult(
            True,
            "2 A",
            2.0,
            "A",
            "ohm_current",
            "dc_circuit_core",
            ["Use Ohm's law."],
            {"expression": "I=U/R", "inputs": {"U": {}, "R": {}}},
            0.9,
        )
        explanation = build_explanation(solver)
        self.assertIn("2 A", explanation)
        self.assertIn("semantic frontend", explanation)
        self.assertIn("constraint graph", explanation)

        telemetry = build_pipeline_telemetry(
            front={"raw_question": "q", "parse_confidence": 1.0},
            route_result=SimpleNamespace(task_type="ohm_law", confidence=0.9),
            solver_result=solver,
            verification=SimpleNamespace(ok=True, confidence=0.9, issues=[]),
        )
        stored = persist_telemetry_event(telemetry)
        self.assertFalse(stored["enabled"])
        self.assertEqual(stored["reason"], "file_logging_removed")

    def test_remaining_uncertain_patterns_are_generalized_or_fail_closed(self):
        cases = [
            (
                "field_midpoint_inverse",
                "A charge q is placed at point O in the air. Ox is an electric field line. Take two points A and B on Ox. Let M be the midpoint of AB. E_A is the electric field strength at A, and E_B is the electric field strength at B. Determine 1/sqrt(E_M) in terms of E_A and E_B.",
                "ok",
                "point_charge_field_midpoint_inverse_expression",
                "1/sqrt(E_M)",
            ),
            (
                "lc_energy_complement",
                "An LC circuit has a capacitance C = 20 μF, an inductance L = 0.5 H, and a total energy of 0.2 J. When the voltage across the capacitor is 100 V, what is the magnetic field energy?",
                "ok",
                "lc_energy_complement",
                "0.1 J",
            ),
            (
                "solenoid_conceptual",
                "The magnetic field energy density in a solenoid is proportional to the square of which quantity?",
                "ok",
                "conceptual_direct",
                "magnetic induction B",
            ),
            (
                "rlc_resonance_yes_no",
                "A series RLC circuit has R=75 Ω, L=0.2 H, C=40 μF. Is 56.3 Hz the resonant frequency?",
                "ok",
                "yes_no_direct",
                "Yes",
            ),
            (
                "sinusoidal_rlc_current",
                "A voltage u = 200√2 cos 100πt (V) is applied to a series RLC circuit with R = 100 Ω, L = 1/π H, C = 10⁻⁴/(2π) F. Calculate the RMS (or effective) current I in the circuit.",
                "ok",
                "rlc_current_from_rlcf_voltage",
                "1.41421 A",
            ),
            (
                "underspecified_three_charge_geometry",
                "Two point charges, q1 = 6 × 10^-6 C and q2 = -6 × 10^-6 C, are placed in air at points A and B, separated by 10 cm. Calculate the electric force exerted on a charge q3 = -3 × 10^-8 C placed at C.",
                "underspecified_geometry",
                None,
                "Uncertain",
            ),
        ]
        for name, question, status, formula_id, answer_fragment in cases:
            with self.subTest(name=name):
                response = core_pipeline.process_question(question, enable_llm=False, planning_mode="deterministic", timeout_seconds=-1)
                self.assertEqual(response["metadata"]["status"], status)
                self.assertIn(answer_fragment, response["answer"])
                self.assertEqual(response["solver"]["formula_id"], formula_id)

    def test_core_implicit_allowlist_is_explicit(self):
        self.assertIn("school_coulomb_constant", allowed_implicit_rule_ids())


if __name__ == "__main__":
    unittest.main()
