"""Pipeline composition through planner boundary."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .answer_check import check_answer
from .cache import get_verified_response, put_verified_response
from .deadlines import start_deadline
from .executor import execute_deterministic
from .front_pipeline import process_question_front
from .implicit_classifier import qwen_implicit_classifier_boundary, semantic_match_implicit_rules
from .implicit_kb import IMPLICIT_RULES
from .llm_budget import LlmBudgetState
from .qwen_planner import plan_with_qwen_if_needed
from .planner_executor import execute_validated_plan
from .retrieval import retrieve_metadata
from .response import format_response
from .router import route
from .telemetry import build_pipeline_telemetry, load_recent_telemetry_history, persist_telemetry_event
from .trace_explanation import build_trace_explanation
from .unit_converter import apply_requested_target_unit, convert_front_quantities_to_si
from .guarded_polish import guarded_polish_boundary
from .validator import validate_registry_state
from .verifier import verify_solver


def process_question(question: str, data_path: Path | None = None, enable_llm: bool = False, timeout_seconds: float = 55.0) -> Dict[str, Any]:
    deadline = start_deadline(timeout_seconds)
    llm_budget = LlmBudgetState()
    cached = get_verified_response(question)
    if cached is not None:
        cached["cache"] = {"hit": True, "policy": "verified_and_answer_checked_only"}
        cached["trace"]["cache"] = {"checked": True, "hit": True}
        cached["trace"]["deadline"] = deadline.to_dict()
        return cached

    if deadline.expired():
        return _timeout_response(deadline)

    front = process_question_front(question)
    front["telemetry_history"] = load_recent_telemetry_history(front.get("raw_question"))
    if deadline.expired():
        return _timeout_response(deadline)
    route_result = route(front)
    unit_conversion = convert_front_quantities_to_si(front)
    solver_result, registry_validation, verification = _execute_route_and_verify(front, route_result, unit_conversion)
    if deadline.expired():
        return _timeout_response(deadline)

    retrieval_hits = []
    if not verification.ok and data_path is not None:
        retrieval_hits = [hit.to_dict() for hit in retrieve_metadata(question, data_path)]
    if deadline.expired():
        return _timeout_response(deadline)

    implicit_classifier_trace = {"stage": "implicit_classifier", "used": False, "matches": [], "issues": []}
    if not verification.ok:
        front, route_result, solver_result, registry_validation, verification, implicit_classifier_trace = _maybe_apply_implicit_classifier(
            front=front,
            route_result=route_result,
            solver_result=solver_result,
            registry_validation=registry_validation,
            verification=verification,
            unit_conversion=unit_conversion,
            enable_llm=enable_llm,
            llm_budget=llm_budget,
            deadline=deadline,
        )

    planner = plan_with_qwen_if_needed(
        front,
        route_result,
        solver_result,
        retrieval_hits,
        enable_llm=enable_llm and not deadline.expired(),
        budget=llm_budget,
    )
    if deadline.expired():
        return _timeout_response(deadline)
    if not verification.ok and not deadline.expired():
        planned_execution = execute_validated_plan(front, planner, unit_conversion)
        if planned_execution is not None:
            planned_route, planned_solver = planned_execution
            planned_registry_validation, planned_verification = _validate_solver(front, planned_route, planned_solver, unit_conversion)
            if planned_verification.ok:
                route_result = planned_route
                solver_result = planned_solver
                registry_validation = planned_registry_validation
                verification = planned_verification

    if verification.ok:
        solver_result, target_unit_trace = apply_requested_target_unit(front, solver_result)
        answer = solver_result.answer
        explanation = build_trace_explanation(solver_result, unit_conversion)
        confidence = verification.confidence
        polish = guarded_polish_boundary(explanation, answer, deadline.remaining_seconds(), llm_budget)
        if polish.accepted:
            explanation = polish.explanation
    else:
        target_unit_trace = {"stage": "target_unit_converter", "applied": False, "reason": "unverified"}
        answer = "Uncertain"
        explanation = "The deterministic solver could not verify an answer for this question."
        confidence = 0.0
        polish = None

    answer_check = check_answer(answer, explanation, solver_result, verification)
    if verification.ok and not answer_check.ok:
        answer = "Uncertain"
        explanation = "The deterministic answer checker rejected the computed answer."
        confidence = 0.0

    telemetry = build_pipeline_telemetry(
        front=front,
        route_result=route_result,
        solver_result=solver_result,
        verification=verification,
        retrieval_hits=retrieval_hits,
        planner=planner,
        deadline=deadline,
    )
    telemetry_store = persist_telemetry_event(telemetry)

    response = format_response(
        front=front,
        route_result=route_result,
        registry_validation=registry_validation,
        unit_conversion=unit_conversion,
        solver_result=solver_result,
        verification=verification,
        retrieval_hits=retrieval_hits,
        planner=planner,
        answer=answer,
        explanation=explanation,
        confidence=confidence,
        answer_check=answer_check,
        cache_hit=False,
        telemetry=telemetry,
        polish=polish,
    )
    response["trace"]["target_unit_conversion"] = target_unit_trace
    response["trace"]["telemetry_store"] = telemetry_store
    response["trace"]["implicit_classifier"] = implicit_classifier_trace
    if verification.ok and answer_check.ok:
        put_verified_response(question, response)
    response["trace"]["deadline"] = deadline.to_dict()
    return response


def _execute_route_and_verify(front: dict, route_result, unit_conversion):
    solver_result = execute_deterministic(front, route_result, unit_conversion)
    registry_validation, verification = _validate_solver(front, route_result, solver_result, unit_conversion)
    return solver_result, registry_validation, verification


def _validate_solver(front: dict, route_result, solver_result, unit_conversion):
    pre_validation = validate_registry_state(front, route_result)
    registry_validation = validate_registry_state(front, route_result, solver_result)
    if not pre_validation.ok:
        registry_validation = _merge_validation(pre_validation, registry_validation)
    verification = verify_solver(front, route_result, solver_result, registry_validation, unit_conversion)
    return registry_validation, verification


def _maybe_apply_implicit_classifier(
    *,
    front: dict,
    route_result,
    solver_result,
    registry_validation,
    verification,
    unit_conversion,
    enable_llm: bool,
    llm_budget: LlmBudgetState,
    deadline,
):
    existing_rule_ids = {fact.get("rule_id") for fact in front.get("implicit_facts", [])}
    deterministic = semantic_match_implicit_rules(front.get("canonical_question", ""))
    matches = [match for match in deterministic.matches if match.get("rule_id") not in existing_rule_ids] if deterministic.ok else []
    trace = {
        "stage": "implicit_classifier",
        "used": bool(matches),
        "semantic": deterministic.to_dict(),
        "qwen": None,
        "matches": list(matches),
        "issues": [],
    }
    if not matches and enable_llm and not deadline.expired():
        if os.environ.get("XAI_ENABLE_QWEN_IMPLICIT", "0").strip().lower() in {"1", "true", "yes", "on"} and llm_budget.record_call("implicit_classifier"):
            candidates = [rule_id for rule_id in IMPLICIT_RULES if rule_id not in existing_rule_ids]
            qwen = qwen_implicit_classifier_boundary(front.get("canonical_question", ""), candidates)
            trace["qwen"] = qwen.to_dict()
            if qwen.ok:
                matches = [match for match in qwen.matches if match.get("rule_id") not in existing_rule_ids]
                trace["matches"] = list(matches)
                trace["used"] = bool(matches)
            else:
                trace["issues"].extend(qwen.issues)
    if not matches:
        return front, route_result, solver_result, registry_validation, verification, trace

    augmented = _augment_front_with_implicit_matches(front, matches)
    new_route = route(augmented)
    new_solver, new_registry, new_verification = _execute_route_and_verify(augmented, new_route, unit_conversion)
    trace["rerun"] = {
        "route_task_type": new_route.task_type,
        "solver_formula_id": new_solver.formula_id,
        "verified": new_verification.ok,
    }
    if new_verification.ok:
        return augmented, new_route, new_solver, new_registry, new_verification, trace
    return front, route_result, solver_result, registry_validation, verification, trace


def _augment_front_with_implicit_matches(front: dict, matches: list[dict]) -> dict:
    augmented = dict(front)
    facts = [dict(fact) for fact in front.get("implicit_facts", [])]
    premises = list(front.get("premises", []))
    applied = set(fact.get("rule_id") for fact in facts)
    for match in matches:
        rule_id = match.get("rule_id")
        if rule_id in applied or rule_id not in IMPLICIT_RULES:
            continue
        rule = IMPLICIT_RULES[rule_id]
        trigger_text = str(match.get("trigger_text") or "")
        question = str(front.get("canonical_question") or "")
        start = question.lower().find(trigger_text.lower()) if trigger_text else -1
        span = (start, start + len(trigger_text)) if start >= 0 else None
        facts.append(
            {
                "rule_id": rule.rule_id,
                "adds": dict(rule.adds),
                "premise": rule.premise,
                "trigger_text": trigger_text,
                "span": span,
                "confidence": min(float(match.get("confidence", rule.confidence)), rule.confidence),
            }
        )
        premises.append(rule.premise)
        applied.add(rule_id)
    augmented["implicit_facts"] = facts
    augmented["premises"] = premises
    trace = dict(front.get("trace", {}))
    implicit_trace = dict(trace.get("implicit_kb", {}))
    implicit_trace["semantic_or_qwen_augmented_rules"] = sorted(applied)
    trace["implicit_kb"] = implicit_trace
    augmented["trace"] = trace
    return augmented


def _merge_validation(first, second):
    if first.ok:
        return second
    if second.ok:
        return first
    issues = list(dict.fromkeys([*first.issues, *second.issues]))
    trace = dict(second.trace)
    trace["pre_validation_issues"] = list(first.issues)
    return type(second)(False, issues, trace)


def _timeout_response(deadline) -> Dict[str, Any]:
    return {
        "answer": "Uncertain",
        "explanation": "The request exceeded the deterministic deadline before execution started.",
        "cot": ["The pipeline returned a controlled timeout fallback."],
        "premises": [],
        "confidence": 0.0,
        "metadata": {"verified": False, "answer_checked": False, "timeout": True},
        "front": {},
        "route": {"task_type": "unknown", "answer_type": "unknown", "confidence": 0.0, "reasons": []},
        "schema_registry_validator": {"ok": False, "issues": ["deadline_expired"], "trace": {}},
        "unit_conversion": {"ok": False, "quantities_si": [], "issues": ["deadline_expired"], "trace": {}},
        "solver": {"solved": False, "answer": "", "value": None, "unit": None, "formula_id": None, "principle_id": None, "premises": [], "trace": {"reason": "deadline_expired"}, "confidence": 0.0},
        "verifier": {"ok": False, "confidence": 0.0, "issues": ["deadline_expired"]},
        "answer_checker": {"ok": True, "issues": [], "trace": {"stage": "answer_checker", "mode": "controlled_fallback"}},
        "retrieval": [],
        "planner": {"used_llm": False, "plan": None, "validation": {"ok": True, "issues": []}, "reason": "deadline_expired"},
        "cache": {"hit": False, "policy": "verified_and_answer_checked_only"},
        "trace": {"stages": ["question", "deadline", "controlled_timeout_response"], "deadline": deadline.to_dict(), "cache": {"checked": False, "hit": False}},
    }
