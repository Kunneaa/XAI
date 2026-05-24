"""Final response assembly."""

from __future__ import annotations


def format_response(
    *,
    front: dict,
    route_result,
    registry_validation,
    unit_conversion,
    solver_result,
    verification,
    retrieval_hits,
    planner,
    answer: str,
    explanation: str,
    confidence: float,
    answer_check,
    cache_hit: bool,
    telemetry=None,
    polish=None,
) -> dict:
    cot = _public_cot(solver_result, verification)
    premises = list(solver_result.premises) if solver_result.premises else []
    metadata = {
        "task_type": route_result.task_type,
        "answer_type": route_result.answer_type,
        "formula_id": solver_result.formula_id,
        "principle_id": solver_result.principle_id,
        "verified": verification.ok,
        "answer_checked": answer_check.ok,
        "target_unit_policy": "si_default_with_requested_target_unit_convert_back",
        "unsupported_execution_modes": [
            "free_form_geometry",
        ],
        "supported_boundaries": [
            "api_wrapper",
            "json_repair",
            "principle_selector",
            "geometry_template_matcher",
            "qwen_implicit_classifier_boundary",
            "guarded_qwen_polish",
            "sympy_worker_pool_boundary",
            "numerical_fallback_boundary",
            "telemetry_event",
        ],
    }
    return {
        "answer": answer,
        "explanation": explanation,
        "cot": cot,
        "premises": premises,
        "confidence": confidence,
        "metadata": metadata,
        "front": front,
        "route": route_result.to_dict(),
        "schema_registry_validator": registry_validation.to_dict(),
        "unit_conversion": unit_conversion.to_dict(),
        "solver": solver_result.to_dict(),
        "verifier": verification.to_dict(),
        "answer_checker": answer_check.to_dict(),
        "retrieval": retrieval_hits,
        "planner": planner.to_dict(),
        "cache": {"hit": cache_hit, "policy": "verified_and_answer_checked_only"},
        "telemetry": telemetry.to_dict() if telemetry is not None else None,
        "polish": polish.to_dict() if polish is not None else None,
        "trace": {
            "stages": [
                "question",
                "deadline",
                "cache",
                "normalize",
                "extract_quantities",
                "implicit_kb",
                "deterministic_router",
                "retrieval_if_needed",
                "qwen_planner_if_needed",
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
                "telemetry",
            ],
            "cache": {"checked": True, "hit": cache_hit},
        },
    }


def _public_cot(solver_result, verification) -> list[str]:
    if not verification.ok:
        return ["The deterministic solver did not verify a final answer."]
    steps = []
    if solver_result.formula_id:
        steps.append(f"Select whitelisted formula `{solver_result.formula_id}`.")
    steps.append("Convert extracted quantities to SI units deterministically.")
    steps.append("Execute the formula in deterministic code.")
    steps.append("Verify the value, unit, and final answer string.")
    return steps
