"""Structured in-response telemetry for deterministic NSP-Core runs."""

from __future__ import annotations

import time
from typing import Any, Dict


def build_pipeline_telemetry(
    *,
    front: dict,
    route_result,
    solver_result,
    verification,
    compiled_plan=None,
    deadline=None,
) -> Dict[str, Any]:
    plan = compiled_plan.to_dict() if hasattr(compiled_plan, "to_dict") else {}
    return {
        "timestamp": time.time(),
        "question": front.get("raw_question"),
        "parse_confidence": front.get("parse_confidence"),
        "route_task_type": route_result.task_type,
        "route_confidence": route_result.confidence,
        "solve_plan_ok": plan.get("ok"),
        "solve_plan_source": ((plan.get("plan") or {}) if isinstance(plan, dict) else {}).get("source"),
        "solve_plan_operations": [
            step.get("operation")
            for step in (((plan.get("plan") or {}) if isinstance(plan, dict) else {}).get("steps") or [])
            if isinstance(step, dict)
        ],
        "solver_solved": solver_result.solved,
        "solver_formula_id": solver_result.formula_id,
        "verifier_ok": verification.ok,
        "verifier_confidence": verification.confidence,
        "verifier_issues": list(verification.issues),
        "deadline_remaining_seconds": deadline.remaining_seconds() if deadline is not None else None,
    }


def persist_telemetry_event(payload: Dict[str, Any]) -> dict:
    """No-op file persistence hook.

    The pipeline keeps telemetry in the API response for debugging, but it does
    not write JSONL logs or create a local ``logs/`` directory.
    """

    del payload
    return {"enabled": False, "reason": "file_logging_removed"}
