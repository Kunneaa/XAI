"""Structured telemetry for pipeline improvement."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TelemetryEvent:
    event_type: str
    payload: dict

    def to_dict(self) -> dict:
        return {"event_type": self.event_type, "payload": dict(self.payload)}


def build_pipeline_telemetry(
    *,
    front: dict,
    route_result,
    solver_result,
    verification,
    retrieval_hits: list[dict],
    planner,
    deadline,
) -> TelemetryEvent:
    """Build a JSON-serializable event without hidden chain-of-thought."""

    event_type = "verified_solution" if verification.ok else "unverified_fallback"
    return TelemetryEvent(
        event_type,
        {
            "raw_question": front.get("raw_question"),
            "quantity_dimensions": [q.get("dimension") for q in front.get("quantities", [])],
            "route_task_type": route_result.task_type,
            "route_confidence": route_result.confidence,
            "solver_formula_id": solver_result.formula_id,
            "solver_reason": solver_result.trace.get("reason"),
            "verifier_ok": verification.ok,
            "verifier_issues": list(verification.issues),
            "retrieval_count": len(retrieval_hits),
            "planner_reason": planner.reason,
            "deadline": deadline.to_dict(),
        },
    )


def persist_telemetry_event(event: TelemetryEvent | dict | None, path: str | Path | None = None) -> dict:
    """Append telemetry as JSONL when explicitly configured.

    Production can set ``XAI_TELEMETRY_PATH``. Tests/default local runs stay
    side-effect free unless a path is passed explicitly.
    """

    raw_target = path if path is not None else os.environ.get("XAI_TELEMETRY_PATH")
    if not raw_target:
        return {"stage": "telemetry_store", "enabled": False, "written": False}
    target = Path(raw_target).expanduser()
    payload = event.to_dict() if hasattr(event, "to_dict") else dict(event or {})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return {"stage": "telemetry_store", "enabled": True, "written": True, "path": str(target)}


def load_recent_telemetry_history(raw_question: str | None = None, path: str | Path | None = None, limit: int = 20) -> list[dict]:
    """Load safe telemetry snippets for adaptive planning.

    Only failure/planner flags are returned; no chain-of-thought or answers are
    exposed back into prompts.
    """

    raw_target = path if path is not None else os.environ.get("XAI_TELEMETRY_PATH")
    if not raw_target:
        return []
    target = Path(raw_target).expanduser()
    if not target.exists():
        return []
    events: list[dict] = []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()[-max(limit * 3, limit) :]
    except OSError:
        return []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if raw_question and payload.get("raw_question") != raw_question:
            continue
        verifier_issues = payload.get("verifier_issues") or []
        planner_reason = str(payload.get("planner_reason") or "")
        events.append(
            {
                "planner_failure": planner_reason
                in {"invalid_qwen_json", "invalid_schema", "qwen_repair_invalid_schema", "split_qwen_invalid_schema"},
                "schema_failure": any("schema" in str(issue) for issue in verifier_issues) or "schema" in planner_reason,
                "route_task_type": payload.get("route_task_type"),
                "solver_reason": payload.get("solver_reason"),
            }
        )
    return events[-limit:]
