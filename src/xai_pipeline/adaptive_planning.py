"""Adaptive planning policy boundary."""

from __future__ import annotations


def choose_planning_mode(telemetry_history: list[dict] | None = None) -> dict:
    """Choose combined or split planning from telemetry.

    The production default is combined planning. Split planning is only allowed
    when telemetry proves repeated schema/routing failures for the same route.
    """

    history = telemetry_history or []
    repeated_failures = sum(1 for event in history if event.get("planner_failure") or event.get("schema_failure"))
    if repeated_failures >= 3:
        return {"stage": "adaptive_planning", "mode": "split_extract_then_plan", "evidence_count": repeated_failures}
    return {"stage": "adaptive_planning", "mode": "combined_normalize_and_plan", "evidence_count": repeated_failures}
