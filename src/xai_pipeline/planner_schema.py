"""Production planner schema checks independent of model backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_PLANNER_FIELDS = {
    "status": str,
    "task_type": str,
    "answer_type": str,
    "given": list,
    "targets": list,
    "formula_ids": list,
    "principle_ids": list,
    "geometry_template_ids": list,
    "implicit_rule_ids": list,
    "decision_notes": list,
    "solve_steps": list,
    "solve_strategy": str,
    "conceptual_answer": (str, type(None)),
    "confidence": (int, float),
    "numeric_answer": type(None),
}


@dataclass(frozen=True)
class PlannerSchemaResult:
    ok: bool
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {"ok": self.ok, "issues": list(self.issues), "trace": dict(self.trace)}


def validate_planner_schema(plan: Any) -> PlannerSchemaResult:
    issues: list[str] = []
    if not isinstance(plan, dict):
        return PlannerSchemaResult(False, ["planner_not_object"], {"stage": "planner_schema"})
    for field, expected_type in REQUIRED_PLANNER_FIELDS.items():
        if field not in plan:
            issues.append(f"missing_field:{field}")
            continue
        if not isinstance(plan[field], expected_type):
            issues.append(f"wrong_type:{field}")
    notes = plan.get("decision_notes", [])
    if isinstance(notes, list):
        if len(notes) > 4:
            issues.append("too_many_decision_notes")
        for note in notes:
            if not isinstance(note, str):
                issues.append("non_string_decision_note")
                continue
            if _contains_arithmetic(note):
                issues.append("decision_note_contains_arithmetic")
    steps = plan.get("solve_steps", [])
    if isinstance(steps, list) and any(not isinstance(step, str) for step in steps):
        issues.append("non_string_solve_step")
    return PlannerSchemaResult(
        not issues,
        issues,
        {
            "stage": "planner_schema",
            "required_fields": sorted(REQUIRED_PLANNER_FIELDS),
            "field_count": len(plan),
        },
    )


def _contains_arithmetic(text: str) -> bool:
    compact = text.replace(" ", "")
    return any(op in compact for op in ["=", "+", "*", "/", "^"]) and any(ch.isdigit() for ch in compact)
