"""Final answer consistency checks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List

from ..knowledge.units import unit_info


NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:×|x|X|\*)\s*10\^?[-+]?\d+|[eE][-+]?\d+)?"
)


@dataclass(frozen=True)
class AnswerCheckResult:
    ok: bool
    issues: List[str]
    trace: dict

    def to_dict(self):
        return {"ok": self.ok, "issues": list(self.issues), "trace": dict(self.trace)}


def check_answer(answer: str, explanation: str, solver_result, verification_result) -> AnswerCheckResult:
    issues: List[str] = []

    if not verification_result.ok:
        if answer != "Uncertain":
            issues.append("unverified_answer_not_uncertain")
        return AnswerCheckResult(not issues, issues, {"stage": "answer_checker", "mode": "controlled_fallback"})

    if not solver_result.solved:
        issues.append("verified_without_solver_result")
    if not answer or answer == "Uncertain":
        issues.append("missing_verified_answer")
    if isinstance(solver_result.value, str):
        mode = "verified_conceptual"
        if solver_result.formula_id not in {"conceptual_direct", "yes_no_direct"}:
            mode = "verified_symbolic"
        if str(answer).strip() != solver_result.answer:
            issues.append("symbolic_answer_mismatch" if mode == "verified_symbolic" else "conceptual_answer_mismatch")
        if solver_result.answer and solver_result.answer not in explanation:
            issues.append("explanation_missing_verified_answer")
        return AnswerCheckResult(
            not issues,
            issues,
            {"stage": "answer_checker", "mode": mode, "answer": answer},
        )
    if isinstance(solver_result.value, list):
        for item in solver_result.value:
            unit = item.get("unit")
            if unit and unit != "-" and unit not in answer:
                issues.append("answer_unit_missing")
                break
    elif solver_result.unit and solver_result.unit not in answer:
        issues.append("answer_unit_missing")
    if solver_result.answer and solver_result.answer not in explanation:
        issues.append("explanation_missing_verified_answer")

    if isinstance(solver_result.value, list):
        answer_values = _numbers(answer)
        expected_values = [_item_value_in_display_unit(item) for item in solver_result.value]
        if len(answer_values) < len(expected_values):
            issues.append("answer_numeric_value_missing")
        else:
            for actual, expected in zip(answer_values, expected_values):
                if not _close(actual, expected):
                    issues.append("answer_numeric_value_mismatch")
                    break
        trace = {
            "stage": "answer_checker",
            "mode": "verified_multi_output",
            "answer_values": answer_values,
            "solver_values": expected_values,
        }
        return AnswerCheckResult(not issues, issues, trace)

    answer_value = _first_number(answer)
    if solver_result.value is None or not math.isfinite(float(solver_result.value)):
        issues.append("non_finite_solver_value")
    elif answer_value is None:
        issues.append("answer_numeric_value_missing")
    elif not _close(answer_value, float(solver_result.value)) and not _close(answer_value, _solver_value_in_answer_unit(solver_result)):
        issues.append("answer_numeric_value_mismatch")

    trace = {
        "stage": "answer_checker",
        "mode": "verified_numeric",
        "answer_value": answer_value,
        "solver_value": solver_result.value,
    }
    return AnswerCheckResult(not issues, issues, trace)


def _solver_value_in_answer_unit(solver_result) -> float:
    try:
        value = float(solver_result.value)
    except (TypeError, ValueError):
        return float("nan")
    info = unit_info(getattr(solver_result, "unit", None) or "")
    if info is None or info.si_factor == 0:
        return value
    return value / info.si_factor


def _item_value_in_display_unit(item: dict) -> float:
    value = float(item["value"])
    info = unit_info(item.get("unit") or "")
    if info is None or info.si_factor == 0:
        return value
    return value / info.si_factor


def _first_number(text: str):
    match = NUMBER_RE.search(str(text or ""))
    return _parse_number(match.group(0)) if match else None


def _numbers(text: str):
    return [_parse_number(match.group(0)) for match in NUMBER_RE.finditer(str(text or ""))]


def _parse_number(token: str) -> float:
    cleaned = str(token).replace(" ", "").replace("×", "x").replace("X", "x").replace("*", "x")
    if "x10" in cleaned:
        base, exponent = cleaned.split("x10", 1)
        return float(base) * (10 ** int(exponent.lstrip("^")))
    return float(cleaned)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=5e-6, abs_tol=5e-12)
