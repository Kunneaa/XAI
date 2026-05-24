"""Guarded Qwen explanation polish boundary."""

from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass

from .qwen_config import QwenRuntimeConfig, resolve_qwen_runtime_config
from .qwen_runtime import generate_planner_text
from .units import unit_info


@dataclass(frozen=True)
class PolishResult:
    accepted: bool
    explanation: str
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {"accepted": self.accepted, "explanation": self.explanation, "issues": list(self.issues), "trace": dict(self.trace)}


def guarded_polish_boundary(
    explanation: str,
    immutable_answer: str,
    remaining_seconds: float,
    llm_budget,
    runtime_config: QwenRuntimeConfig | None = None,
) -> PolishResult:
    if remaining_seconds < 5.0:
        return PolishResult(False, explanation, ["insufficient_time_for_polish"], {"stage": "guarded_polish", "llm_used": False})
    if not llm_budget.can_polish():
        return PolishResult(False, explanation, ["llm_budget_disallows_polish"], {"stage": "guarded_polish", "llm_used": False, "budget": llm_budget.to_dict()})
    if os.environ.get("XAI_ENABLE_QWEN_POLISH", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return PolishResult(False, explanation, ["guarded_polish_disabled"], {"stage": "guarded_polish", "llm_used": False, "immutable_answer": immutable_answer})
    if not llm_budget.record_call("guarded_polish"):
        return PolishResult(False, explanation, ["llm_budget_disallows_polish"], {"stage": "guarded_polish", "llm_used": False, "budget": llm_budget.to_dict()})

    config = runtime_config or resolve_qwen_runtime_config()
    if not config.enabled:
        return PolishResult(False, explanation, ["local_qwen_disabled"], {"stage": "guarded_polish", "llm_used": False, "budget": llm_budget.to_dict(), "qwen_runtime": config.to_dict()})
    if not config.readiness.ready:
        return PolishResult(False, explanation, ["local_qwen_not_ready", *config.readiness.issues], {"stage": "guarded_polish", "llm_used": False, "budget": llm_budget.to_dict(), "qwen_runtime": config.to_dict()})

    prompt = _build_polish_prompt(explanation, immutable_answer)
    generation = generate_planner_text(prompt, config)
    if not generation.ok:
        return PolishResult(False, explanation, ["polish_generation_failed", *generation.issues], {"stage": "guarded_polish", "llm_used": True, "budget": llm_budget.to_dict(), "qwen_runtime": generation.to_dict()})
    polished = generation.text.strip().strip('"')
    issues = _validate_polish(polished, immutable_answer)
    if issues:
        return PolishResult(False, explanation, issues, {"stage": "guarded_polish", "llm_used": True, "accepted": False, "budget": llm_budget.to_dict(), "qwen_runtime": {k: v for k, v in generation.to_dict().items() if k != "text"}})
    return PolishResult(True, polished, [], {"stage": "guarded_polish", "llm_used": True, "accepted": True, "budget": llm_budget.to_dict(), "qwen_runtime": {k: v for k, v in generation.to_dict().items() if k != "text"}})


def _build_polish_prompt(explanation: str, immutable_answer: str) -> str:
    return (
        "You are polishing a verified physics explanation.\n"
        "Return only the polished explanation text, no markdown and no JSON.\n"
        "Rules:\n"
        f"- The final answer must remain exactly: {immutable_answer}\n"
        "- Do not add new formulas, assumptions, constants, or calculations.\n"
        "- Do not change any number or unit.\n"
        "- Keep it concise and teacher-like.\n"
        "Verified deterministic explanation:\n"
        f"{explanation}\n"
    )


def _validate_polish(polished: str, immutable_answer: str) -> list[str]:
    issues: list[str] = []
    if not polished:
        issues.append("polish_empty")
    if immutable_answer not in polished and not _contains_equivalent_answer(polished, immutable_answer):
        issues.append("polish_missing_immutable_answer")
    forbidden_markers = ["```", "{", "}", "numeric_answer", "I think", "approximately maybe"]
    if any(marker in polished for marker in forbidden_markers):
        issues.append("polish_contains_forbidden_marker")
    return issues


def _contains_equivalent_answer(polished: str, immutable_answer: str) -> bool:
    expected = _parse_answer_number_unit(immutable_answer)
    if expected is None:
        return False
    expected_value, expected_unit = expected
    expected_info = unit_info(expected_unit)
    for value, unit in _iter_number_units(polished):
        if unit == expected_unit and math.isclose(value, expected_value, rel_tol=1e-6, abs_tol=1e-12):
            return True
        candidate_info = unit_info(unit)
        if expected_info is not None and candidate_info is not None and expected_info.dimension == candidate_info.dimension:
            expected_si = expected_value * expected_info.si_factor
            candidate_si = value * candidate_info.si_factor
            if math.isclose(candidate_si, expected_si, rel_tol=1e-6, abs_tol=1e-12):
                return True
    return False


def _parse_answer_number_unit(text: str):
    matches = list(_iter_number_units(text))
    return matches[0] if matches else None


def _iter_number_units(text: str):
    pattern = re.compile(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s*([A-Za-zμΩ/%.-]+)")
    for match in pattern.finditer(text):
        try:
            yield float(match.group(1)), match.group(2).strip().strip(".,;:")
        except ValueError:
            continue
