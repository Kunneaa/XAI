"""Deterministic semantic matcher and Qwen classifier boundary for implicit KB."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .implicit_kb import IMPLICIT_RULES, allowed_implicit_rule_ids
from .json_repair import parse_or_repair_json
from .qwen_config import QwenRuntimeConfig, resolve_qwen_runtime_config
from .qwen_runtime import generate_planner_text


SEMANTIC_ALIASES = {
    "school_coulomb_constant": ("air", "vacuum", "point charge", "electric charge", "coulomb"),
    "vacuum_permittivity": ("parallel plate", "free space", "vacuum permittivity"),
    "magnetic_constant": ("solenoid", "mu0", "permeability"),
    "electron": ("electron",),
    "proton": ("proton",),
    "ideal_lc_no_loss": ("ideal lc", "lossless", "no loss"),
    "series_rlc_resonance": ("resonance", "resonant", "series rlc"),
}


@dataclass(frozen=True)
class ImplicitClassifierResult:
    ok: bool
    matches: list[dict]
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {"ok": self.ok, "matches": list(self.matches), "issues": list(self.issues), "trace": dict(self.trace)}


def semantic_match_implicit_rules(question: str, threshold: float = 0.66) -> ImplicitClassifierResult:
    text = str(question or "").lower()
    matches: list[dict] = []
    for rule_id, aliases in SEMANTIC_ALIASES.items():
        if rule_id not in IMPLICIT_RULES:
            continue
        hit_aliases = [alias for alias in aliases if alias in text]
        if not hit_aliases:
            continue
        confidence = min(0.95, 0.55 + 0.15 * len(hit_aliases))
        if confidence >= threshold:
            matches.append({"rule_id": rule_id, "trigger_text": hit_aliases[0], "confidence": confidence})
    return ImplicitClassifierResult(
        True,
        matches,
        [],
        {
            "stage": "implicit_semantic_matcher",
            "threshold": threshold,
            "allowed_rule_ids": allowed_implicit_rule_ids(),
            "llm_used": False,
        },
    )


def qwen_implicit_classifier_boundary(
    question: str,
    candidate_rule_ids: list[str],
    runtime_config: QwenRuntimeConfig | None = None,
    threshold: float = 0.75,
) -> ImplicitClassifierResult:
    unknown = [rule_id for rule_id in candidate_rule_ids if rule_id not in IMPLICIT_RULES]
    if unknown:
        return ImplicitClassifierResult(False, [], [f"unknown_implicit_rule:{rule_id}" for rule_id in unknown], {"stage": "qwen_implicit_classifier", "llm_used": False})
    if os.environ.get("XAI_ENABLE_QWEN_IMPLICIT", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return ImplicitClassifierResult(
            False,
            [],
            ["qwen_implicit_classifier_disabled"],
            {"stage": "qwen_implicit_classifier", "candidate_rule_ids": list(candidate_rule_ids), "llm_used": False},
        )
    config = runtime_config or resolve_qwen_runtime_config()
    if not config.enabled:
        return ImplicitClassifierResult(False, [], ["local_qwen_disabled"], {"stage": "qwen_implicit_classifier", "candidate_rule_ids": list(candidate_rule_ids), "llm_used": False, "qwen_runtime": config.to_dict()})
    if not config.readiness.ready:
        return ImplicitClassifierResult(False, [], ["local_qwen_not_ready", *config.readiness.issues], {"stage": "qwen_implicit_classifier", "candidate_rule_ids": list(candidate_rule_ids), "llm_used": False, "qwen_runtime": config.to_dict()})
    prompt = _implicit_prompt(question, candidate_rule_ids)
    generation = generate_planner_text(prompt, config)
    if not generation.ok:
        return ImplicitClassifierResult(False, [], ["qwen_generation_failed", *generation.issues], {"stage": "qwen_implicit_classifier", "llm_used": True, "qwen_runtime": generation.to_dict()})
    parsed = parse_or_repair_json(generation.text)
    if not parsed.ok or not isinstance(parsed.value, dict):
        return ImplicitClassifierResult(False, [], ["invalid_implicit_classifier_json", *parsed.issues], {"stage": "qwen_implicit_classifier", "llm_used": True, "json_repair": parsed.to_dict()})
    text = str(question or "")
    matches: list[dict] = []
    for item in parsed.value.get("matches", []):
        if not isinstance(item, dict):
            continue
        rule_id = item.get("rule_id")
        trigger_span = str(item.get("trigger_span") or "")
        confidence = float(item.get("confidence", 0.0))
        if rule_id not in candidate_rule_ids or rule_id not in IMPLICIT_RULES:
            continue
        if confidence < threshold or not trigger_span or trigger_span not in text:
            continue
        matches.append({"rule_id": rule_id, "trigger_text": trigger_span, "confidence": confidence})
    if not matches:
        return ImplicitClassifierResult(False, [], ["no_valid_qwen_implicit_matches"], {"stage": "qwen_implicit_classifier", "llm_used": True, "json_repair": parsed.to_dict()})
    return ImplicitClassifierResult(
        True,
        matches,
        [],
        {"stage": "qwen_implicit_classifier", "candidate_rule_ids": list(candidate_rule_ids), "llm_used": True, "json_repair": parsed.to_dict()},
    )


def _implicit_prompt(question: str, candidate_rule_ids: list[str]) -> str:
    descriptions = {
        rule_id: {"premise": IMPLICIT_RULES[rule_id].premise, "adds": IMPLICIT_RULES[rule_id].adds}
        for rule_id in candidate_rule_ids
        if rule_id in IMPLICIT_RULES
    }
    return (
        "Select only triggered implicit physics rules from the finite allowlist.\n"
        "Return exactly JSON: {\"matches\":[{\"rule_id\":\"...\",\"trigger_span\":\"exact substring from question\",\"confidence\":0.0}]}.\n"
        "Do not invent rules or values. The trigger_span must be copied from the question.\n"
        f"Question: {question}\n"
        f"Allowed rules: {descriptions}\n"
    )
