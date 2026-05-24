"""Guarded Qwen planner adapter.

This module intentionally defaults to a no-LLM path. When enabled later, it
must return only schema-bound metadata and never a final numeric answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .json_repair import parse_or_repair_json
from .llm_budget import LlmBudgetState
from .constrained_decoding import generate_constrained_planner_text
from .qwen_config import QwenRuntimeConfig, resolve_qwen_runtime_config
from .qwen_prompt import build_qwen_planner_prompt
from .qwen_runtime import generate_planner_text
from .structured_output import select_structured_output_backend
from .verifier import validate_plan


@dataclass(frozen=True)
class PlannerResult:
    used_llm: bool
    plan: Optional[dict]
    validation: dict
    reason: str
    budget: Optional[dict] = None

    def to_dict(self):
        return {
            "used_llm": self.used_llm,
            "plan": self.plan,
            "validation": dict(self.validation),
            "reason": self.reason,
            "budget": self.budget,
        }


def plan_with_qwen_if_needed(
    front_payload: dict,
    route_result,
    solver_result,
    retrieval_hits,
    enable_llm: bool = False,
    runtime_config: QwenRuntimeConfig | None = None,
    budget: LlmBudgetState | None = None,
) -> PlannerResult:
    budget = budget or LlmBudgetState()
    if solver_result.solved:
        return PlannerResult(False, None, {"ok": True, "issues": []}, "solver_already_solved", budget.to_dict())
    if not enable_llm:
        return PlannerResult(False, None, {"ok": True, "issues": []}, "llm_disabled", budget.to_dict())
    config = runtime_config or resolve_qwen_runtime_config()
    if _should_use_split_planning(front_payload):
        split = _plan_with_split_calls(front_payload, route_result, retrieval_hits, config, budget)
        if split is not None:
            return split
    if not budget.record_call("combined_normalize_and_plan"):
        return PlannerResult(False, None, {"ok": False, "issues": ["llm_budget_exhausted"]}, "llm_budget_exhausted", budget.to_dict())
    backend = select_structured_output_backend(config.structured_backend)
    config_dict = config.to_dict()

    if not config.enabled:
        return PlannerResult(
            False,
            _fallback_plan(front_payload, route_result, "Local Qwen is configured but disabled by XAI_ENABLE_LOCAL_QWEN."),
            {
                "ok": True,
                "issues": [],
                "structured_output_backend": backend.to_dict(),
                "qwen_runtime": {"enabled": False, "config": config_dict},
            },
            "local_qwen_disabled",
            budget.to_dict(),
        )
    if not config.readiness.ready:
        return PlannerResult(
            False,
            _fallback_plan(front_payload, route_result, "Local Qwen model folder is not ready."),
            {
                "ok": False,
                "issues": ["local_qwen_not_ready", *config.readiness.issues],
                "structured_output_backend": backend.to_dict(),
                "qwen_runtime": {"enabled": True, "config": config_dict},
            },
            "local_qwen_not_ready",
            budget.to_dict(),
        )

    prompt = build_qwen_planner_prompt(front_payload, route_result, retrieval_hits, backend.to_dict())
    if getattr(config, "require_constrained_decoding", False) and not (backend.available and backend.name in {"vllm_guided_json", "sglang"}):
        return PlannerResult(
            False,
            _fallback_plan(front_payload, route_result, "True constrained decoding was required but no vLLM/SGLang structured endpoint is available."),
            {
                "ok": False,
                "issues": ["true_constrained_decoding_required_but_unavailable"],
                "structured_output_backend": backend.to_dict(),
                "qwen_runtime": {"enabled": True, "config": config_dict},
            },
            "constrained_decoding_unavailable",
            budget.to_dict(),
        )
    if backend.available and backend.name in {"vllm_guided_json", "sglang"}:
        generation = generate_constrained_planner_text(prompt, config, backend.name)
        generation_kind = "constrained"
    else:
        generation = generate_planner_text(prompt, config)
        generation_kind = "local_guarded"
    if not generation.ok:
        return PlannerResult(
            False,
            _fallback_plan(front_payload, route_result, "Local Qwen runtime failed before validated JSON was available."),
            {
                "ok": False,
                "issues": generation.issues,
                "structured_output_backend": backend.to_dict(),
                "qwen_runtime": generation.to_dict(),
                "generation_kind": generation_kind,
            },
            "qwen_runtime_failed",
            budget.to_dict(),
        )
    parsed = parse_or_repair_json(generation.text)
    if not parsed.ok:
        repaired = _attempt_planner_repair(generation.text, parsed.issues, prompt, config, backend, budget)
        if repaired is not None:
            return repaired
        return PlannerResult(
            True,
            None,
            {
                "ok": False,
                "issues": ["invalid_qwen_json", *parsed.issues],
                "json_repair": parsed.to_dict(),
                "structured_output_backend": backend.to_dict(),
                "qwen_runtime": {k: v for k, v in generation.to_dict().items() if k != "text"},
                "generation_kind": generation_kind,
            },
            "invalid_qwen_json",
            budget.to_dict(),
        )
    if not isinstance(parsed.value, dict):
        return PlannerResult(
            True,
            None,
            {
                "ok": False,
                "issues": ["planner_json_not_object"],
                "json_repair": parsed.to_dict(),
                "structured_output_backend": backend.to_dict(),
                "qwen_runtime": {k: v for k, v in generation.to_dict().items() if k != "text"},
                "generation_kind": generation_kind,
            },
            "invalid_schema",
            budget.to_dict(),
        )
    validation = validate_plan(parsed.value, front_payload)
    validation_dict = validation.to_dict()
    validation_dict["json_repair"] = parsed.to_dict()
    validation_dict["structured_output_backend"] = backend.to_dict()
    validation_dict["qwen_runtime"] = {k: v for k, v in generation.to_dict().items() if k != "text"}
    validation_dict["generation_kind"] = generation_kind
    if not validation.ok:
        repaired = _attempt_planner_repair(generation.text, validation.issues, prompt, config, backend, budget, front_payload)
        if repaired is not None:
            return repaired
    return PlannerResult(True, parsed.value, validation_dict, "validated_local_qwen_json", budget.to_dict())


def _should_use_split_planning(front_payload: dict) -> bool:
    telemetry_history = front_payload.get("telemetry_history")
    if not isinstance(telemetry_history, list):
        return False
    from .adaptive_planning import choose_planning_mode

    return choose_planning_mode(telemetry_history).get("mode") == "split_extract_then_plan"


def _plan_with_split_calls(front_payload: dict, route_result, retrieval_hits, config: QwenRuntimeConfig, budget: LlmBudgetState) -> PlannerResult | None:
    if not budget.record_call("extract_entities"):
        return PlannerResult(False, None, {"ok": False, "issues": ["llm_budget_exhausted_for_split"]}, "split_planning_budget_exhausted", budget.to_dict())
    if not budget.record_call("split_planning"):
        return PlannerResult(False, None, {"ok": False, "issues": ["llm_budget_exhausted_for_split"]}, "split_planning_budget_exhausted", budget.to_dict())
    backend = select_structured_output_backend(config.structured_backend)
    if not config.enabled:
        return PlannerResult(False, _fallback_plan(front_payload, route_result, "Split planning requested but local Qwen is disabled."), {"ok": True, "issues": [], "structured_output_backend": backend.to_dict(), "qwen_runtime": {"enabled": False, "config": config.to_dict()}}, "split_local_qwen_disabled", budget.to_dict())
    if not config.readiness.ready:
        return PlannerResult(False, _fallback_plan(front_payload, route_result, "Split planning requested but local Qwen model is not ready."), {"ok": False, "issues": ["local_qwen_not_ready", *config.readiness.issues], "structured_output_backend": backend.to_dict()}, "split_local_qwen_not_ready", budget.to_dict())
    entity_prompt = (
        "Extract only current-question entities for a deterministic physics planner.\n"
        "Return JSON with keys quantities, targets, answer_type. Do not solve.\n"
        f"Question: {front_payload.get('canonical_question')}\n"
        f"Existing deterministic quantities: {front_payload.get('quantities', [])}\n"
    )
    entity_generation = generate_planner_text(entity_prompt, config)
    if not entity_generation.ok:
        return PlannerResult(True, None, {"ok": False, "issues": ["split_entity_generation_failed", *entity_generation.issues]}, "split_entity_generation_failed", budget.to_dict())
    entity_json = parse_or_repair_json(entity_generation.text)
    if not entity_json.ok or not isinstance(entity_json.value, dict):
        return PlannerResult(True, None, {"ok": False, "issues": ["split_entity_json_invalid", *entity_json.issues], "json_repair": entity_json.to_dict()}, "split_entity_json_invalid", budget.to_dict())

    planning_front = dict(front_payload)
    planning_front["split_extracted_entities"] = entity_json.value
    prompt = build_qwen_planner_prompt(planning_front, route_result, retrieval_hits, backend.to_dict())
    if getattr(config, "require_constrained_decoding", False) and not (backend.available and backend.name in {"vllm_guided_json", "sglang"}):
        return PlannerResult(True, None, {"ok": False, "issues": ["true_constrained_decoding_required_but_unavailable"], "structured_output_backend": backend.to_dict()}, "split_constrained_decoding_unavailable", budget.to_dict())
    if backend.available and backend.name in {"vllm_guided_json", "sglang"}:
        generation = generate_constrained_planner_text(prompt, config, backend.name)
        generation_kind = "split_constrained"
    else:
        generation = generate_planner_text(prompt, config)
        generation_kind = "split_local_guarded"
    if not generation.ok:
        return PlannerResult(True, None, {"ok": False, "issues": ["split_plan_generation_failed", *generation.issues]}, "split_plan_generation_failed", budget.to_dict())
    parsed = parse_or_repair_json(generation.text)
    if not parsed.ok or not isinstance(parsed.value, dict):
        return PlannerResult(True, None, {"ok": False, "issues": ["split_plan_json_invalid", *parsed.issues], "json_repair": parsed.to_dict()}, "split_plan_json_invalid", budget.to_dict())
    validation = validate_plan(parsed.value, front_payload)
    validation_dict = validation.to_dict()
    validation_dict["split_entities"] = entity_json.to_dict()
    validation_dict["json_repair"] = parsed.to_dict()
    validation_dict["structured_output_backend"] = backend.to_dict()
    validation_dict["generation_kind"] = generation_kind
    return PlannerResult(True, parsed.value if validation.ok else None, validation_dict, "validated_split_qwen_json" if validation.ok else "split_qwen_invalid_schema", budget.to_dict())


def _attempt_planner_repair(
    bad_text: str,
    issues: list[str],
    original_prompt: str,
    config: QwenRuntimeConfig,
    backend,
    budget: LlmBudgetState,
    front_payload: dict | None = None,
) -> PlannerResult | None:
    if not budget.record_call("repair"):
        return None
    repair_prompt = (
        "Repair the previous planner output into exactly one valid JSON object.\n"
        "Do not compute any numeric answer; numeric_answer must be null.\n"
        f"Validation issues: {issues}\n"
        "Original task prompt:\n"
        f"{original_prompt}\n"
        "Bad output:\n"
        f"{bad_text}\n"
        "Return repaired JSON only."
    )
    if backend.available and backend.name in {"vllm_guided_json", "sglang"}:
        generation = generate_constrained_planner_text(repair_prompt, config, backend.name)
    else:
        generation = generate_planner_text(repair_prompt, config)
    if not generation.ok:
        return PlannerResult(
            True,
            None,
            {"ok": False, "issues": ["repair_generation_failed", *generation.issues], "structured_output_backend": backend.to_dict(), "qwen_runtime": generation.to_dict()},
            "qwen_repair_failed",
            budget.to_dict(),
        )
    parsed = parse_or_repair_json(generation.text)
    if not parsed.ok or not isinstance(parsed.value, dict):
        return PlannerResult(
            True,
            None,
            {"ok": False, "issues": ["repair_invalid_json", *parsed.issues], "json_repair": parsed.to_dict(), "structured_output_backend": backend.to_dict()},
            "qwen_repair_invalid_json",
            budget.to_dict(),
        )
    if front_payload is None:
        return PlannerResult(True, parsed.value, {"ok": True, "issues": [], "json_repair": parsed.to_dict(), "structured_output_backend": backend.to_dict()}, "repaired_local_qwen_json", budget.to_dict())
    validation = validate_plan(parsed.value, front_payload)
    validation_dict = validation.to_dict()
    validation_dict["json_repair"] = parsed.to_dict()
    validation_dict["structured_output_backend"] = backend.to_dict()
    validation_dict["qwen_runtime"] = {k: v for k, v in generation.to_dict().items() if k != "text"}
    return PlannerResult(True, parsed.value if validation.ok else None, validation_dict, "validated_repaired_local_qwen_json" if validation.ok else "qwen_repair_invalid_schema", budget.to_dict())


def _fallback_plan(front_payload: dict, route_result, note: str) -> dict:
    target_hints = front_payload["target_hints"] or ["unknown"]
    draft_plan = {
        "status": "needs_fallback",
        "task_type": route_result.task_type,
        "answer_type": front_payload["answer_type_hint"],
        "given": front_payload["quantities"],
        "targets": [{"symbol": hint, "name": hint} for hint in target_hints],
        "formula_ids": [],
        "principle_ids": [],
        "geometry_template_ids": [],
        "implicit_rule_ids": [fact["rule_id"] for fact in front_payload["implicit_facts"]],
        "decision_notes": [note],
        "solve_steps": [],
        "solve_strategy": "direct",
        "conceptual_answer": None,
        "confidence": 0.0,
        "numeric_answer": None,
    }
    return draft_plan


def validate_planner_json(text: str, front_payload: dict) -> PlannerResult:
    """Validate externally supplied planner JSON without trusting it."""

    budget = LlmBudgetState()
    budget.record_call("combined_normalize_and_plan")
    parsed = parse_or_repair_json(text)
    if not parsed.ok:
        return PlannerResult(True, None, parsed.to_dict(), "invalid_json", budget.to_dict())
    if not isinstance(parsed.value, dict):
        return PlannerResult(True, None, {"ok": False, "issues": ["planner_json_not_object"], "json_repair": parsed.to_dict()}, "invalid_schema", budget.to_dict())
    validation = validate_plan(parsed.value, front_payload)
    validation_dict = validation.to_dict()
    validation_dict["json_repair"] = parsed.to_dict()
    return PlannerResult(True, parsed.value, validation_dict, "validated_external_json", budget.to_dict())
