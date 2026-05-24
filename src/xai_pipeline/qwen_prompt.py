"""Prompt builder for guarded Qwen planning."""

from __future__ import annotations

import json

from .planner_schema import REQUIRED_PLANNER_FIELDS
from .registries import FORMULA_IDS, GEOMETRY_TEMPLATE_IDS, PRINCIPLE_IDS, SOLVE_STRATEGIES


def build_qwen_planner_prompt(front_payload: dict, route_result, retrieval_hits: list[dict], structured_backend: dict) -> str:
    """Build a strict JSON-only planner prompt without ground-truth answers."""

    safe_retrieval = [_safe_retrieval_hit(hit) for hit in retrieval_hits[:3]]
    schema_summary = {field: _type_name(expected) for field, expected in REQUIRED_PLANNER_FIELDS.items()}
    payload = {
        "question": front_payload.get("canonical_question", front_payload.get("raw_question", "")),
        "answer_type_hint": front_payload.get("answer_type_hint"),
        "route": route_result.to_dict() if hasattr(route_result, "to_dict") else dict(route_result),
        "quantities": front_payload.get("quantities", []),
        "target_hints": front_payload.get("target_hints", []),
        "implicit_facts": front_payload.get("implicit_facts", []),
        "retrieval_metadata_only": safe_retrieval,
        "allowed_formula_ids": sorted(FORMULA_IDS),
        "allowed_principle_ids": sorted(PRINCIPLE_IDS),
        "allowed_geometry_template_ids": sorted(GEOMETRY_TEMPLATE_IDS),
        "allowed_solve_strategies": sorted(SOLVE_STRATEGIES),
        "schema": schema_summary,
        "structured_output_backend": structured_backend,
    }
    return (
        "You are the planning module for a deterministic physics solver.\n"
        "Return exactly one JSON object and no markdown.\n"
        "Core rules:\n"
        "- Do not compute, simplify, estimate, or reveal any final numeric answer.\n"
        "- numeric_answer must always be null.\n"
        "- Use only allowed formula_ids, principle_ids, geometry_template_ids, and implicit_rule_ids already present in implicit_facts.\n"
        "- decision_notes must be short qualitative reasons without arithmetic expressions.\n"
        "- solve_steps may name deterministic operations, but must not contain calculated numeric results.\n"
        "- If the problem is ambiguous or unsupported, set status to needs_fallback or unsupported.\n"
        "Input JSON follows:\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "Return JSON now."
    )


def _safe_retrieval_hit(hit: dict) -> dict:
    metadata = dict(hit.get("task_metadata", {}))
    return {
        "problem_id": hit.get("problem_id"),
        "score": hit.get("score"),
        "task_metadata": {
            "concepts": metadata.get("concepts", []),
            "answer_type_hint": metadata.get("answer_type_hint"),
            "task_type": metadata.get("task_type"),
            "target_hints": metadata.get("target_hints", []),
            "quantity_dimensions": metadata.get("quantity_dimensions", []),
            "formula_ids": metadata.get("formula_ids", []),
            "principle_ids": metadata.get("principle_ids", []),
            "safe_fields_only": True,
        },
    }


def _type_name(expected) -> str:
    if isinstance(expected, tuple):
        return "|".join(_type_name(item) for item in expected)
    return getattr(expected, "__name__", str(expected))
