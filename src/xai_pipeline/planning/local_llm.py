"""Controlled local-LLM structured-plan boundary.

The Core allows a fine-tuned open-source model only behind semantic audit,
structured solve-plan proposal, one compiler-driven re-plan attempt, and one
verifier-driven repair. This module intentionally fails closed unless the
runtime dependencies and model files are present. It never returns final
numeric answers, new formulas, constants, units, coordinates, or code.
"""

from __future__ import annotations

import importlib.util
import copy
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

from .answer_formats import build_output_format, prompt_answer_format_contract
from .solve_plan import build_deterministic_solve_plan, plan_summary
from ..engines.spatial_engine import match_geometry_templates
from ..frontend.canonical import build_canonical_structures
from ..knowledge.formula_catalog import formula_prompt_pack
from ..knowledge.registries import ANSWER_TYPES, FORMULA_REGISTRY, GEOMETRY_TEMPLATE_IDS, PLAN_OPERATION_IDS, formula_execution_branch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "models/deepseek-r1-distill-qwen-7b-exact-lora"
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json")
_RUNTIME: dict[str, Any] = {}
_READINESS_CACHE: dict[str, dict] = {}
SOLVE_PLAN_ALLOWED_OPERATIONS = sorted(PLAN_OPERATION_IDS)
STATUS_VALUES = ["ok", "needs_fallback", "unsupported"]
SOLVE_PLAN_TOP_LEVEL_KEYS = {
    "status",
    "plan_template_id",
    "plan_card_id",
    "formula_id",
    "principle_id",
    "geometry_template_id",
    "public_cot",
    "task_type",
    "answer_type",
    "targets",
    "assumptions",
    "steps",
    "output_format",
    "source",
    "notes",
}
SEMANTIC_TOP_LEVEL_KEYS = {
    "status",
    "missing_fields",
    "ambiguous_references",
    "candidate_target_texts",
    "answer_type_hint",
    "target_overrides",
    "relation_hints",
    "symbol_dimension_overrides",
    "proposed_ir_patch",
    "trigger_spans",
    "notes",
}
FORBIDDEN_LLM_KEYS = {
    "answer",
    "final_answer",
    "numeric_answer",
    "coordinates",
    "coordinate",
    "python",
    "code",
}
DEFAULT_SOLVE_PLAN_MAX_NEW_TOKENS = 224
DEFAULT_FRONT_REPAIR_MAX_NEW_TOKENS = 128
JSON_STOP_CHECK_INTERVAL = 4
PUBLIC_PLAN_TEXT_FORBIDDEN_CUES = (
    "\n",
    "```",
    "__import__",
    "lambda ",
    "final answer",
    "answer is",
    "therefore",
    "hence",
    "coordinate",
    "python",
    "code",
)
PUBLIC_PLAN_NUMERIC_UNIT_RE = re.compile(
    r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*(?:N/C|V/m|ohm|Ω|N|A|V|J|F|C|W|Hz|H|T|Wb|m|s|%)\b",
    re.IGNORECASE,
)
FRONT_REPAIR_DIMENSIONS = frozenset(
    {
        "area",
        "capacitance",
        "charge",
        "count",
        "current",
        "dimensionless",
        "electric_field",
        "energy",
        "force",
        "frequency",
        "inductance",
        "length",
        "magnetic_field",
        "magnetic_flux",
        "percent",
        "phase_angle",
        "power",
        "resistance",
        "resistivity",
        "time",
        "voltage",
    }
)
FRONT_REPAIR_RELATION_TYPES = frozenset({"geometry", "topology", "state", "concept"})
FRONT_REPAIR_RELATION_QUALIFIERS = frozenset(
    {
        "balanced_bridge",
        "center",
        "collinear",
        "equilateral_triangle",
        "external_point_on_line",
        "midpoint",
        "parallel",
        "perpendicular_bisector",
        "rectangle",
        "resonance",
        "right_isosceles_triangle",
        "series",
        "square",
        "triangle",
    }
)


def check_local_llm_readiness(
    adapter_dir: str | Path | None = None,
    base_model_dir: str | Path | None = None,
) -> dict:
    path = _resolve_path(adapter_dir or os.environ.get("XAI_LLM_ADAPTER_DIR") or DEFAULT_ADAPTER_DIR)
    adapter_config = _read_adapter_config(path)
    base_path = _resolve_path(
        base_model_dir
        or os.environ.get("XAI_LLM_BASE_MODEL_DIR")
        or adapter_config.get("base_model_name_or_path")
        or PROJECT_ROOT / "models/DeepSeek-R1-Distill-Qwen-7B"
    )
    runtime_config = _runtime_config_snapshot()
    cache_key = _readiness_cache_key(path, base_path, runtime_config)
    if cache_key in _READINESS_CACHE:
        return _READINESS_CACHE[cache_key]
    files = {name: (path / name).exists() for name in REQUIRED_ADAPTER_FILES}
    deps = {
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "peft": importlib.util.find_spec("peft") is not None,
        "safetensors": importlib.util.find_spec("safetensors") is not None,
        "accelerate": importlib.util.find_spec("accelerate") is not None,
    }
    base_ready = base_path.exists()
    ready = path.exists() and base_ready and all(files.values()) and all(deps.values())
    issues = []
    if not path.exists():
        issues.append(f"adapter_dir_missing:{path}")
    if not base_ready:
        issues.append(f"base_model_dir_missing:{base_path}")
    issues.extend(f"adapter_file_missing:{name}" for name, ok in files.items() if not ok)
    issues.extend(f"dependency_missing:{name}" for name, ok in deps.items() if not ok)
    issues.extend(_device_runtime_issues(runtime_config, deps))
    ready = ready and not any(issue.startswith("device_unavailable:") for issue in issues)
    readiness = {
        "ready": ready,
        "adapter_dir": str(path),
        "base_model_dir": str(base_path),
        "adapter_config": {
            "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
            "peft_type": adapter_config.get("peft_type"),
            "task_type": adapter_config.get("task_type"),
            "r": adapter_config.get("r"),
            "lora_alpha": adapter_config.get("lora_alpha"),
        },
        "files": files,
        "dependencies": deps,
        "runtime_config": runtime_config,
        "issues": issues,
    }
    _READINESS_CACHE[cache_key] = readiness
    return readiness


def _device_runtime_issues(runtime_config: dict[str, Any], deps: dict[str, bool]) -> list[str]:
    device = str(runtime_config.get("device") or "").strip().lower()
    device_map = str(runtime_config.get("device_map") or "").strip().lower()
    if not device or device == "auto" or device_map not in {"", "none", "false", "0"}:
        return []
    if not deps.get("torch"):
        return []
    try:
        import torch
    except Exception:
        return []
    if device == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None:
            return ["device_unavailable:mps:no_torch_backend"]
        built = bool(mps.is_built()) if hasattr(mps, "is_built") else None
        available = bool(mps.is_available())
        if not available:
            mac_version = platform.mac_ver()[0] or "unknown"
            return [f"device_unavailable:mps:built={built}:available={available}:macos={mac_version}"]
    if device == "cuda" and not bool(torch.cuda.is_available()):
        return ["device_unavailable:cuda"]
    return []


def refine_front_ir_if_enabled(front_payload: dict, enable_llm: bool = False) -> tuple[dict, dict]:
    """Return deterministic IR unless a validated local refinement is available."""

    if not enable_llm and os.environ.get("XAI_ENABLE_LOCAL_LLM") != "1":
        return front_payload, {"stage": "local_llm_refinement", "used": False, "reason": "disabled"}
    if not _front_repair_enabled() and not _truthy_env("XAI_LLM_ENABLE_SEMANTIC_AUDIT"):
        return front_payload, {
            "stage": "local_llm_refinement",
            "used": False,
            "reason": "front_repair_disabled",
            "policy": "direct_llm_plan_only_after_deterministic_frontend",
        }
    if not _truthy_env("XAI_LLM_ENABLE_SEMANTIC_AUDIT") and not _front_refinement_needed(front_payload):
        return front_payload, {
            "stage": "local_llm_refinement",
            "used": False,
            "reason": "front_repair_not_needed",
            "policy": "front_repair_is_selective_to_save_llm_tokens",
        }
    readiness = check_local_llm_readiness()
    if not readiness["ready"]:
        return front_payload, {"stage": "local_llm_refinement", "used": False, "reason": "runtime_unavailable", "readiness": readiness}
    prompt = _semantic_refinement_prompt(front_payload)
    generation = generate_local_llm_json(prompt, max_new_tokens=_max_new_tokens_for("semantic", DEFAULT_FRONT_REPAIR_MAX_NEW_TOKENS))
    repaired_front, patch_trace = _apply_front_repair_patch(front_payload, generation.get("json") if generation.get("ok") else None)
    trace = {
        "stage": "local_llm_refinement",
        "used": generation["used"],
        "applied": patch_trace["applied"],
        "reason": patch_trace["reason"] if patch_trace["applied"] else "generation_failed_or_no_valid_patch",
        "readiness": readiness,
        "generation": generation,
        "patch_validation": patch_trace,
        "policy": "schema_bound_front_patch_only_no_solver_authority",
    }
    if not generation["ok"]:
        trace["reason"] = "generation_failed_or_invalid_json"
    return repaired_front, trace


def propose_solve_plan_if_enabled(
    front_payload: dict,
    route_result=None,
    graph_selection=None,
    enable_llm: bool = False,
) -> tuple[dict | None, dict]:
    """Ask the local adapter for a structured solve plan, never for an answer."""

    if not enable_llm and os.environ.get("XAI_ENABLE_LOCAL_LLM") != "1":
        return None, {"stage": "local_llm_solve_plan", "used": False, "reason": "disabled"}
    readiness = check_local_llm_readiness()
    if not readiness["ready"]:
        return None, {"stage": "local_llm_solve_plan", "used": False, "reason": "runtime_unavailable", "readiness": readiness}
    prompt = _solve_plan_prompt(front_payload, route_result, graph_selection)
    generation = generate_local_llm_json(
        prompt,
        max_new_tokens=_max_new_tokens_for("solve_plan", DEFAULT_SOLVE_PLAN_MAX_NEW_TOKENS),
        schema="solve_plan",
    )
    raw_plan = generation.get("json") if generation.get("ok") and isinstance(generation.get("json"), dict) else None
    plan = _complete_llm_solve_plan_payload(raw_plan, front_payload, route_result, graph_selection)
    trace = {
        "stage": "local_llm_solve_plan",
        "used": generation["used"],
        "applied": False,
        "reason": "proposal_generated_unvalidated" if plan else "generation_failed_or_invalid_json",
        "readiness": readiness,
        "generation": generation,
        "completion": _completion_summary(raw_plan, plan),
        "plan_summary": plan_summary(plan),
        "policy": "schema_bound_solve_plan_only_no_numeric_answer_authority",
    }
    return plan, trace


def repair_solve_plan_once(
    front_payload: dict,
    error_packet: dict,
    route_result=None,
    graph_selection=None,
    enable_llm: bool = False,
) -> tuple[dict | None, dict]:
    """Verifier/compiler-driven one-shot re-plan hook."""

    if not enable_llm and os.environ.get("XAI_ENABLE_LOCAL_LLM") != "1":
        return None, {"stage": "local_llm_solve_plan_repair", "used": False, "reason": "disabled", "error_packet": error_packet}
    readiness = check_local_llm_readiness()
    if not readiness["ready"]:
        return None, {
            "stage": "local_llm_solve_plan_repair",
            "used": False,
            "reason": "runtime_unavailable",
            "readiness": readiness,
            "error_packet": error_packet,
        }
    prompt = _solve_plan_repair_prompt(front_payload, error_packet, route_result, graph_selection)
    generation = generate_local_llm_json(
        prompt,
        max_new_tokens=_max_new_tokens_for("solve_plan_repair", DEFAULT_SOLVE_PLAN_MAX_NEW_TOKENS),
        schema="solve_plan",
    )
    raw_repaired_plan = generation.get("json") if generation.get("ok") and isinstance(generation.get("json"), dict) else None
    repaired_plan = _complete_llm_solve_plan_payload(raw_repaired_plan, front_payload, route_result, graph_selection)
    return repaired_plan, {
        "stage": "local_llm_solve_plan_repair",
        "used": generation["used"],
        "applied": False,
        "reason": "repair_plan_generated_unvalidated" if repaired_plan else "generation_failed_or_invalid_json",
        "readiness": readiness,
        "error_packet": error_packet,
        "generation": generation,
        "completion": _completion_summary(raw_repaired_plan, repaired_plan),
        "plan_summary": plan_summary(repaired_plan),
        "policy": "one_replan_max_schema_bound_no_numeric_answer_authority",
    }


def repair_front_ir_once(front_payload: dict, verification_result, enable_llm: bool = False) -> tuple[dict, dict]:
    """Verifier-driven one-shot repair hook with strict fail-closed behavior."""

    error_packet = {
        "stage": "verifier",
        "issues": list(getattr(verification_result, "issues", [])),
        "conflicts": list(getattr(verification_result, "conflicts", [])),
        "allowed_edits": ["entities", "quantities", "relations", "conditions", "targets", "states", "events", "steps", "bindings"],
        "forbidden_outputs": ["numeric_answers", "new_formulas", "new_constants", "new_units", "coordinates", "free_form_cot"],
    }
    if not enable_llm and os.environ.get("XAI_ENABLE_LOCAL_LLM") != "1":
        return front_payload, {"stage": "local_llm_repair", "used": False, "reason": "disabled", "error_packet": error_packet}
    readiness = check_local_llm_readiness()
    if not readiness["ready"]:
        return front_payload, {
            "stage": "local_llm_repair",
            "used": False,
            "reason": "runtime_unavailable",
            "readiness": readiness,
            "error_packet": error_packet,
        }
    prompt = _repair_prompt(front_payload, error_packet)
    generation = generate_local_llm_json(prompt, max_new_tokens=_max_new_tokens_for("semantic_repair", DEFAULT_FRONT_REPAIR_MAX_NEW_TOKENS))
    repaired_front, patch_trace = _apply_front_repair_patch(front_payload, generation.get("json") if generation.get("ok") else None)
    return repaired_front, {
        "stage": "local_llm_repair",
        "used": generation["used"],
        "applied": patch_trace["applied"],
        "reason": patch_trace["reason"] if generation["ok"] else "generation_failed_or_invalid_json",
        "readiness": readiness,
        "error_packet": error_packet,
        "generation": generation,
        "patch_validation": patch_trace,
        "policy": "one_repair_max_schema_bound_ir_only",
    }


def _complete_llm_solve_plan_payload(
    raw_plan: dict | None,
    front_payload: dict,
    route_result=None,
    graph_selection=None,
) -> dict | None:
    """Fill missing static envelope fields while preserving the LLM step DAG."""

    if not isinstance(raw_plan, dict):
        return None
    raw_plan = _resolve_plan_card_choice(raw_plan, front_payload, route_result, graph_selection)
    deterministic_envelope = build_deterministic_solve_plan(front_payload, route_result, graph_selection).to_dict()
    status = raw_plan.get("status") if raw_plan.get("status") in STATUS_VALUES else deterministic_envelope.get("status")
    targets = list(raw_plan.get("targets") or deterministic_envelope.get("targets") or [])
    task_type = raw_plan.get("task_type") or deterministic_envelope.get("task_type")
    answer_type = raw_plan.get("answer_type") or deterministic_envelope.get("answer_type")
    output_format = raw_plan.get("output_format") if isinstance(raw_plan.get("output_format"), dict) else deterministic_envelope.get("output_format")
    steps = []
    if raw_plan.get("plan_template_id") or raw_plan.get("plan_card_id"):
        steps = _steps_from_compact_template_plan(raw_plan, deterministic_envelope)
    if not steps:
        steps = _normalize_llm_plan_steps(list(raw_plan.get("steps") or []))
    if status == "ok" and not steps:
        return None
    completed = {
        "status": status,
        "task_type": task_type,
        "answer_type": answer_type,
        "targets": targets,
        "assumptions": list(raw_plan.get("assumptions") or deterministic_envelope.get("assumptions") or []),
        "steps": steps,
        "output_format": dict(output_format or {}),
        "source": "local_llm",
        "notes": _safe_string_list(raw_plan.get("notes"), limit=4),
    }
    if status != "ok":
        completed["steps"] = []
        completed["output_format"] = {
            "format_kind": "controlled_fallback",
            "ordered_targets": [target.get("id") for target in targets if isinstance(target, dict) and target.get("id")],
            "target_count": len(targets),
        }
    elif not completed["output_format"]:
        completed["output_format"] = build_output_format(front_payload, str(completed["answer_type"]), str(completed["task_type"]), targets)
    return completed


def _resolve_plan_card_choice(raw_plan: dict, front_payload: dict, route_result=None, graph_selection=None) -> dict:
    """Resolve a tiny model choice like ``{"plan_card_id":"p1"}``."""

    card_id = str(raw_plan.get("plan_card_id") or "").strip()
    if not card_id:
        return raw_plan
    for card in _plan_cards_for_context(front_payload, route_result, graph_selection):
        if card.get("card_id") != card_id:
            continue
        resolved = dict(raw_plan)
        for key in ("plan_template_id", "formula_id", "principle_id", "geometry_template_id", "public_cot"):
            if card.get(key) is not None:
                resolved[key] = card[key]
        return resolved
    return raw_plan


def _plan_cards_for_context(front_payload: dict, route_result=None, graph_selection=None) -> list[dict]:
    candidate_formula_ids = _planning_candidate_formula_ids(front_payload, route_result, graph_selection)
    route_task_type = getattr(route_result, "task_type", None)
    prompt_pack = formula_prompt_pack(
        route_task_type=route_task_type,
        candidate_formula_ids=candidate_formula_ids,
        available_dimensions=_front_available_dimensions(front_payload),
        target_dimensions=_front_target_dimensions(front_payload),
        route_reasons=list(getattr(route_result, "reasons", []) or [])[:4],
    )
    return _compact_plan_cards(front_payload, route_result, prompt_pack, _compact_targets(front_payload))


def _planning_candidate_formula_ids(front_payload: dict, route_result=None, graph_selection=None) -> list[str]:
    """Merge graph-selected formulas with deterministic IR-derived plan formulas.

    The dimension graph can deliberately omit zero-dimensional symmetry cards.
    If deterministic IR has enough structure to produce such a card, expose it
    to the LLM as a selectable route-local option instead of falling back to
    unrelated scalar formulas from the same task family.
    """

    ordered: list[str] = []

    def add(formula_id: Any) -> None:
        formula_text = str(formula_id or "")
        if formula_text in FORMULA_REGISTRY and formula_text not in ordered:
            ordered.append(formula_text)

    for formula_id in list(getattr(graph_selection, "formula_ids", []) or []):
        add(formula_id)
    try:
        deterministic = build_deterministic_solve_plan(front_payload, route_result, graph_selection).to_dict()
    except Exception:
        deterministic = {}
    for step in deterministic.get("steps") or []:
        if isinstance(step, dict):
            add(step.get("formula_id"))
    return ordered[:12]


def _steps_from_compact_template_plan(raw_plan: dict, deterministic_envelope: dict) -> list[dict]:
    """Expand a compact LLM plan-card choice into executable registry steps.

    The model still owns the strategy choice through ``plan_template_id`` plus
    selected registry IDs; code only expands that choice into the local step
    schema that the compiler already validates.
    """

    template_id = str(raw_plan.get("plan_template_id") or raw_plan.get("plan_card_id") or "").strip()
    if not template_id:
        return []
    formula_id = _safe_formula_id(raw_plan.get("formula_id"))
    geometry_id = _safe_geometry_template_id(raw_plan.get("geometry_template_id"))
    principle_id = _safe_principle_id(raw_plan.get("principle_id"), formula_id)
    targets = deterministic_envelope.get("targets") or []
    target_id = "goal:1"
    if targets and isinstance(targets[0], dict) and targets[0].get("id"):
        target_id = str(targets[0]["id"])
    cot = _safe_string_list(raw_plan.get("public_cot"), limit=4)

    if template_id == "direct_formula":
        if not formula_id:
            return []
        return [
            {
                "step_id": "s1",
                "operation": "apply_formula",
                "formula_id": formula_id,
                "principle_id": principle_id,
                "inputs": {},
                "output": target_id,
                "depends_on": [],
                "public_cot": _cot_or_default(cot, 0, "Apply the selected registry relation to accepted facts."),
            }
        ]
    if template_id in {"spatial_pairwise_force", "spatial_vector_resolution"}:
        if not formula_id or not geometry_id:
            return []
        vector_operation = "compute_pairwise_force" if template_id == "spatial_pairwise_force" else "resolve_vector_components"
        return [
            {
                "step_id": "s1",
                "operation": "construct_geometry",
                "geometry_constructor_id": geometry_id,
                "inputs": {"facts": "formal_ir.geometry"},
                "output": "geom",
                "depends_on": [],
                "public_cot": _cot_or_default(cot, 0, "Construct accepted geometry."),
            },
            {
                "step_id": "s2",
                "operation": vector_operation,
                "formula_id": formula_id,
                "principle_id": principle_id,
                "inputs": {"geometry": "geom", "facts": "formal_ir.charges"},
                "output": target_id,
                "depends_on": ["s1"],
                "public_cot": _cot_or_default(cot, 1, "Resolve vector contributions."),
            },
        ]
    if template_id == "vector_resolution":
        if not formula_id:
            return []
        return [
            {
                "step_id": "s1",
                "operation": "resolve_vector_components",
                "formula_id": formula_id,
                "principle_id": principle_id,
                "inputs": {},
                "output": target_id,
                "depends_on": [],
                "public_cot": _cot_or_default(cot, 0, "Resolve accepted vector components."),
            }
        ]
    if template_id == "equation_subset":
        return [
            {
                "step_id": "s1",
                "operation": "solve_equation_subset",
                **({"formula_id": formula_id, "principle_id": principle_id} if formula_id else {}),
                "inputs": {},
                "output": target_id,
                "depends_on": [],
                "public_cot": _cot_or_default(cot, 0, "Solve the selected registry equation subset."),
            }
        ]
    if template_id == "logic_condition":
        return [
            {
                "step_id": "s1",
                "operation": "check_condition",
                **({"principle_id": principle_id} if principle_id else {}),
                "inputs": {},
                "output": target_id,
                "depends_on": [],
                "public_cot": _cot_or_default(cot, 0, "Check the stated physical condition."),
            }
        ]
    if template_id == "logic_principle":
        return [
            {
                "step_id": "s1",
                "operation": "apply_logic_rule",
                **({"principle_id": principle_id} if principle_id else {}),
                "inputs": {},
                "output": target_id,
                "depends_on": [],
                "public_cot": _cot_or_default(cot, 0, "Apply the selected physical principle."),
            }
        ]
    if template_id == "multi_output":
        return [
            {
                "step_id": "s1",
                "operation": "solve_equation_subset",
                **({"formula_id": formula_id, "principle_id": principle_id} if formula_id else {}),
                "inputs": {},
                "output": "multi:values",
                "depends_on": [],
                "public_cot": _cot_or_default(cot, 0, "Solve the ordered target subset."),
            },
            {
                "step_id": "s2",
                "operation": "format_target",
                "inputs": {"values": "multi:values"},
                "output": target_id,
                "depends_on": ["s1"],
                "public_cot": _cot_or_default(cot, 1, "Format ordered target values."),
            },
        ]
    return []


def _safe_formula_id(value: Any) -> str | None:
    formula_id = str(value or "").strip()
    return formula_id if formula_id in FORMULA_REGISTRY else None


def _safe_principle_id(value: Any, formula_id: str | None) -> str | None:
    if formula_id and formula_id in FORMULA_REGISTRY:
        expected = FORMULA_REGISTRY[formula_id].principle_id
        return str(value).strip() if value and str(value).strip() == expected else expected
    text = str(value or "").strip()
    return text or None


def _safe_geometry_template_id(value: Any) -> str | None:
    geometry_id = _geometry_template_alias(str(value or "").strip())
    return geometry_id if geometry_id in GEOMETRY_TEMPLATE_IDS else None


def _cot_or_default(cot: list[str], index: int, default: str) -> str:
    if index < len(cot) and cot[index]:
        return cot[index]
    return default


def _normalize_llm_plan_steps(steps: list[Any]) -> list[dict]:
    """Normalize harmless schema drift while preserving the model's step DAG."""

    normalized: list[dict] = []
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        if not isinstance(step.get("inputs"), dict):
            step["inputs"] = {}
        if not isinstance(step.get("depends_on"), list):
            step["depends_on"] = []
        geometry_id = step.get("geometry_constructor_id")
        if isinstance(geometry_id, str):
            step["geometry_constructor_id"] = _geometry_template_alias(geometry_id)
        if step.get("operation") == "construct_geometry":
            step.pop("formula_id", None)
            step.pop("principle_id", None)
        normalized.append(step)
    return normalized


def _geometry_template_alias(template_id: str) -> str:
    normalized = template_id.strip()
    aliases = {
        "equilateral_triangle": "equilateral_triangle_vertex",
        "regular_triangle": "equilateral_triangle_vertex",
        "right_isosceles_triangle": "right_isosceles_triangle_vertex",
        "square": "square_vertex_field",
        "rectangle": "rectangle_vertex_field",
        "collinear": "two_charges_collinear",
    }
    return aliases.get(normalized, normalized)


def _completion_summary(raw_plan: dict | None, completed_plan: dict | None) -> dict:
    if not isinstance(raw_plan, dict) or not isinstance(completed_plan, dict):
        return {"applied": False}
    raw_keys = sorted(str(key) for key in raw_plan)
    filled = [
        key
        for key in ("task_type", "answer_type", "targets", "assumptions", "output_format", "source")
        if key not in raw_plan and key in completed_plan
    ]
    return {
        "applied": True,
        "raw_top_level_keys": raw_keys,
        "code_filled_keys": filled,
        "policy": "llm_owns_step_dag_code_fills_missing_static_contract",
    }


def _apply_front_repair_patch(front_payload: dict, patch_payload: dict | None) -> tuple[dict, dict]:
    if not isinstance(patch_payload, dict):
        return front_payload, {"stage": "front_patch_validator", "applied": False, "reason": "patch_missing_or_not_object"}
    if patch_payload.get("status") not in {"ok", "needs_fallback"}:
        return front_payload, {"stage": "front_patch_validator", "applied": False, "reason": f"patch_status:{patch_payload.get('status')}"}
    if _contains_forbidden_llm_key(patch_payload, extra={"formula", "formula_ids", "constant", "unit", "value", "si_value"}):
        return front_payload, {"stage": "front_patch_validator", "applied": False, "reason": "forbidden_payload_in_front_patch"}

    patch = _front_patch_body(patch_payload)
    question = str(front_payload.get("canonical_question") or "")
    patched = copy.deepcopy(front_payload)
    accepted: list[dict] = []
    rejected: list[dict] = []
    _apply_answer_type_patch(patched, patch, question, accepted, rejected)
    _apply_target_patches(patched, patch, question, accepted, rejected)
    _apply_relation_patches(patched, patch, question, accepted, rejected)
    _apply_symbol_dimension_patches(patched, patch, question, accepted, rejected)

    if not accepted:
        return front_payload, {
            "stage": "front_patch_validator",
            "applied": False,
            "reason": "no_valid_patch_items",
            "rejected": rejected[:8],
        }

    patched["canonical_structures"] = build_canonical_structures(patched)
    patched["parse_confidence"] = max(float(patched.get("parse_confidence") or 0.0), min(0.82, float(front_payload.get("parse_confidence") or 0.0) + 0.04))
    trace = patched.setdefault("trace", {})
    validation = {
        "stage": "front_patch_validator",
        "applied": True,
        "reason": "accepted_safe_semantic_patch",
        "accepted": accepted,
        "rejected": rejected[:8],
        "policy": "exact_evidence_no_values_no_units_no_formulas_no_coordinates",
    }
    trace["front_patch_validator"] = validation
    semantic_trace = trace.setdefault("semantic_parser", {})
    semantic_trace["llm_used"] = True
    semantic_trace["llm_front_patch_applied"] = True
    semantic_trace["goal_count"] = len(patched.get("goals") or [])
    semantic_trace["relation_count"] = len(patched.get("relations") or [])
    return patched, validation


def _front_patch_body(patch_payload: dict) -> dict:
    nested = patch_payload.get("proposed_ir_patch")
    if isinstance(nested, dict):
        merged = dict(patch_payload)
        merged.update(nested)
        return merged
    return patch_payload


def _apply_answer_type_patch(patched: dict, patch: dict, question: str, accepted: list[dict], rejected: list[dict]) -> None:
    answer_type = patch.get("answer_type_hint")
    if not answer_type:
        return
    if answer_type not in ANSWER_TYPES:
        rejected.append({"kind": "answer_type_hint", "reason": "unknown_answer_type", "value": answer_type})
        return
    evidence = patch.get("evidence") or patch.get("trigger_span") or ""
    if evidence and _evidence_span(question, str(evidence)) is None:
        rejected.append({"kind": "answer_type_hint", "reason": "evidence_not_in_question", "evidence": evidence})
        return
    patched["answer_type_hint"] = answer_type
    accepted.append({"kind": "answer_type_hint", "value": answer_type})


def _apply_target_patches(patched: dict, patch: dict, question: str, accepted: list[dict], rejected: list[dict]) -> None:
    goals = list(patched.get("goals") or [])
    for item in _limited_dicts(patch.get("target_overrides"), limit=4):
        dimension = item.get("dimension")
        if dimension and dimension not in FRONT_REPAIR_DIMENSIONS:
            rejected.append({"kind": "target", "reason": "unknown_dimension", "dimension": dimension})
            continue
        text = str(item.get("text") or item.get("evidence") or "").strip()
        evidence = str(item.get("evidence") or text).strip()
        span = _evidence_span(question, evidence)
        if not evidence or span is None:
            rejected.append({"kind": "target", "reason": "evidence_not_in_question", "evidence": evidence})
            continue
        goal_id = str(item.get("goal_id") or item.get("id") or f"goal:{len(goals) + 1}")
        payload = {
            "goal_id": goal_id,
            "text": text or evidence,
            "dimension": dimension,
            "symbol": _safe_symbol(item.get("symbol")),
            "span": span,
            "confidence": 0.68,
        }
        replaced = False
        for index, goal in enumerate(goals):
            if isinstance(goal, dict) and goal.get("goal_id") == goal_id:
                goals[index] = {**goal, **{key: value for key, value in payload.items() if value is not None}}
                replaced = True
                break
        if not replaced:
            goals.append(payload)
        accepted.append({"kind": "target", "goal_id": goal_id, "dimension": dimension, "evidence": evidence})
    if goals:
        patched["goals"] = goals
        patched["target_hints"] = [goal.get("text") for goal in goals if isinstance(goal, dict) and goal.get("text")][:6]


def _apply_relation_patches(patched: dict, patch: dict, question: str, accepted: list[dict], rejected: list[dict]) -> None:
    relations = list(patched.get("relations") or [])
    existing = {
        (relation.get("relation_type"), relation.get("qualifier"), relation.get("evidence"))
        for relation in relations
        if isinstance(relation, dict)
    }
    for item in _limited_dicts(patch.get("relation_hints"), limit=6):
        relation_type = item.get("relation_type")
        qualifier = item.get("qualifier")
        evidence = str(item.get("evidence") or "").strip()
        span = _evidence_span(question, evidence)
        if relation_type not in FRONT_REPAIR_RELATION_TYPES:
            rejected.append({"kind": "relation", "reason": "unknown_relation_type", "relation_type": relation_type})
            continue
        if qualifier not in FRONT_REPAIR_RELATION_QUALIFIERS:
            rejected.append({"kind": "relation", "reason": "unknown_relation_qualifier", "qualifier": qualifier})
            continue
        if not evidence or span is None:
            rejected.append({"kind": "relation", "reason": "evidence_not_in_question", "evidence": evidence})
            continue
        key = (relation_type, qualifier, evidence)
        if key in existing:
            continue
        relations.append(
            {
                "relation_type": relation_type,
                "subject": "question",
                "object": None,
                "qualifier": qualifier,
                "span": span,
                "evidence": evidence,
                "confidence": 0.66,
            }
        )
        existing.add(key)
        accepted.append({"kind": "relation", "relation_type": relation_type, "qualifier": qualifier, "evidence": evidence})
    patched["relations"] = relations


def _apply_symbol_dimension_patches(patched: dict, patch: dict, question: str, accepted: list[dict], rejected: list[dict]) -> None:
    for item in _limited_dicts(patch.get("symbol_dimension_overrides"), limit=8):
        symbol = _safe_symbol(item.get("symbol"))
        dimension = item.get("dimension")
        evidence = str(item.get("evidence") or symbol or "").strip()
        span = _evidence_span(question, evidence)
        if not symbol:
            rejected.append({"kind": "symbol_dimension", "reason": "missing_symbol"})
            continue
        if dimension not in FRONT_REPAIR_DIMENSIONS:
            rejected.append({"kind": "symbol_dimension", "reason": "unknown_dimension", "symbol": symbol, "dimension": dimension})
            continue
        if span is None:
            rejected.append({"kind": "symbol_dimension", "reason": "evidence_not_in_question", "symbol": symbol, "evidence": evidence})
            continue
        changed = False
        for collection_name in ("quantities", "symbolic_quantities"):
            collection = patched.get(collection_name) or []
            for quantity in collection:
                if isinstance(quantity, dict) and str(quantity.get("symbol") or "") == symbol:
                    quantity["dimension"] = dimension
                    quantity["confidence"] = min(float(quantity.get("confidence") or 0.7), 0.72)
                    changed = True
        if changed:
            accepted.append({"kind": "symbol_dimension", "symbol": symbol, "dimension": dimension, "evidence": evidence})
        else:
            rejected.append({"kind": "symbol_dimension", "reason": "symbol_not_found", "symbol": symbol})


def _limited_dicts(value: Any, limit: int) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value[:limit] if isinstance(item, dict)]


def _evidence_span(question: str, evidence: str) -> list[int] | None:
    if not evidence:
        return None
    start = question.lower().find(str(evidence).lower())
    if start < 0:
        return None
    return [start, start + len(evidence)]


def _safe_symbol(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if len(text) <= 24 and all(char.isalnum() or char in "_'′μΩωφθ" for char in text) else None


def _safe_string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:limit]:
        note = _safe_public_plan_text(item, limit=160)
        if note:
            out.append(note)
    return out


def _safe_public_plan_text(value: Any, limit: int = 220) -> str | None:
    """Keep optional LLM notes as audit labels, never as reasoning/results."""

    if not isinstance(value, str):
        return None
    text = _clip_text(value, limit).strip()
    if not text:
        return None
    lowered = text.lower()
    if any(cue in lowered for cue in PUBLIC_PLAN_TEXT_FORBIDDEN_CUES):
        return None
    if "=" in text:
        return None
    if PUBLIC_PLAN_NUMERIC_UNIT_RE.search(text):
        return None
    return text


def generate_local_llm_json(prompt: str, max_new_tokens: int = 256, schema: str = "semantic") -> dict:
    """Run the local adapter and parse the first JSON object from its output."""

    hard_timeout = _hard_timeout_seconds()
    if hard_timeout > 0 and os.environ.get("XAI_LLM_CHILD_PROCESS") != "1":
        return _generate_local_llm_json_with_hard_timeout(prompt, max_new_tokens=max_new_tokens, schema=schema, timeout_seconds=hard_timeout)
    return _generate_local_llm_json_inline(prompt, max_new_tokens=max_new_tokens, schema=schema)


def _generate_local_llm_json_inline(prompt: str, max_new_tokens: int = 256, schema: str = "semantic") -> dict:
    """Inline generation path used by the parent or the timeout worker."""

    readiness = check_local_llm_readiness()
    if not readiness["ready"]:
        return {"ok": False, "used": False, "reason": "runtime_unavailable", "readiness": readiness}
    try:
        import time

        load_started_at = time.monotonic()
        runtime = _load_runtime(readiness)
        load_elapsed = time.monotonic() - load_started_at
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        torch = runtime["torch"]
        encoded, response_prefix = _encode_json_prompt(tokenizer, prompt)
        device = _model_device(model)
        model_inputs, input_ids = _prepare_model_inputs(encoded, device)
        stopping_criteria = _json_stopping_criteria(
            tokenizer=tokenizer,
            prompt_token_count=int(input_ids.shape[-1]),
            response_prefix=response_prefix,
            schema=schema,
        )
        generate_kwargs = {
            **model_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if stopping_criteria is not None:
            generate_kwargs["stopping_criteria"] = stopping_criteria
        max_time = _generation_max_time_seconds()
        if max_time > 0:
            generate_kwargs["max_time"] = max_time
        started_at = time.monotonic()
        with torch.no_grad():
            output = model.generate(**generate_kwargs)
        elapsed = time.monotonic() - started_at
        generated_ids = output[0][input_ids.shape[-1] :]
        text = (response_prefix + tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()
        parsed = _extract_first_safe_json_object(text, schema=schema)
        safe = parsed is not None and _llm_json_is_safe(parsed, schema=schema)
        return {
            "ok": safe,
            "used": True,
            "reason": "ok" if safe else ("schema_rejected" if parsed is not None else "no_json_object"),
            "raw_text": text[:2000],
            "json": parsed,
            "schema": schema,
            "prompt_chars": len(prompt),
            "generated_tokens": int(generated_ids.shape[-1]),
            "elapsed_seconds": round(elapsed, 3),
            "runtime_load_seconds": round(load_elapsed, 3),
            "total_elapsed_seconds": round(load_elapsed + elapsed, 3),
            "max_time_seconds": max_time,
            "json_early_stop": stopping_criteria is not None,
            "json_early_stop_interval": JSON_STOP_CHECK_INTERVAL if stopping_criteria is not None else None,
            "max_new_tokens": max_new_tokens,
        }
    except Exception as exc:
        return {"ok": False, "used": False, "reason": f"generation_error:{type(exc).__name__}", "error": repr(exc)[:500], "readiness": readiness}


def _generate_local_llm_json_with_hard_timeout(prompt: str, max_new_tokens: int, schema: str, timeout_seconds: float) -> dict:
    """Run generation in a child process so CPU 7B calls cannot block the API."""

    import multiprocessing as mp
    import queue
    import time

    readiness = check_local_llm_readiness()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    started_at = time.monotonic()
    process = ctx.Process(target=_generation_worker, args=(result_queue, prompt, max_new_tokens, schema), daemon=True)
    process.start()
    process.join(timeout_seconds)
    elapsed = time.monotonic() - started_at
    if process.is_alive():
        process.terminate()
        process.join(5)
        _close_mp_queue(result_queue)
        return {
            "ok": False,
            "used": True,
            "reason": "hard_timeout",
            "schema": schema,
            "prompt_chars": len(prompt),
            "elapsed_seconds": round(elapsed, 3),
            "hard_timeout_seconds": timeout_seconds,
            "max_new_tokens": max_new_tokens,
            "readiness": readiness,
        }
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = {
            "ok": False,
            "used": True,
            "reason": f"worker_exited_without_result:{process.exitcode}",
            "schema": schema,
            "elapsed_seconds": round(elapsed, 3),
            "hard_timeout_seconds": timeout_seconds,
            "max_new_tokens": max_new_tokens,
            "readiness": readiness,
        }
    _close_mp_queue(result_queue)
    if isinstance(result, dict):
        result.setdefault("elapsed_seconds_parent", round(elapsed, 3))
        result.setdefault("hard_timeout_seconds", timeout_seconds)
        return result
    return {
        "ok": False,
        "used": True,
        "reason": "worker_returned_non_dict",
        "schema": schema,
        "elapsed_seconds": round(elapsed, 3),
        "hard_timeout_seconds": timeout_seconds,
        "max_new_tokens": max_new_tokens,
        "readiness": readiness,
    }


def _generation_worker(result_queue, prompt: str, max_new_tokens: int, schema: str) -> None:
    os.environ["XAI_LLM_CHILD_PROCESS"] = "1"
    try:
        result_queue.put(_generate_local_llm_json_inline(prompt, max_new_tokens=max_new_tokens, schema=schema))
    except Exception as exc:
        result_queue.put({"ok": False, "used": False, "reason": f"worker_error:{type(exc).__name__}", "error": repr(exc)[:500], "schema": schema})
    finally:
        _close_mp_queue(result_queue)


def _close_mp_queue(result_queue) -> None:
    try:
        result_queue.close()
    except Exception:
        pass
    try:
        result_queue.join_thread()
    except Exception:
        pass


def _load_runtime(readiness: dict) -> dict[str, Any]:
    cache_key = _runtime_cache_key(readiness)
    if _RUNTIME.get("cache_key") == cache_key:
        return _RUNTIME
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = readiness["adapter_dir"]
    base_model_dir = readiness["base_model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True, trust_remote_code=True)
    model_kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "dtype": os.environ.get("XAI_LLM_TORCH_DTYPE", "auto"),
    }
    device_map = os.environ.get("XAI_LLM_DEVICE_MAP", "").strip()
    if device_map and device_map.lower() not in {"none", "false", "0"}:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, **model_kwargs)
    model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=True, low_cpu_mem_usage=False)
    runtime_device = os.environ.get("XAI_LLM_DEVICE", "").strip()
    if runtime_device.lower() in {"auto", "none", "false", "0"}:
        runtime_device = ""
    if runtime_device and not (device_map and device_map.lower() not in {"none", "false", "0"}):
        model = model.to(runtime_device)
    model.eval()
    _RUNTIME.clear()
    _RUNTIME.update({"cache_key": cache_key, "torch": torch, "tokenizer": tokenizer, "model": model})
    return _RUNTIME


def _runtime_config_snapshot() -> dict[str, Any]:
    snapshot = {
        "device": os.environ.get("XAI_LLM_DEVICE") or "auto",
        "device_map": os.environ.get("XAI_LLM_DEVICE_MAP") or "none",
        "torch_dtype": os.environ.get("XAI_LLM_TORCH_DTYPE") or "auto",
        "generate_max_time_seconds": _generation_max_time_seconds(),
        "hard_timeout_seconds": _hard_timeout_seconds(),
        "semantic_audit_enabled": _truthy_env("XAI_LLM_ENABLE_SEMANTIC_AUDIT"),
        "json_prefill": os.environ.get("XAI_LLM_JSON_PREFILL", "1"),
        "json_early_stop": os.environ.get("XAI_LLM_JSON_EARLY_STOP", "1"),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    if snapshot["device"] == "mps":
        snapshot["mps_diagnostic"] = _mps_diagnostic()
    return snapshot


def _mps_diagnostic() -> dict[str, Any]:
    diagnostic: dict[str, Any] = {
        "python_machine": platform.machine(),
        "macos": platform.mac_ver()[0] or "unknown",
    }
    try:
        import torch

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        diagnostic["torch_version"] = getattr(torch, "__version__", None)
        diagnostic["mps_backend_present"] = mps is not None
        diagnostic["mps_built"] = bool(mps.is_built()) if mps is not None and hasattr(mps, "is_built") else None
        diagnostic["mps_available"] = bool(mps.is_available()) if mps is not None else None
    except Exception as exc:
        diagnostic["error"] = repr(exc)[:300]
    return diagnostic


def _readiness_cache_key(adapter_path: Path, base_path: Path, runtime_config: dict[str, Any]) -> str:
    return json.dumps(
        {
            "adapter_dir": str(adapter_path),
            "base_model_dir": str(base_path),
            "runtime_config": runtime_config,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _runtime_cache_key(readiness: dict) -> str:
    runtime_config = readiness.get("runtime_config") or {}
    model_config = {
        "adapter_dir": readiness.get("adapter_dir"),
        "base_model_dir": readiness.get("base_model_dir"),
        "device": runtime_config.get("device"),
        "device_map": runtime_config.get("device_map"),
        "torch_dtype": runtime_config.get("torch_dtype"),
    }
    return json.dumps(model_config, sort_keys=True, separators=(",", ":"))


def _model_device(model) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _prepare_model_inputs(encoded: Any, device: Any) -> tuple[dict[str, Any], Any]:
    if isinstance(encoded, dict) or hasattr(encoded, "items"):
        model_inputs = {}
        for key, value in encoded.items():
            model_inputs[key] = value.to(device) if device is not None and hasattr(value, "to") else value
        return model_inputs, model_inputs["input_ids"]
    input_ids = encoded.to(device) if device is not None and hasattr(encoded, "to") else encoded
    return {"input_ids": input_ids}, input_ids


def _json_stopping_criteria(
    tokenizer: Any,
    prompt_token_count: int,
    response_prefix: str,
    schema: str,
) -> Any | None:
    """Stop generation as soon as a complete safe JSON object is available."""

    if os.environ.get("XAI_LLM_JSON_EARLY_STOP", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:
        return None

    class SafeJsonStoppingCriteria(StoppingCriteria):
        def __init__(self) -> None:
            self.last_checked_token_count = -1

        def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001, ARG002
            generated_count = int(input_ids.shape[-1]) - prompt_token_count
            if generated_count < 8:
                return False
            if generated_count == self.last_checked_token_count:
                return False
            if generated_count % JSON_STOP_CHECK_INTERVAL != 0:
                return False
            self.last_checked_token_count = generated_count
            try:
                generated_ids = input_ids[0][prompt_token_count:]
                text = (response_prefix + tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()
            except Exception:
                return False
            parsed = _extract_first_safe_json_object(text, schema=schema)
            return parsed is not None and _llm_json_is_safe(parsed, schema=schema)

    return StoppingCriteriaList([SafeJsonStoppingCriteria()])


def _encode_json_prompt(tokenizer: Any, prompt: str) -> tuple[Any, str]:
    """Encode a prompt with an assistant JSON prefill to avoid R1 think-mode output."""

    if os.environ.get("XAI_LLM_JSON_PREFILL", "1") in {"0", "false", "False"}:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True), ""
        return tokenizer(prompt, return_tensors="pt").input_ids, ""
    json_prompt = (
        f"{prompt}\n\n"
        "The assistant response is prefilled with {\"status\":\". Continue with exactly one of "
        "ok, needs_fallback, or unsupported, then complete one valid JSON object. "
        "Do not include markdown, prose, <think>, or analysis text."
    )
    messages = [{"role": "user", "content": json_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return tokenizer(f"{rendered}{{\"status\":\"", return_tensors="pt"), '{"status":"'
        except TypeError:
            pass
    return tokenizer(f"{json_prompt}\n{{\"status\":\"", return_tensors="pt"), '{"status":"'


def _semantic_refinement_prompt(front_payload: dict) -> str:
    safe_front = {
        "q": _clip_text(front_payload.get("canonical_question"), 620),
        "answer_type_hint": front_payload.get("answer_type_hint"),
        "quantity_dims": [
            {"symbol": item.get("symbol"), "dim": item.get("dimension"), "raw": _clip_text(item.get("raw_text"), 60)}
            for item in (front_payload.get("quantities") or [])[:10]
            if isinstance(item, dict)
        ],
        "symbol_dims": [
            {"symbol": item.get("symbol"), "dim": item.get("dimension")}
            for item in (front_payload.get("symbolic_quantities") or [])[:8]
            if isinstance(item, dict)
        ],
        "targets": _compact_targets(front_payload),
        "relations": _compact_relations(front_payload),
        "warnings": list(front_payload.get("warnings") or [])[:6],
    }
    return (
        "JSON-only Physics semantic front-repair. Patch only language grounding; do not plan or solve. "
        "Allowed top keys: status, target_overrides, relation_hints, symbol_dimension_overrides, notes. "
        "Use exact evidence substrings from q. Do not output values, units, formulas, constants, coordinates, code, or final answers. "
        "If no safe patch is needed, return {\"status\":\"ok\",\"notes\":[\"no_patch_needed\"]}.\n"
        "Patch schema: target_overrides items may contain goal_id,text,dimension,symbol,evidence; "
        "relation_hints items may contain relation_type,qualifier,evidence; "
        "symbol_dimension_overrides items may contain symbol,dimension,evidence.\n"
        f"ALLOWED_DIMENSIONS:{json.dumps(sorted(FRONT_REPAIR_DIMENSIONS), separators=(',', ':'))}\n"
        f"ALLOWED_RELATIONS:{json.dumps({'types': sorted(FRONT_REPAIR_RELATION_TYPES), 'qualifiers': sorted(FRONT_REPAIR_RELATION_QUALIFIERS)}, separators=(',', ':'))}\n"
        f"FRONT:{json.dumps(safe_front, ensure_ascii=False, separators=(',', ':'))}"
    )


def _repair_prompt(front_payload: dict, error_packet: dict) -> str:
    safe_payload = {
        "q": _clip_text(front_payload.get("canonical_question"), 620),
        "answer_type_hint": front_payload.get("answer_type_hint"),
        "targets": _compact_targets(front_payload),
        "relations": _compact_relations(front_payload),
        "issues": error_packet.get("issues", []),
        "conflicts": error_packet.get("conflicts", []),
        "allowed_edits": error_packet.get("allowed_edits", []),
        "forbidden_outputs": error_packet.get("forbidden_outputs", []),
    }
    return (
        "JSON-only verifier-driven front-repair. Patch only semantic grounding; do not plan or solve. "
        "Allowed top keys: status, target_overrides, relation_hints, symbol_dimension_overrides, notes. "
        "Use exact evidence substrings from q. Do not output values, units, formulas, constants, coordinates, code, or final answers. "
        "If repair is not supported, return {\"status\":\"unsupported\",\"notes\":[\"no_safe_patch\"]}.\n"
        "Patch schema: target_overrides items may contain goal_id,text,dimension,symbol,evidence; "
        "relation_hints items may contain relation_type,qualifier,evidence; "
        "symbol_dimension_overrides items may contain symbol,dimension,evidence.\n"
        f"ALLOWED_DIMENSIONS:{json.dumps(sorted(FRONT_REPAIR_DIMENSIONS), separators=(',', ':'))}\n"
        f"ALLOWED_RELATIONS:{json.dumps({'types': sorted(FRONT_REPAIR_RELATION_TYPES), 'qualifiers': sorted(FRONT_REPAIR_RELATION_QUALIFIERS)}, separators=(',', ':'))}\n"
        f"ERROR_PACKET:{json.dumps(safe_payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def _solve_plan_prompt(front_payload: dict, route_result=None, graph_selection=None) -> str:
    candidate_formula_ids = _planning_candidate_formula_ids(front_payload, route_result, graph_selection)
    route_task_type = getattr(route_result, "task_type", None)
    available_dimensions = _front_available_dimensions(front_payload)
    target_dimensions = _front_target_dimensions(front_payload)
    route_reasons = list(getattr(route_result, "reasons", []) or [])[:4]
    if not _truthy_env("XAI_LLM_VERBOSE_PROMPT"):
        return _compact_solve_plan_prompt(front_payload, route_result, candidate_formula_ids, route_task_type)
    safe_payload = {
        "question": front_payload.get("canonical_question"),
        "answer_type_hint": front_payload.get("answer_type_hint"),
        "quantities": front_payload.get("quantities", [])[:14],
        "goals": front_payload.get("goals", [])[:6],
        "relations": front_payload.get("relations", [])[:8],
        "constraints": front_payload.get("constraints", [])[:8],
        "canonical_structures": front_payload.get("canonical_structures", {}),
        "route": route_result.to_dict() if hasattr(route_result, "to_dict") else {},
        "candidate_formula_ids": candidate_formula_ids,
        "answer_format_contract": prompt_answer_format_contract(front_payload, route_result),
        "formula_catalog": formula_prompt_pack(
            route_task_type=route_task_type,
            candidate_formula_ids=candidate_formula_ids,
            available_dimensions=available_dimensions,
            target_dimensions=target_dimensions,
            route_reasons=route_reasons,
        ),
    }
    return (
        "You are the structured solve-plan proposer for a Physics XAI system. "
        "The user-facing CoT must be represented only as per-step public_cot labels inside the JSON plan. "
        "Do not solve numerically. Do not output final answers. Do not invent formulas, constants, units, coordinates, or code. "
        "Return only one JSON object matching the schema. Every operation must be from the allowed list, and every formula_id must come from formula_catalog.allowed_formula_ids. "
        "Choose formula IDs using formula_catalog.decision_evidence.selection_rule. "
        "Each step must include public_cot: one short public action label, not hidden chain-of-thought. "
        "public_cot must not contain arithmetic, numeric results, equations, final-answer wording, coordinates, or free-form reasoning. "
        "When status is ok, omit notes; notes are only for needs_fallback or unsupported. "
        "Use registry IDs such as formula_id/principle_id instead of formula text.\n\n"
        f"STRICT_SOLVE_PLAN_SCHEMA:\n{json.dumps(_solve_plan_schema_hint(), ensure_ascii=False)}\n\n"
        f"ANSWER_FORMAT_CONTRACT:\n{json.dumps(prompt_answer_format_contract(front_payload, route_result), ensure_ascii=False)}\n\n"
        f"PUBLIC_COT_RULES:\n{json.dumps(_public_cot_rules(), ensure_ascii=False)}\n\n"
        f"NORMALIZED_IR:\n{json.dumps(safe_payload, ensure_ascii=False)}"
    )


def _solve_plan_repair_prompt(front_payload: dict, error_packet: dict, route_result=None, graph_selection=None) -> str:
    candidate_formula_ids = _planning_candidate_formula_ids(front_payload, route_result, graph_selection)
    route_task_type = getattr(route_result, "task_type", None)
    available_dimensions = _front_available_dimensions(front_payload)
    target_dimensions = _front_target_dimensions(front_payload)
    route_reasons = list(getattr(route_result, "reasons", []) or [])[:4]
    if not _truthy_env("XAI_LLM_VERBOSE_PROMPT"):
        return _compact_solve_plan_prompt(front_payload, route_result, candidate_formula_ids, route_task_type, error_packet=error_packet)
    base = {
        "question": front_payload.get("canonical_question"),
        "answer_type_hint": front_payload.get("answer_type_hint"),
        "quantities": front_payload.get("quantities", [])[:14],
        "goals": front_payload.get("goals", [])[:6],
        "relations": front_payload.get("relations", [])[:8],
        "constraints": front_payload.get("constraints", [])[:8],
        "canonical_structures": front_payload.get("canonical_structures", {}),
        "route": route_result.to_dict() if hasattr(route_result, "to_dict") else {},
        "candidate_formula_ids": candidate_formula_ids,
        "answer_format_contract": prompt_answer_format_contract(front_payload, route_result),
        "formula_catalog": formula_prompt_pack(
            route_task_type=route_task_type,
            candidate_formula_ids=candidate_formula_ids,
            available_dimensions=available_dimensions,
            target_dimensions=target_dimensions,
            route_reasons=route_reasons,
        ),
    }
    return (
        "You are repairing a rejected Structured Solve Plan for a Physics XAI system. "
        "Return exactly one full replacement JSON plan. Do not solve numerically. "
        "Do not output final answers, coordinates, new formulas, new constants, new units, code, or free-form chain-of-thought. "
        "Every step must include public_cot as a short public action label. "
        "public_cot must not contain arithmetic, numeric results, equations, final-answer wording, coordinates, or free-form reasoning. "
        "When status is ok, omit notes; notes are only for needs_fallback or unsupported. "
        "Fix only the plan structure, step dependencies, registry IDs, and bindings allowed in ERROR_PACKET. "
        "Every operation must come from allowed_operations. Choose formula IDs using formula_catalog.decision_evidence.selection_rule.\n\n"
        f"STRICT_SOLVE_PLAN_SCHEMA:\n{json.dumps(_solve_plan_schema_hint(), ensure_ascii=False)}\n\n"
        f"ANSWER_FORMAT_CONTRACT:\n{json.dumps(prompt_answer_format_contract(front_payload, route_result), ensure_ascii=False)}\n\n"
        f"PUBLIC_COT_RULES:\n{json.dumps(_public_cot_rules(), ensure_ascii=False)}\n\n"
        f"ERROR_PACKET:\n{json.dumps(error_packet, ensure_ascii=False)}\n\n"
        f"NORMALIZED_IR:\n{json.dumps(base, ensure_ascii=False)}"
    )


def _compact_solve_plan_prompt(
    front_payload: dict,
    route_result=None,
    candidate_formula_ids: list[str] | None = None,
    route_task_type: str | None = None,
    error_packet: dict | None = None,
) -> str:
    """Small CPU-friendly prompt for direct local planning."""

    available_dimensions = _front_available_dimensions(front_payload)
    target_dimensions = _front_target_dimensions(front_payload)
    route_reasons = list(getattr(route_result, "reasons", []) or [])[:4]
    prompt_pack = formula_prompt_pack(
        route_task_type=route_task_type,
        candidate_formula_ids=candidate_formula_ids or [],
        available_dimensions=available_dimensions,
        target_dimensions=target_dimensions,
        route_reasons=route_reasons,
    )
    targets = _compact_targets(front_payload)
    include_geometry_templates = _needs_geometry_templates(front_payload, route_task_type, prompt_pack)
    plan_cards = _compact_plan_cards(front_payload, route_result, prompt_pack, targets)
    compact_ir = {
        "q": _clip_text(front_payload.get("canonical_question"), 380),
        "route": _compact_route(route_result, route_task_type, front_payload.get("answer_type_hint")),
        "facts": _compact_facts(front_payload),
        "targets": targets,
        "relations": _compact_relations(front_payload),
        "geometry": _compact_geometry(front_payload, include_template_ids=include_geometry_templates),
        "topology": _compact_topology(front_payload),
        "formula_menu": _compact_formula_menu(prompt_pack),
        "plan_contract": {
            "return_keys": ["status", "plan_card_id"],
            "code_expands": ["steps", "targets", "output_format"],
            "allowed_operations": sorted(PLAN_OPERATION_IDS),
        },
        "plan_cards": plan_cards,
        "operation_templates": _operation_templates(front_payload, route_result, prompt_pack),
    }
    if error_packet:
        compact_ir["repair_errors"] = {
            "issues": list(error_packet.get("issues") or [])[:8],
            "allowed_operations": list(error_packet.get("allowed_operations") or [])[:12],
        }
    return (
        "JSON-only Physics planner. Choose exactly one DATA.plan_cards entry.\n"
        "For ok output exactly {\"status\":\"ok\",\"plan_card_id\":\"pN\"} using one listed card_id.\n"
        "Do NOT output steps, formula_id, geometry_template_id, public_cot, task_type, targets, inputs, assumptions, output_format, notes, equations, or prose.\n"
        "No numeric solving, final answer, formulas text, constants, units, coordinates, code, markdown, or free-form CoT.\n"
        "Copy only the plan_card_id. Do not invent IDs or use placeholder words.\n"
        "You own the executable step DAG by selecting the plan card. Code expands, computes, and verifies.\n"
        f"DATA:{json.dumps(compact_ir, ensure_ascii=False, separators=(',', ':'))}"
    )


def _compact_route(route_result, route_task_type: str | None, answer_type_hint: str | None) -> dict:
    payload = route_result.to_dict() if hasattr(route_result, "to_dict") else {}
    return {
        "task_type": route_task_type or payload.get("task_type") or "unknown",
        "answer_type": answer_type_hint or payload.get("answer_type") or "unknown",
        "confidence": payload.get("confidence"),
        "reasons": list(payload.get("reasons") or [])[:3],
    }


def _needs_geometry_templates(front_payload: dict, route_task_type: str | None, prompt_pack: dict) -> bool:
    if route_task_type in {"coulomb_force", "electric_field_point"}:
        return True
    if any(isinstance(card, dict) and card.get("branch") == "spatial_vector" for card in (prompt_pack.get("cards") or [])):
        return True
    return any(
        isinstance(relation, dict) and relation.get("relation_type") == "geometry"
        for relation in (front_payload.get("relations") or [])
    )


def _compact_facts(front_payload: dict) -> dict:
    quantities = []
    for quantity in (front_payload.get("quantities") or [])[:12]:
        if not isinstance(quantity, dict):
            continue
        quantities.append(
            {
                "symbol": quantity.get("symbol"),
                "dim": quantity.get("dimension"),
                "value": quantity.get("value"),
                "unit": quantity.get("unit"),
                "entity": quantity.get("entity_id"),
                "state": quantity.get("state_id"),
            }
        )
    symbolic = []
    for quantity in (front_payload.get("symbolic_quantities") or [])[:10]:
        if not isinstance(quantity, dict):
            continue
        symbolic.append(
            {
                "symbol": quantity.get("symbol"),
                "dim": quantity.get("dimension"),
                "unit": quantity.get("unit"),
            }
        )
    return {
        "quantities": quantities,
        "symbolic_quantities": symbolic,
        "implicit_rule_ids": [fact.get("rule_id") for fact in (front_payload.get("implicit_facts") or [])[:6] if isinstance(fact, dict)],
    }


def _compact_targets(front_payload: dict) -> list[dict]:
    targets = []
    answer_type = str(front_payload.get("answer_type_hint") or "unknown")
    for index, goal in enumerate((front_payload.get("goals") or [])[:4], start=1):
        if not isinstance(goal, dict):
            continue
        targets.append(
            {
                "id": goal.get("goal_id") or f"target:{index}",
                "quantity": goal.get("dimension") or goal.get("quantity") or "unknown",
                "symbol": goal.get("symbol"),
                "unit": None if answer_type in {"conceptual", "yes_no"} else goal.get("unit"),
                "text": _clip_text(goal.get("text") or goal.get("raw_text"), 120),
            }
        )
    return targets or [{"id": "target:1", "quantity": "unknown", "symbol": None, "unit": None, "text": ""}]


def _compact_relations(front_payload: dict) -> list[dict]:
    relations = []
    allowed_keys = ("relation_type", "type", "subtype", "subject", "object", "qualifier", "evidence", "lhs", "rhs")
    for relation in (front_payload.get("relations") or [])[:8]:
        if not isinstance(relation, dict):
            continue
        item = {key: _clip_text(relation.get(key), 96) for key in allowed_keys if relation.get(key) is not None}
        if item:
            relations.append(item)
    for relation in (front_payload.get("symbolic_relations") or [])[:6]:
        if not isinstance(relation, dict):
            continue
        item = {key: _clip_text(relation.get(key), 96) for key in ("lhs", "rhs", "operator", "raw_text") if relation.get(key) is not None}
        if item:
            relations.append(item)
    return relations


def _compact_geometry(front_payload: dict, include_template_ids: bool = True) -> dict:
    structures = front_payload.get("canonical_structures") or {}
    geometry = structures.get("geometry") if isinstance(structures, dict) else {}
    triangles = geometry.get("triangles") if isinstance(geometry, dict) else []
    squares = geometry.get("squares") if isinstance(geometry, dict) else []
    payload = {
        "triangles": [
            {
                "labels": triangle.get("labels"),
                "right_angle_at": triangle.get("right_angle_at"),
                "canonical_right_angle_at": triangle.get("canonical_right_angle_at"),
                "side_symbols": triangle.get("side_symbols"),
                "charge_symbols": triangle.get("charge_symbols"),
            }
            for triangle in (triangles or [])[:3]
            if isinstance(triangle, dict)
        ],
        "squares": [
            {
                "labels": square.get("labels"),
                "side_symbols": square.get("side_symbols"),
                "charge_symbols": square.get("charge_symbols"),
            }
            for square in (squares or [])[:2]
            if isinstance(square, dict)
        ],
        "template_hints": [
            relation.get("qualifier")
            for relation in (front_payload.get("relations") or [])[:6]
            if isinstance(relation, dict) and relation.get("relation_type") == "geometry" and relation.get("qualifier")
        ],
    }
    if include_template_ids:
        payload["allowed_template_ids"] = sorted(GEOMETRY_TEMPLATE_IDS)
    return payload


def _compact_topology(front_payload: dict) -> dict:
    topology = front_payload.get("topology_graph") or front_payload.get("topology") or {}
    if not isinstance(topology, dict):
        return {}
    return {
        "canonical_form": topology.get("canonical_form"),
        "is_complex": topology.get("is_complex"),
        "node_count": len(topology.get("nodes") or []),
        "edge_count": len(topology.get("edges") or []),
        "ambiguity": list(topology.get("ambiguity") or [])[:4],
    }


def _compact_formula_menu(prompt_pack: dict) -> dict:
    evidence_by_id = {
        item.get("formula_id"): item
        for item in ((prompt_pack.get("decision_evidence") or {}).get("candidates") or [])
        if isinstance(item, dict)
    }
    cards = []
    for card in prompt_pack.get("cards") or []:
        if not isinstance(card, dict):
            continue
        evidence = evidence_by_id.get(card.get("id")) or {}
        cards.append(
            {
                "id": card.get("id"),
                "principle": card.get("principle"),
                "branch": card.get("branch"),
                "selected": evidence.get("selected_by_graph"),
                "missing": evidence.get("missing_dimensions") or [],
            }
        )
    return {
        "allowed_formula_ids": list(prompt_pack.get("allowed_formula_ids") or []),
        "candidate_formula_ids": list(prompt_pack.get("candidate_formula_ids") or []),
        "cards": cards,
        "selection_rule": (prompt_pack.get("decision_evidence") or {}).get("selection_rule"),
    }


def _operation_templates(front_payload: dict, route_result, prompt_pack: dict) -> list[dict]:
    answer_type = str(front_payload.get("answer_type_hint") or getattr(route_result, "answer_type", "unknown"))
    task_type = str(getattr(route_result, "task_type", None) or prompt_pack.get("route_task_type") or "unknown")
    cards = prompt_pack.get("cards") or []
    branches = {card.get("branch") for card in cards if isinstance(card, dict)}
    if task_type == "multi_output" or answer_type == "multi_output":
        return [
            {
                "when": "multiple ordered targets",
                "steps": ["solve_equation_subset", "format_target"],
            },
        ]
    if answer_type in {"conceptual", "yes_no"}:
        return [{"when": "principle/yes-no question", "steps": ["apply_logic_rule or check_condition"]}]
    if "spatial_vector" in branches or task_type in {"coulomb_force", "electric_field_point"}:
        return [
            {
                "when": "geometry/vector relation is present",
                "steps": ["construct_geometry", "compute_pairwise_force or resolve_vector_components", "vector_sum when multiple components remain"],
            },
            {"when": "no geometry template is justified", "steps": ["needs_fallback"]},
        ]
    if "algebraic_system" in branches:
        return [{"when": "small coupled registry equation subset", "steps": ["solve_equation_subset"]}]
    return [{"when": "direct scalar registry formula fits dimensions and target", "steps": ["apply_formula"]}]


def _compact_plan_cards(front_payload: dict, route_result, prompt_pack: dict, targets: list[dict]) -> list[dict]:
    """Build route-local plan choices so the LLM selects, not writes schemas."""

    target_id = targets[0].get("id") if targets and isinstance(targets[0], dict) else "goal:1"
    cards: list[dict] = []
    geometry_id = _preferred_geometry_template_id(front_payload)

    def add_card(card: dict) -> None:
        cards.append({"card_id": f"p{len(cards) + 1}", **card})

    for card in prompt_pack.get("cards") or []:
        if not isinstance(card, dict):
            continue
        formula_id = card.get("id")
        if formula_id not in FORMULA_REGISTRY:
            continue
        branch = card.get("branch") or formula_execution_branch(str(formula_id))
        spec = FORMULA_REGISTRY[str(formula_id)]
        if branch == "spatial_vector":
            if geometry_id:
                spatial_operation = _spatial_card_operation(str(formula_id), str(getattr(route_result, "task_type", "") or ""))
                add_card(
                    {
                        "plan_template_id": "spatial_pairwise_force" if spatial_operation == "compute_pairwise_force" else "spatial_vector_resolution",
                        "formula_id": formula_id,
                        "principle_id": spec.principle_id,
                        "geometry_template_id": geometry_id,
                        "operations": ["construct_geometry", spatial_operation],
                        "public_cot": ["Construct accepted geometry.", "Resolve vector contributions."],
                        "target": target_id,
                    }
                )
            add_card(
                {
                    "plan_template_id": "vector_resolution",
                    "formula_id": formula_id,
                    "principle_id": spec.principle_id,
                    "operations": ["resolve_vector_components"],
                    "public_cot": ["Resolve accepted vector components."],
                    "target": target_id,
                }
            )
        elif branch in {"algebraic_system", "topology", "measurement"}:
            add_card(
                {
                    "plan_template_id": "equation_subset",
                    "formula_id": formula_id,
                    "principle_id": spec.principle_id,
                    "operations": ["solve_equation_subset"],
                    "public_cot": ["Solve the selected registry equation subset."],
                    "target": target_id,
                }
            )
        elif branch == "logic":
            add_card(
                {
                    "plan_template_id": "logic_condition" if str(formula_id) == "yes_no_direct" else "logic_principle",
                    "formula_id": formula_id,
                    "principle_id": spec.principle_id,
                    "operations": ["check_condition" if str(formula_id) == "yes_no_direct" else "apply_logic_rule"],
                    "public_cot": ["Check the stated physical condition." if str(formula_id) == "yes_no_direct" else "Apply the selected physical principle."],
                    "target": target_id,
                }
            )
        elif branch == "multi_output":
            add_card(
                {
                    "plan_template_id": "multi_output",
                    "formula_id": formula_id,
                    "principle_id": spec.principle_id,
                    "operations": ["solve_equation_subset", "format_target"],
                    "public_cot": ["Solve the ordered target subset.", "Format ordered target values."],
                    "target": target_id,
                }
            )
        else:
            add_card(
                {
                    "plan_template_id": "direct_formula",
                    "formula_id": formula_id,
                    "principle_id": spec.principle_id,
                    "operations": ["apply_formula"],
                    "public_cot": ["Apply the selected registry relation to accepted facts."],
                    "target": target_id,
                }
            )
    if not cards and str(front_payload.get("answer_type_hint") or "") in {"conceptual", "yes_no"}:
        add_card(
            {
                "plan_template_id": "logic_condition" if front_payload.get("answer_type_hint") == "yes_no" else "logic_principle",
                "principle_id": "conceptual_core",
                "operations": ["check_condition" if front_payload.get("answer_type_hint") == "yes_no" else "apply_logic_rule"],
                "public_cot": ["Check the stated physical condition." if front_payload.get("answer_type_hint") == "yes_no" else "Apply the selected physical principle."],
                "target": target_id,
            }
        )
    return cards[:6]


def _spatial_card_operation(formula_id: str, route_task_type: str) -> str:
    if route_task_type == "coulomb_force" and formula_id == "coulomb_force_triangle_sides":
        return "compute_pairwise_force"
    return "resolve_vector_components"


def _preferred_geometry_template_id(front_payload: dict) -> str | None:
    qualifier_to_template = {
        "collinear": "two_charges_collinear",
        "equilateral_triangle": "equilateral_triangle_vertex",
        "external_point_on_line": "external_point_on_line",
        "midpoint": "point_on_midpoint",
        "perpendicular_bisector": "point_on_perpendicular_bisector",
        "rectangle": "rectangle_vertex_field",
        "right_isosceles_triangle": "right_isosceles_triangle_vertex",
        "square": "square_vertex_field",
        "triangle": "triangle_sides",
    }
    for relation in front_payload.get("relations") or []:
        if not isinstance(relation, dict) or relation.get("relation_type") != "geometry":
            continue
        template_id = qualifier_to_template.get(str(relation.get("qualifier") or ""))
        if template_id in GEOMETRY_TEMPLATE_IDS:
            return template_id
    matches = match_geometry_templates(front_payload)
    if matches:
        return matches[0].template_id
    geometry = (front_payload.get("canonical_structures") or {}).get("geometry") if isinstance(front_payload.get("canonical_structures"), dict) else {}
    triangles = geometry.get("triangles") if isinstance(geometry, dict) else []
    squares = geometry.get("squares") if isinstance(geometry, dict) else []
    if squares:
        return "square_vertex_field"
    if triangles:
        relation_qualifiers = {
            str(relation.get("qualifier") or "")
            for relation in front_payload.get("relations") or []
            if isinstance(relation, dict) and relation.get("relation_type") == "geometry"
        }
        if "equilateral_triangle" in relation_qualifiers:
            return "equilateral_triangle_vertex"
        if "right_isosceles_triangle" in relation_qualifiers:
            return "right_isosceles_triangle_vertex"
        return "triangle_sides"
    return None


def _clip_text(value: Any, limit: int) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _front_available_dimensions(front_payload: dict) -> list[str]:
    dimensions: list[str] = []
    for quantity in [*(front_payload.get("quantities") or []), *(front_payload.get("symbolic_quantities") or [])]:
        if not isinstance(quantity, dict):
            continue
        dimension = quantity.get("dimension")
        if dimension and dimension != "constant":
            dimensions.append(str(dimension))
    return dimensions


def _front_target_dimensions(front_payload: dict) -> list[str]:
    dimensions: list[str] = []
    for goal in front_payload.get("goals") or []:
        if not isinstance(goal, dict):
            continue
        dimension = goal.get("dimension") or goal.get("quantity")
        if dimension and dimension not in dimensions:
            dimensions.append(str(dimension))
    return dimensions


def _solve_plan_schema_hint() -> dict:
    return {
        "status": "ok | needs_fallback | unsupported",
        "task_type": "must match the provided route when possible",
        "answer_type": "numeric | symbolic | conceptual | yes_no | multi_output | unknown",
        "targets": [{"id": "target id", "quantity": "dimension", "symbol": None, "unit": "requested/base unit"}],
        "assumptions": [{"assumption_id": "triggered rule id only", "trigger_span": "text evidence"}],
        "steps": [
            {
                "step_id": "s1",
                "operation": "apply_formula | construct_geometry | compute_pairwise_force | resolve_vector_components | vector_sum | apply_logic_rule | solve_equation_subset | format_target",
                "formula_id": "registry formula id when needed",
                "principle_id": "registry principle id when needed",
                "geometry_constructor_id": "known geometry template id when needed",
                "inputs": {},
                "output": "intermediate or target id",
                "depends_on": [],
                "public_cot": "short public action label; no arithmetic, no equations, no final answer",
            }
        ],
        "output_format": {
            "format_kind": "numeric_scalar | dimensionless_numeric | symbolic_expression | conceptual_text | yes_no | ordered_multi_output | controlled_fallback",
            "ordered_targets": [],
            "preferred_unit": None,
            "target_count": 1,
        },
        "notes": "only for needs_fallback/unsupported; omit when status is ok",
    }


def _public_cot_rules() -> list[str]:
    return [
        "Required for every step when status is ok.",
        "One sentence or fragment under 220 characters.",
        "Describe the executable action only.",
        "Use registry IDs, not formula text.",
        "No arithmetic and no numeric result with unit.",
        "No equations or equals signs.",
        "No coordinates or diagram invention.",
        "No final-answer wording.",
    ]


def _extract_json_object(text: str) -> dict | None:
    candidates = _extract_json_objects(text)
    return candidates[0] if candidates else None


def _extract_first_safe_json_object(text: str, schema: str) -> dict | None:
    candidates = _extract_json_objects(text)
    for candidate in candidates:
        if _llm_json_is_safe(candidate, schema=schema):
            return candidate
    if schema == "solve_plan":
        partial = _extract_partial_solve_plan_header(text)
        if partial is not None and _llm_json_is_safe(partial, schema=schema):
            return partial
    return candidates[0] if candidates else None


def _extract_partial_solve_plan_header(text: str) -> dict | None:
    """Recover only safe registry IDs from a truncated solve-plan prefix.

    If the model starts with a valid compact plan choice but then drifts into a
    long CoT string, the JSON may be cut off. We recover only allowlisted IDs
    that appeared before the drift and discard all generated reasoning text.
    """

    status = _extract_json_string_field(text, "status")
    if status not in {"ok", "needs_fallback", "unsupported"}:
        return None
    payload: dict[str, Any] = {"status": status}
    card_id = _extract_json_string_field(text, "plan_card_id")
    if card_id:
        payload["plan_card_id"] = card_id
        return payload
    template_id = _extract_json_string_field(text, "plan_template_id")
    formula_id = _extract_json_string_field(text, "formula_id")
    geometry_id = _extract_json_string_field(text, "geometry_template_id")
    if template_id:
        payload["plan_template_id"] = template_id
    if formula_id in FORMULA_REGISTRY:
        payload["formula_id"] = formula_id
    if geometry_id:
        geometry_id = _geometry_template_alias(geometry_id)
        if geometry_id in GEOMETRY_TEMPLATE_IDS:
            payload["geometry_template_id"] = geometry_id
    if status == "ok" and not (payload.get("plan_template_id") and payload.get("formula_id")):
        return None
    return payload


def _extract_json_string_field(text: str, field: str) -> str | None:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return None


def _extract_json_objects(text: str) -> list[dict]:
    objects: list[dict] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        parsed = _extract_json_object_at(text, start)
        if parsed is not None:
            objects.append(parsed)
    return objects


def _extract_json_object_at(text: str, start: int) -> dict | None:
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _llm_json_is_safe(payload: dict, schema: str = "semantic") -> bool:
    if schema == "solve_plan":
        return _llm_solve_plan_json_is_safe(payload)
    keys = {str(key).lower() for key in payload}
    if "status" not in keys:
        return False
    if not isinstance(payload.get("status"), str) or payload.get("status") not in {"ok", "needs_fallback", "unsupported"}:
        return False
    if not keys <= SEMANTIC_TOP_LEVEL_KEYS:
        return False
    if _contains_forbidden_llm_key(payload, extra={"formula", "formula_ids", "constant", "unit", "value", "si_value"}):
        return False
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    return not any(cue in serialized for cue in ["final answer", "```", "__import__", "lambda "])


def _llm_solve_plan_json_is_safe(payload: dict) -> bool:
    keys = {str(key).lower() for key in payload}
    if "status" not in keys:
        return False
    if not isinstance(payload.get("status"), str) or payload.get("status") not in {"ok", "needs_fallback", "unsupported"}:
        return False
    if not keys <= SOLVE_PLAN_TOP_LEVEL_KEYS:
        return False
    if "targets" in payload and not isinstance(payload.get("targets"), list):
        return False
    if "assumptions" in payload and not isinstance(payload.get("assumptions"), list):
        return False
    if "notes" in payload and not isinstance(payload.get("notes"), list):
        return False
    if "output_format" in payload and not isinstance(payload.get("output_format"), dict):
        return False
    if "steps" in payload and not isinstance(payload.get("steps"), list):
        return False
    if "public_cot" in payload and not isinstance(payload.get("public_cot"), list):
        return False
    if payload.get("status") == "ok" and not (payload.get("steps") or payload.get("plan_template_id") or payload.get("plan_card_id")):
        return False
    if _contains_forbidden_llm_key(payload):
        return False
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    if any(cue in serialized for cue in PUBLIC_PLAN_TEXT_FORBIDDEN_CUES):
        return False
    if "=" in serialized:
        return False
    if PUBLIC_PLAN_NUMERIC_UNIT_RE.search(serialized):
        return False
    return True


def _contains_forbidden_llm_key(payload: Any, extra: set[str] | None = None) -> bool:
    forbidden = FORBIDDEN_LLM_KEYS | set(extra or set())
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in forbidden:
                return True
            if _contains_forbidden_llm_key(value, extra=extra):
                return True
    if isinstance(payload, list):
        return any(_contains_forbidden_llm_key(item, extra=extra) for item in payload)
    return False


def _read_adapter_config(path: Path) -> dict:
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _front_repair_enabled() -> bool:
    return _truthy_env("XAI_LLM_ENABLE_FRONT_REPAIR")


def _front_refinement_needed(front_payload: dict) -> bool:
    if float(front_payload.get("parse_confidence") or 0.0) < 0.72:
        return True
    if front_payload.get("answer_type_hint") in {None, "unknown"}:
        return True
    if not front_payload.get("goals"):
        return True
    if any(isinstance(goal, dict) and not goal.get("dimension") for goal in front_payload.get("goals") or []):
        return True
    if front_payload.get("warnings"):
        return True
    topology = front_payload.get("topology_graph") or {}
    if topology.get("ambiguity"):
        return True
    return False


def _max_new_tokens_for(schema: str, default: int) -> int:
    env_name = f"XAI_LLM_{schema.upper()}_MAX_NEW_TOKENS"
    raw = os.environ.get(env_name) or os.environ.get("XAI_LLM_MAX_NEW_TOKENS")
    if raw is None:
        return default
    try:
        return max(32, int(raw))
    except (TypeError, ValueError):
        return default


def _generation_max_time_seconds() -> float:
    raw = os.environ.get("XAI_LLM_GENERATE_MAX_TIME_SECONDS", "0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _hard_timeout_seconds() -> float:
    raw = os.environ.get("XAI_LLM_HARD_TIMEOUT_SECONDS", "0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0
