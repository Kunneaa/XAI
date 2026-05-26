"""Manual runner for the core Physics XAI pipeline.

Run:

    PYTHONPATH=src python3 manual_question_test.py
    PYTHONPATH=src python3 manual_question_test.py "A resistor R = 10 Ω has U = 20 V. Find I."
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from xai_pipeline.core.api import handle_request
from xai_pipeline.planning.local_llm import check_local_llm_readiness, generate_local_llm_json


QUESTION = "In a vacuum, two equal point charges q1 and q2, each having a magnitude of 16 × 10^-8 C, are fixed at points A and B, which are 8 cm apart. Calculate the magnitude of the resultant electric field at point C if the distances from C to A and from C to B are both 8 cm."
TIMEOUT_SECONDS = -1.0
LLM_PROBE_PROMPT = (
    "Return exactly one JSON object. Do not solve and do not explain. "
    "Schema: status must be one of ok, needs_fallback, unsupported; "
    "missing_fields, ambiguous_references, candidate_target_texts, and notes must be arrays. "
    'Example valid response: {"status":"ok","missing_fields":[],"ambiguous_references":[],"candidate_target_texts":["current"],"notes":["probe"]}'
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one question through NSP-Core.")
    parser.add_argument("question", nargs="*", help="Optional question override.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON response.")
    parser.add_argument("--output-json", help="Write a compact important JSON run report to this file after the run.")
    parser.add_argument("--output-json-full", action="store_true", help="With --output-json, write the full raw API response instead of the compact report.")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="Request timeout seconds; negative disables the manual-runner deadline.")
    parser.add_argument("--enable-llm", action="store_true", help="Enable local LLM structured solve-plan proposal/repair boundary.")
    parser.add_argument("--direct-llm", action="store_true", help="Alias for --enable-llm; use deterministic IR then ask the local LLM for the structured solve plan.")
    parser.add_argument(
        "--planning-mode",
        choices=["llm_required", "hybrid", "deterministic"],
        help="Plan authority mode. Defaults to llm_required when --enable-llm/--direct-llm is set, otherwise deterministic.",
    )
    parser.add_argument("--llm-status", action="store_true", help="Print local model readiness before running.")
    parser.add_argument("--llm-probe", action="store_true", help="Run a short local model JSON probe before the question.")
    parser.add_argument("--show-llm-raw", action="store_true", help="Print raw local LLM text, parsed JSON, and accepted plan details.")
    parser.add_argument("--hide-llm-raw", action="store_true", help="Do not auto-print raw LLM output when --enable-llm/--direct-llm is used.")
    parser.add_argument("--adapter-dir", help="Override XAI_LLM_ADAPTER_DIR.")
    parser.add_argument("--base-model-dir", help="Override XAI_LLM_BASE_MODEL_DIR.")
    parser.add_argument("--llm-device", help="Optional device override, for example cpu, mps, or cuda.")
    parser.add_argument("--llm-device-map", help="Optional Transformers device_map override, for example auto.")
    parser.add_argument("--llm-dtype", help="Optional torch_dtype override, for example auto, float16, bfloat16, or float32.")
    parser.add_argument("--apple-mps", action="store_true", help="Use Apple Silicon optimized local LLM settings: mps, float16, no device_map, JSON prefill/early-stop.")
    parser.add_argument("--no-apple-mps", action="store_true", help="Disable automatic Apple Silicon MPS defaults.")
    parser.add_argument("--llm-generate-max-time", type=float, default=0.0, help="Maximum seconds for each local model.generate call; 0 waits until generation stops.")
    parser.add_argument("--llm-hard-timeout", type=float, default=0.0, help="Hard wall-clock timeout for each local LLM call; 0 disables the child-process timeout and loads inline.")
    parser.add_argument("--semantic-audit", action="store_true", help="Also run the optional semantic-audit LLM call before solve planning.")
    parser.add_argument("--enable-front-repair", action="store_true", help="Opt in to one selective LLM front-repair/refinement call before or after solving.")
    parser.add_argument("--disable-front-repair", action="store_true", help="Compatibility flag; keeps LLM front-repair/refinement disabled.")
    parser.add_argument("--enable-plan-repair", action="store_true", help="Opt in to one strict LLM solve-plan repair call if the first LLM plan is invalid.")
    parser.add_argument("--max-new-tokens", type=int, default=224, help="Local LLM max_new_tokens for solve-plan/probe generation.")
    parser.add_argument("--require-llm-used", action="store_true", help="Exit with error if the local LLM generation path was not invoked.")
    parser.add_argument("--require-llm-applied", action="store_true", help="Exit with error if an LLM-generated solve plan was not accepted by the compiler.")
    args = parser.parse_args()

    question = " ".join(args.question).strip() or QUESTION
    pipeline_llm_requested = _pipeline_llm_requested(args)
    runtime_llm_requested = pipeline_llm_requested or args.llm_probe or args.llm_status

    if args.adapter_dir:
        os.environ["XAI_LLM_ADAPTER_DIR"] = str(Path(args.adapter_dir).expanduser())
    if args.base_model_dir:
        os.environ["XAI_LLM_BASE_MODEL_DIR"] = str(Path(args.base_model_dir).expanduser())
    if args.apple_mps or _should_auto_enable_apple_mps(args, runtime_llm_requested):
        _enable_apple_mps_env()
    if args.llm_device:
        os.environ["XAI_LLM_DEVICE"] = args.llm_device
    if args.llm_device_map:
        os.environ["XAI_LLM_DEVICE_MAP"] = args.llm_device_map
    if args.llm_dtype:
        os.environ["XAI_LLM_TORCH_DTYPE"] = args.llm_dtype
    os.environ["XAI_LLM_GENERATE_MAX_TIME_SECONDS"] = str(args.llm_generate_max_time)
    os.environ["XAI_LLM_HARD_TIMEOUT_SECONDS"] = str(args.llm_hard_timeout)
    os.environ["XAI_LLM_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    if args.semantic_audit:
        os.environ["XAI_LLM_ENABLE_SEMANTIC_AUDIT"] = "1"
    if args.enable_front_repair:
        os.environ["XAI_LLM_ENABLE_FRONT_REPAIR"] = "1"
    if args.disable_front_repair:
        os.environ["XAI_LLM_ENABLE_FRONT_REPAIR"] = "0"
    if args.enable_plan_repair:
        os.environ["XAI_LLM_ENABLE_PLAN_REPAIR"] = "1"
    if pipeline_llm_requested:
        os.environ["XAI_ENABLE_LOCAL_LLM"] = "1"
    planning_mode = args.planning_mode or ("llm_required" if pipeline_llm_requested else "deterministic")
    os.environ["XAI_PLANNING_MODE"] = planning_mode
    os.environ.pop("XAI_TELEMETRY_PATH", None)

    enable_llm = pipeline_llm_requested
    show_llm_raw = args.show_llm_raw or (enable_llm and not args.hide_llm_raw)
    if args.llm_status or enable_llm or args.llm_probe:
        readiness = check_local_llm_readiness()
        print("LOCAL LLM READINESS:")
        print(json.dumps(readiness, ensure_ascii=False, indent=2))
        print()

    if args.llm_probe:
        print("LOCAL LLM PROBE:")
        print(json.dumps(generate_local_llm_json(LLM_PROBE_PROMPT, max_new_tokens=args.max_new_tokens), ensure_ascii=False, indent=2))
        print()

    response = handle_request({"question": question}, enable_llm=enable_llm, timeout_seconds=args.timeout, planning_mode=planning_mode)
    _enforce_llm_requirements(response, require_used=args.require_llm_used, require_applied=args.require_llm_applied)
    if args.output_json:
        output_path = _write_response_json(response, args.output_json, full=args.output_json_full)
        print(f"SAVED_JSON: {output_path}", file=sys.stderr)
    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return

    print("QUESTION:")
    print(question)
    print()
    print("ANSWER:")
    print(response.get("answer"))
    print()
    print("CONFIDENCE:")
    print(response.get("confidence"))
    print()
    print("ROUTE:")
    route = response.get("route", {})
    print(f"- task_type: {route.get('task_type')}")
    print(f"- answer_type: {route.get('answer_type')}")
    print(f"- reasons: {route.get('reasons')}")
    print()
    print("SOLVE PLAN:")
    solve_plan = response.get("solve_plan", {})
    plan_steps = [step for step in ((solve_plan.get("plan") or {}).get("steps") or []) if isinstance(step, dict)]
    print(f"- ok: {solve_plan.get('ok')}")
    print(f"- planning_mode: {response.get('metadata', {}).get('planning_mode')}")
    print(f"- source: {(solve_plan.get('plan') or {}).get('source')}")
    print(f"- operations: {[step.get('operation') for step in plan_steps]}")
    print(f"- public_cot: {[step.get('public_cot') for step in plan_steps]}")
    print(f"- issues: {solve_plan.get('issues')}")
    print()
    print("PUBLIC COT:")
    for item in response.get("cot") or []:
        print(f"- {item}")
    print()
    print("SOLVER:")
    solver = response.get("solver", {})
    print(f"- solved: {solver.get('solved')}")
    print(f"- formula_id: {solver.get('formula_id')}")
    print(f"- trace_reason: {solver.get('trace', {}).get('reason')}")
    print()
    print("VERIFIER:")
    verifier = response.get("verifier", {})
    print(f"- ok: {verifier.get('ok')}")
    print(f"- issues: {verifier.get('issues')}")
    print()
    print("EXPLANATION:")
    print(response.get("explanation"))
    print()
    print("TRACE:")
    trace = response.get("trace", {})
    print(f"- target_unit_conversion: {trace.get('target_unit_conversion')}")
    print(f"- telemetry_store: {trace.get('telemetry_store')}")
    print()
    print("LOCAL LLM TRACE:")
    print(json.dumps(_local_llm_summary(response), ensure_ascii=False, indent=2))
    if show_llm_raw:
        _print_llm_raw_blocks(response)


def _pipeline_llm_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.enable_llm
        or args.direct_llm
        or args.semantic_audit
        or args.enable_front_repair
        or args.enable_plan_repair
        or args.require_llm_used
        or args.require_llm_applied
    )


def _should_auto_enable_apple_mps(args: argparse.Namespace, runtime_llm_requested: bool) -> bool:
    if not runtime_llm_requested or args.no_apple_mps:
        return False
    if args.llm_device or args.llm_device_map or args.llm_dtype:
        return False
    if os.environ.get("XAI_LLM_DEVICE") or os.environ.get("XAI_LLM_DEVICE_MAP") or os.environ.get("XAI_LLM_TORCH_DTYPE"):
        return False
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    return _mps_available()


def _mps_available() -> bool:
    try:
        import torch

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        return bool(mps is not None and mps.is_built() and mps.is_available())
    except Exception:
        return False


def _enable_apple_mps_env() -> None:
    os.environ["XAI_LLM_DEVICE"] = "mps"
    os.environ["XAI_LLM_TORCH_DTYPE"] = "float16"
    os.environ["XAI_LLM_DEVICE_MAP"] = "none"
    os.environ["XAI_LLM_JSON_PREFILL"] = "1"
    os.environ["XAI_LLM_JSON_EARLY_STOP"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def _enforce_llm_requirements(response: dict, *, require_used: bool, require_applied: bool) -> None:
    summary = _local_llm_summary(response)
    solve_plan_trace = (((response.get("front") or {}).get("trace") or {}).get("local_llm_solve_plan") or {})
    plan_source = (response.get("solve_plan", {}).get("plan") or {}).get("source")
    used = _any_llm_call_used(summary)
    applied = bool(summary.get("applied")) and plan_source in {"local_llm", "local_llm_repair"}
    if require_used and not used:
        raise SystemExit("LOCAL_LLM_REQUIRED_BUT_NOT_USED: " + json.dumps(summary, ensure_ascii=False))
    if require_applied and not applied:
        raise SystemExit(
            "LOCAL_LLM_REQUIRED_BUT_NOT_APPLIED: "
            + json.dumps(
                {
                    "solve_plan_trace": _generation_summary(solve_plan_trace.get("generation") or {}),
                    "llm_summary": summary,
                    "plan_source": plan_source,
                },
                ensure_ascii=False,
            )
        )


def _any_llm_call_used(summary: dict) -> bool:
    return bool(
        summary.get("used")
        or (summary.get("front_refinement") or {}).get("used")
        or (summary.get("front_repair") or {}).get("used")
        or (summary.get("repair") or {}).get("used")
    )


def _write_response_json(response: dict, output_json: str, *, full: bool = False) -> Path:
    output_path = Path(output_json).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = response if full else _compact_output_payload(response)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _compact_output_payload(response: dict) -> dict:
    """Keep saved manual-run JSON focused on audit/debug fields, not raw traces."""
    return {
        "schema": "manual_question_test.compact_result.v1",
        "question": response.get("front", {}).get("raw_question"),
        "answer": response.get("answer"),
        "confidence": response.get("confidence"),
        "status": response.get("metadata", {}).get("status"),
        "explanation": response.get("explanation"),
        "cot": response.get("cot") or [],
        "premises": response.get("premises") or [],
        "semantic_frontend": _semantic_summary(response),
        "route": _route_summary(response),
        "solve_plan": _solve_plan_summary(response),
        "constraint_graph": _constraint_graph_summary(response),
        "solver": _solver_summary(response),
        "verifier": _verifier_summary(response),
        "answer_checker": _answer_checker_summary(response),
        "proof": _proof_summary(response),
        "runtime": _runtime_summary(response),
    }


def _semantic_summary(response: dict) -> dict:
    front = response.get("front") or {}
    trace = front.get("trace") or {}
    return {
        "canonical_question": front.get("canonical_question"),
        "parse_confidence": front.get("parse_confidence"),
        "answer_type_hint": front.get("answer_type_hint"),
        "quantities": [_quantity_summary(item) for item in front.get("quantities") or []],
        "symbolic_quantities": front.get("symbolic_quantities") or [],
        "entities": [_entity_summary(item) for item in front.get("entities") or []],
        "relations": [_relation_summary(item) for item in front.get("relations") or []],
        "constraints": front.get("constraints") or [],
        "goals": [_goal_summary(item) for item in front.get("goals") or []],
        "implicit_facts": [_implicit_fact_summary(item) for item in front.get("implicit_facts") or []],
        "warnings": front.get("warnings") or [],
        "topology": _topology_summary(front.get("topology_graph") or {}),
        "canonical_structures": _canonical_structure_summary(front.get("canonical_structures") or {}),
        "stage_summary": {
            "semantic_parser": _semantic_parser_stage_summary(trace.get("semantic_parser") or {}),
            "logic_engine": _logic_engine_stage_summary(trace.get("logic_engine") or {}),
        },
    }


def _quantity_summary(item: dict) -> dict:
    return {
        "raw_text": item.get("raw_text"),
        "symbol": item.get("symbol"),
        "value": item.get("value"),
        "unit": item.get("unit"),
        "dimension": item.get("dimension"),
        "entity_id": item.get("entity_id"),
        "state_id": item.get("state_id"),
        "role": item.get("role"),
        "confidence": item.get("confidence"),
    }


def _entity_summary(item: dict) -> dict:
    return {
        "entity_id": item.get("entity_id"),
        "label": item.get("label"),
        "entity_type": item.get("entity_type"),
        "symbol": item.get("symbol"),
        "confidence": item.get("confidence"),
    }


def _relation_summary(item: dict) -> dict:
    return {
        "relation_type": item.get("relation_type"),
        "subject": item.get("subject"),
        "object": item.get("object"),
        "qualifier": item.get("qualifier"),
        "evidence": item.get("evidence"),
        "confidence": item.get("confidence"),
    }


def _goal_summary(item: dict) -> dict:
    return {
        "goal_id": item.get("goal_id"),
        "text": item.get("text"),
        "dimension": item.get("dimension"),
        "symbol": item.get("symbol"),
        "confidence": item.get("confidence"),
    }


def _implicit_fact_summary(item: dict) -> dict:
    return {
        "rule_id": item.get("rule_id"),
        "adds": item.get("adds"),
        "premise": item.get("premise"),
        "trigger_text": item.get("trigger_text"),
        "confidence": item.get("confidence"),
    }


def _topology_summary(topology: dict) -> dict:
    return {
        "canonical_form": topology.get("canonical_form"),
        "node_count": len(topology.get("nodes") or []),
        "edge_count": len(topology.get("edges") or []),
        "is_complex": topology.get("is_complex"),
        "ambiguity": topology.get("ambiguity") or [],
    }


def _canonical_structure_summary(structures: dict) -> dict:
    geometry = structures.get("geometry") or {}
    component_groups = structures.get("component_groups") or {}
    return {
        "geometry": geometry,
        "component_counts": {key: len(value or []) for key, value in component_groups.items()},
    }


def _semantic_parser_stage_summary(stage: dict) -> dict:
    return {
        "quantity_count": stage.get("quantity_count"),
        "symbolic_quantity_count": stage.get("symbolic_quantity_count"),
        "entity_count": stage.get("entity_count"),
        "relation_count": stage.get("relation_count"),
        "constraint_count": stage.get("constraint_count"),
        "goal_count": stage.get("goal_count"),
        "topology_canonical_form": stage.get("topology_canonical_form"),
        "warnings": stage.get("warnings") or [],
        "llm_used": stage.get("llm_used"),
    }


def _logic_engine_stage_summary(stage: dict) -> dict:
    return {
        "rules_applied": stage.get("rules_applied") or [],
        "derived_fact_ids": stage.get("derived_fact_ids") or [],
        "forward_chaining": stage.get("forward_chaining"),
        "llm_used": stage.get("llm_used"),
    }


def _route_summary(response: dict) -> dict:
    route = response.get("route") or {}
    return {
        "task_type": route.get("task_type"),
        "answer_type": route.get("answer_type"),
        "confidence": route.get("confidence"),
        "reasons": route.get("reasons") or [],
    }


def _solve_plan_summary(response: dict) -> dict:
    solve_plan = response.get("solve_plan") or {}
    plan = solve_plan.get("plan") or {}
    steps = [step for step in plan.get("steps") or [] if isinstance(step, dict)]
    return {
        "ok": solve_plan.get("ok"),
        "status": plan.get("status"),
        "source": plan.get("source"),
        "planning_mode": response.get("metadata", {}).get("planning_mode"),
        "planning_authority": (solve_plan.get("trace") or {}).get("planning_authority"),
        "task_type": plan.get("task_type"),
        "answer_type": plan.get("answer_type"),
        "selected_formula_ids": solve_plan.get("selected_formula_ids") or [],
        "preferred_engine_order": solve_plan.get("preferred_engine_order") or [],
        "issues": solve_plan.get("issues") or [],
        "steps": [_plan_step_summary(step) for step in steps],
        "output_format": plan.get("output_format"),
        "llm": _local_llm_summary(response),
        "validation": (solve_plan.get("trace") or {}).get("validation"),
    }


def _plan_step_summary(step: dict) -> dict:
    return {
        "step_id": step.get("step_id"),
        "operation": step.get("operation"),
        "formula_id": step.get("formula_id"),
        "principle_id": step.get("principle_id"),
        "geometry_constructor_id": step.get("geometry_constructor_id"),
        "depends_on": step.get("depends_on") or [],
        "output": step.get("output"),
        "public_cot": step.get("public_cot"),
    }


def _local_llm_summary(response: dict) -> dict:
    front_trace = response.get("front", {}).get("trace", {}) or {}
    refinement_trace = front_trace.get("local_llm_refinement") or {}
    solve_plan_trace = front_trace.get("local_llm_solve_plan") or {}
    repair_trace = front_trace.get("local_llm_solve_plan_repair") or {}
    solver_repair_trace = (response.get("solver", {}).get("trace") or {}).get("local_llm_repair") or {}
    compiled_llm_trace = ((response.get("solve_plan") or {}).get("trace") or {}).get("llm_plan_trace") or {}
    generation = solve_plan_trace.get("generation") or {}
    repair_generation = repair_trace.get("generation") or {}
    readiness = solve_plan_trace.get("readiness") or generation.get("readiness") or refinement_trace.get("readiness") or {}
    runtime_config = readiness.get("runtime_config") or {}
    return {
        "used": bool(solve_plan_trace.get("used")),
        "applied": bool(compiled_llm_trace.get("applied")),
        "source": (response.get("solve_plan", {}).get("plan") or {}).get("source"),
        "reason": solve_plan_trace.get("reason"),
        "compiler_reason": compiled_llm_trace.get("reason"),
        "front_refinement": {
            "used": bool(refinement_trace.get("used")),
            "applied": bool(refinement_trace.get("applied")),
            "reason": refinement_trace.get("reason"),
            "patch_validation": refinement_trace.get("patch_validation"),
            "generation": _generation_summary(refinement_trace.get("generation") or {}),
        },
        "front_repair": {
            "used": bool(solver_repair_trace.get("used")),
            "applied": bool(solver_repair_trace.get("applied")),
            "reason": solver_repair_trace.get("reason"),
            "patch_validation": solver_repair_trace.get("patch_validation"),
            "generation": _generation_summary(solver_repair_trace.get("generation") or {}),
        },
        "generation": _generation_summary(generation),
        "repair": {
            "used": bool(repair_trace.get("used")),
            "applied": bool(repair_trace.get("applied")),
            "reason": repair_trace.get("reason"),
            "generation": _generation_summary(repair_generation),
        },
        "runtime_config": runtime_config,
        "readiness_issues": readiness.get("issues") or [],
    }


def _generation_summary(generation: dict) -> dict:
    if not generation:
        return {}
    parsed = generation.get("json")
    return {
        "ok": generation.get("ok"),
        "used": generation.get("used"),
        "reason": generation.get("reason"),
        "schema": generation.get("schema"),
        "elapsed_seconds": generation.get("elapsed_seconds"),
        "runtime_load_seconds": generation.get("runtime_load_seconds"),
        "total_elapsed_seconds": generation.get("total_elapsed_seconds"),
        "hard_timeout_seconds": generation.get("hard_timeout_seconds"),
        "max_new_tokens": generation.get("max_new_tokens"),
        "prompt_chars": generation.get("prompt_chars"),
        "generated_tokens": generation.get("generated_tokens"),
        "parsed_json": _parsed_json_summary(parsed),
        "has_json": isinstance(parsed, dict),
    }


def _parsed_json_summary(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    summary = {"top_level_keys": sorted(str(key) for key in payload)}
    for key in ("status", "plan_card_id", "plan_template_id", "formula_id", "principle_id", "geometry_template_id"):
        if key in payload:
            summary[key] = payload.get(key)
    if isinstance(payload.get("public_cot"), list):
        summary["public_cot"] = [str(item) for item in payload["public_cot"][:4]]
    if isinstance(payload.get("steps"), list):
        summary["step_count"] = len(payload["steps"])
        summary["operations"] = [
            step.get("operation")
            for step in payload["steps"][:6]
            if isinstance(step, dict) and step.get("operation")
        ]
    if isinstance(payload.get("notes"), list):
        summary["note_count"] = len(payload["notes"])
    return summary


def _print_llm_raw_blocks(response: dict) -> None:
    front_trace = response.get("front", {}).get("trace", {}) or {}
    blocks = [
        ("LOCAL LLM SOLVE PLAN RAW", (front_trace.get("local_llm_solve_plan") or {}).get("generation") or {}),
        ("LOCAL LLM SOLVE PLAN REPAIR RAW", (front_trace.get("local_llm_solve_plan_repair") or {}).get("generation") or {}),
        ("LOCAL LLM FRONT REFINEMENT RAW", (front_trace.get("local_llm_refinement") or {}).get("generation") or {}),
        ("LOCAL LLM FRONT REPAIR RAW", (((response.get("solver") or {}).get("trace") or {}).get("local_llm_repair") or {}).get("generation") or {}),
    ]
    printed = False
    for title, generation in blocks:
        if not generation:
            continue
        printed = True
        print()
        print(title + ":")
        print(json.dumps(_llm_raw_generation_payload(generation), ensure_ascii=False, indent=2))
    print()
    print("ACCEPTED/COMPILED PLAN:")
    solve_plan = response.get("solve_plan") or {}
    print(json.dumps(
        {
            "ok": solve_plan.get("ok"),
            "issues": solve_plan.get("issues") or [],
            "plan": solve_plan.get("plan") or {},
            "trace": {
                "planning_mode": (solve_plan.get("trace") or {}).get("planning_mode"),
                "planning_authority": (solve_plan.get("trace") or {}).get("planning_authority"),
                "llm_plan_trace": (solve_plan.get("trace") or {}).get("llm_plan_trace"),
            },
        },
        ensure_ascii=False,
        indent=2,
    ))
    if not printed:
        print()
        print("LOCAL LLM RAW GENERATION:")
        print("No local LLM generation block was recorded for this run.")


def _llm_raw_generation_payload(generation: dict) -> dict:
    return {
        "ok": generation.get("ok"),
        "used": generation.get("used"),
        "reason": generation.get("reason"),
        "schema": generation.get("schema"),
        "elapsed_seconds": generation.get("elapsed_seconds"),
        "runtime_load_seconds": generation.get("runtime_load_seconds"),
        "total_elapsed_seconds": generation.get("total_elapsed_seconds"),
        "hard_timeout_seconds": generation.get("hard_timeout_seconds"),
        "max_time_seconds": generation.get("max_time_seconds"),
        "max_new_tokens": generation.get("max_new_tokens"),
        "prompt_chars": generation.get("prompt_chars"),
        "generated_tokens": generation.get("generated_tokens"),
        "json_early_stop": generation.get("json_early_stop"),
        "raw_text": generation.get("raw_text"),
        "parsed_json": generation.get("json"),
    }


def _constraint_graph_summary(response: dict) -> dict:
    graph_result = response.get("constraint_graph") or {}
    graph_trace = graph_result.get("trace") or {}
    graph = graph_trace.get("graph") or {}
    return {
        "ok": graph_result.get("ok"),
        "formula_ids": graph_result.get("formula_ids") or [],
        "issues": graph_result.get("issues") or [],
        "available_dimensions": graph_trace.get("available_dimensions") or [],
        "target_dimensions": graph.get("target_dimensions") or [],
        "reachable_formula_ids": graph.get("reachable_formula_ids") or [],
        "selected_formula_ids": graph.get("selected_formula_ids") or [],
        "solvable_target_dimensions": graph.get("solvable_target_dimensions") or [],
        "derived_variables": graph.get("derived_variables") or [],
    }


def _solver_summary(response: dict) -> dict:
    solver = response.get("solver") or {}
    trace = solver.get("trace") or {}
    return {
        "solved": solver.get("solved"),
        "answer": solver.get("answer"),
        "value": solver.get("value"),
        "unit": solver.get("unit"),
        "formula_id": solver.get("formula_id"),
        "principle_id": solver.get("principle_id"),
        "confidence": solver.get("confidence"),
        "stage": trace.get("stage"),
        "expression": trace.get("expression"),
        "target_dimension": trace.get("target_dimension"),
        "reason": trace.get("reason"),
        "inputs": _input_binding_summary(trace.get("inputs") or {}),
        "constants": trace.get("constants") or {},
        "binding_audit": trace.get("binding_audit") or {},
        "geometry": _geometry_execution_summary(trace),
        "components": _pick_existing(trace, ["components", "force_components", "vector_components", "contributions"]),
        "attempted_formula_ids": trace.get("attempted_formula_ids") or [],
    }


def _input_binding_summary(inputs: dict) -> dict:
    output = {}
    for key, value in inputs.items():
        if not isinstance(value, dict):
            output[key] = value
            continue
        output[key] = {
            "dimension": value.get("dimension"),
            "raw_text": value.get("raw_text"),
            "unit": value.get("unit"),
            "si_value": value.get("si_value"),
            "symbol": value.get("symbol"),
            "entity_id": value.get("entity_id"),
            "binding_policy": value.get("binding_policy"),
            "candidate_count": value.get("candidate_count"),
        }
    return output


def _geometry_execution_summary(trace: dict) -> Any:
    geometry = trace.get("geometry")
    if geometry is None:
        geometry = trace.get("geometry_audit")
    if geometry is None:
        geometry = trace.get("geometry_template")
    return geometry


def _pick_existing(source: dict, keys: list[str]) -> Any:
    for key in keys:
        if key in source:
            return source.get(key)
    return None


def _verifier_summary(response: dict) -> dict:
    verifier = response.get("verifier") or {}
    audit = verifier.get("audit") or {}
    return {
        "ok": verifier.get("ok"),
        "confidence": verifier.get("confidence"),
        "issues": verifier.get("issues") or [],
        "conflicts": verifier.get("conflicts") or [],
        "residual": audit.get("residual"),
        "physical_domain": audit.get("physical_domain"),
        "multi_path": audit.get("multi_path"),
        "uncertainty": audit.get("uncertainty"),
        "structured_solve_plan": audit.get("structured_solve_plan"),
    }


def _answer_checker_summary(response: dict) -> dict:
    checker = response.get("answer_checker") or {}
    return {
        "ok": checker.get("ok"),
        "issues": checker.get("issues") or [],
        "mode": (checker.get("trace") or {}).get("mode"),
    }


def _proof_summary(response: dict) -> dict:
    proof = (response.get("trace") or {}).get("proof_dag") or (response.get("solver", {}).get("trace") or {}).get("proof_dag") or {}
    return {
        "status": proof.get("status"),
        "selected_formula_ids": proof.get("selected_formula_ids") or [],
        "audit": proof.get("audit"),
        "certificate": proof.get("certificate"),
        "node_count": len(proof.get("nodes") or []),
        "edge_count": len(proof.get("edges") or []),
    }


def _runtime_summary(response: dict) -> dict:
    trace = response.get("trace") or {}
    metadata = response.get("metadata") or {}
    return {
        "cache": response.get("cache") or {"hit": (trace.get("cache") or {}).get("hit")},
        "deadline": trace.get("deadline"),
        "telemetry_store": trace.get("telemetry_store"),
        "versions": metadata.get("versions") or {},
        "xai_policy": metadata.get("xai_policy") or {},
    }


if __name__ == "__main__":
    main()
