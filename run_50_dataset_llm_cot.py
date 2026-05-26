"""Batch runner for Physics_Problems_Text_Only.csv.

The runner keeps dataset evaluation explicit and replayable:

- input columns are preserved;
- `system_answer` is produced by the verified pipeline;
- `llm_suggested_cot` contains only compiler-accepted public CoT labels from a
  local-LLM plan. In deterministic mode it is `[]` by design.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from xai_pipeline.core.api import handle_request


DEFAULT_INPUT = Path("Physics_Problems_Text_Only.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CSV slice through NSP-Core and save system answers.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path.")
    parser.add_argument("--output", help="Output CSV path. Defaults to <input_stem>_<row_count>_llm.csv.")
    parser.add_argument("--offset", type=int, default=0, help="Zero-based row offset.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum rows to run; <=0 means all rows after offset.")
    parser.add_argument("--request-timeout", type=float, default=-1.0, help="Per-question timeout; negative disables the deadline.")
    parser.add_argument("--enable-llm", action="store_true", help="Enable local LLM planning.")
    parser.add_argument("--no-llm", action="store_true", help="Force deterministic-only planning.")
    parser.add_argument(
        "--planning-mode",
        choices=["deterministic", "hybrid", "llm_required"],
        help="Planning authority. Defaults to llm_required with --enable-llm, otherwise deterministic.",
    )
    parser.add_argument("--apple-mps", action="store_true", help="Use Apple Silicon local-LLM defaults.")
    parser.add_argument("--adapter-dir", help="Override XAI_LLM_ADAPTER_DIR.")
    parser.add_argument("--base-model-dir", help="Override XAI_LLM_BASE_MODEL_DIR.")
    parser.add_argument("--llm-device", help="Optional local LLM device override, e.g. mps/cpu/cuda.")
    parser.add_argument("--llm-device-map", help="Optional Transformers device_map override.")
    parser.add_argument("--llm-dtype", help="Optional torch dtype override, e.g. float16/float32/auto.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Local LLM solve-plan max_new_tokens.")
    parser.add_argument("--llm-generate-max-time", type=float, default=60.0, help="Transformers generate max_time; 0 disables.")
    parser.add_argument("--llm-hard-timeout", type=float, default=0.0, help="Hard LLM timeout; 0 disables child timeout.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    rows = _load_rows(input_path)
    selected_rows = _slice_rows(rows, offset=args.offset, limit=args.limit)

    output_path = _output_path(args.output, input_path, len(selected_rows))
    _configure_runtime(args)
    enable_llm = bool(args.enable_llm and not args.no_llm)
    planning_mode = args.planning_mode or ("llm_required" if enable_llm else "deterministic")

    print(f"RUN_ROWS={len(selected_rows)} OUTPUT={output_path}")
    summary = {"rows": len(selected_rows), "ok": 0, "uncertain": 0, "errors": 0, "llm_used": 0, "llm_applied": 0}
    fieldnames = _output_fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_index, row in enumerate(selected_rows, start=1):
            result, error = _run_row(row, enable_llm=enable_llm, planning_mode=planning_mode, timeout=args.request_timeout)
            system_answer = result.get("answer") if result else "Uncertain"
            llm_cot = _llm_suggested_cot(result) if result else []
            output_row = {key: row.get(key, "") for key in fieldnames}
            output_row["system_answer"] = system_answer
            output_row["llm_suggested_cot"] = json.dumps(llm_cot, ensure_ascii=False)
            writer.writerow(output_row)

            event = _row_event(run_index, row, result, error, planning_mode)
            print(json.dumps(event, ensure_ascii=False))
            if error:
                summary["errors"] += 1
            elif system_answer == "Uncertain":
                summary["uncertain"] += 1
            else:
                summary["ok"] += 1
            if event.get("llm_used"):
                summary["llm_used"] += 1
            if event.get("llm_applied"):
                summary["llm_applied"] += 1

    print(f"SUMMARY={json.dumps(summary, ensure_ascii=False)}")
    print(f"DONE={output_path}")


def _load_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _slice_rows(rows: list[dict[str, str]], *, offset: int, limit: int) -> list[dict[str, str]]:
    start = max(0, offset)
    if limit <= 0:
        return rows[start:]
    return rows[start : start + limit]


def _output_path(raw_output: str | None, input_path: Path, row_count: int) -> Path:
    if raw_output:
        output_path = Path(raw_output).expanduser()
        return output_path if output_path.is_absolute() else Path.cwd() / output_path
    return input_path.with_name(f"{input_path.stem}_{row_count}_llm.csv")


def _output_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    base = list(rows[0].keys()) if rows else ["id", "question", "cot", "answer", "unit"]
    for field in ("system_answer", "llm_suggested_cot"):
        if field not in base:
            base.append(field)
    return base


def _configure_runtime(args: argparse.Namespace) -> None:
    os.environ.pop("XAI_TELEMETRY_PATH", None)
    if args.apple_mps:
        os.environ["XAI_LLM_DEVICE"] = "mps"
        os.environ["XAI_LLM_TORCH_DTYPE"] = "float16"
        os.environ["XAI_LLM_DEVICE_MAP"] = "none"
        os.environ["XAI_LLM_JSON_PREFILL"] = "1"
        os.environ["XAI_LLM_JSON_EARLY_STOP"] = "1"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    for attr, env_name in (
        ("adapter_dir", "XAI_LLM_ADAPTER_DIR"),
        ("base_model_dir", "XAI_LLM_BASE_MODEL_DIR"),
        ("llm_device", "XAI_LLM_DEVICE"),
        ("llm_device_map", "XAI_LLM_DEVICE_MAP"),
        ("llm_dtype", "XAI_LLM_TORCH_DTYPE"),
    ):
        value = getattr(args, attr)
        if value:
            os.environ[env_name] = str(Path(value).expanduser()) if attr.endswith("_dir") else str(value)
    os.environ["XAI_LLM_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["XAI_LLM_GENERATE_MAX_TIME_SECONDS"] = str(args.llm_generate_max_time)
    os.environ["XAI_LLM_HARD_TIMEOUT_SECONDS"] = str(args.llm_hard_timeout)
    os.environ["XAI_ENABLE_LOCAL_LLM"] = "1" if args.enable_llm and not args.no_llm else "0"


def _run_row(row: dict[str, str], *, enable_llm: bool, planning_mode: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return (
            handle_request(
                {"question": row.get("question", "")},
                enable_llm=enable_llm,
                timeout_seconds=timeout,
                planning_mode=planning_mode,
            ),
            None,
        )
    except Exception as exc:  # pragma: no cover - kept for long batch resilience.
        return None, repr(exc)


def _llm_suggested_cot(response: dict[str, Any] | None) -> list[str]:
    if not response:
        return []
    plan = ((response.get("solve_plan") or {}).get("plan") or {})
    if plan.get("source") not in {"local_llm", "local_llm_repair"}:
        return []
    return [
        str(step.get("public_cot"))
        for step in plan.get("steps") or []
        if isinstance(step, dict) and step.get("public_cot")
    ]


def _row_event(
    run_index: int,
    row: dict[str, str],
    response: dict[str, Any] | None,
    error: str | None,
    planning_mode: str,
) -> dict[str, Any]:
    if response is None:
        return {
            "row": run_index,
            "id": row.get("id"),
            "answer": "Uncertain",
            "status": "error",
            "error": error,
            "planning_mode": planning_mode,
        }
    solve_plan = response.get("solve_plan") or {}
    plan = solve_plan.get("plan") or {}
    llm_trace = (((response.get("front") or {}).get("trace") or {}).get("local_llm_solve_plan") or {})
    generation = llm_trace.get("generation") or {}
    return {
        "row": run_index,
        "id": row.get("id"),
        "answer": response.get("answer"),
        "status": response.get("metadata", {}).get("status"),
        "verifier_ok": response.get("verifier", {}).get("ok"),
        "llm_used": llm_trace.get("used"),
        "llm_applied": ((solve_plan.get("trace") or {}).get("llm_plan_trace") or {}).get("applied"),
        "llm_reason": llm_trace.get("reason"),
        "plan_card_id": (generation.get("json") or {}).get("plan_card_id"),
        "llm_tokens": generation.get("generated_tokens"),
        "llm_seconds": generation.get("total_elapsed_seconds") or generation.get("elapsed_seconds"),
        "planning_mode": response.get("metadata", {}).get("planning_mode") or planning_mode,
        "plan_source": plan.get("source"),
        "operations": [step.get("operation") for step in plan.get("steps") or [] if isinstance(step, dict)],
        "cot_items": len(_llm_suggested_cot(response)),
    }


if __name__ == "__main__":
    main()
