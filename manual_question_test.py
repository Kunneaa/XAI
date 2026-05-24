"""Manual real-question runner for the Physics XAI pipeline.

Edit QUESTION below, then run:

    PYTHONPATH=src python3 manual_question_test.py

Production-like local run with Qwen enabled when deterministic solving fails:

    PYTHONPATH=src python3 manual_question_test.py --qwen-real

Optional guarded explanation polish:

    PYTHONPATH=src python3 manual_question_test.py --qwen-real --polish

You can also pass a question directly:

    PYTHONPATH=src python3 manual_question_test.py "A resistor R = 10 Ω has U = 20 V. Find I."
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from xai_pipeline.api import handle_request
from xai_pipeline.qwen_config import DEFAULT_MODEL_DIR, check_qwen_model_readiness, resolve_qwen_runtime_config


# Put your real question here when you want to test by editing this file.
QUESTION = "Points J and L are separated by 20 cm in air. Charges q1 = -3 × 10^-6 C and q2 = 8 × 10^-6 C are placed at J and L, respectively. J test charge q3 = 2 × 10^-6 C is placed at point C such that JC = 12 cm and LC = 16 cm. Calculate the magnitude of the electric force acting on q3."

# Production-like manual defaults. The pipeline remains deterministic-first, so
# Qwen is called only if the verified deterministic path cannot solve.
ENABLE_LLM = True

# Set to True if you want retrieval metadata for unsolved questions.
USE_RETRIEVAL = True
DATA_PATH = Path("Physics_Problems_Text_Only.csv")
MODEL_DIR = DEFAULT_MODEL_DIR
TIMEOUT_SECONDS = 55.0
TELEMETRY_PATH = Path("logs/manual_telemetry.jsonl")
ENABLE_TELEMETRY = True
ENABLE_QWEN_POLISH = False

# "summary" is easier to read. Use "json" for the full pipeline response.
OUTPUT_MODE = "summary"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real question through the Physics XAI pipeline.")
    parser.add_argument("question", nargs="*", help="Optional question override.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON response.")
    parser.add_argument("--llm", dest="llm", action="store_true", default=None, help="Enable guarded Qwen planner if deterministic solver is uncertain.")
    parser.add_argument("--no-llm", dest="llm", action="store_false", help="Disable Qwen planner.")
    parser.add_argument("--retrieval", dest="retrieval", action="store_true", default=None, help="Use safe metadata retrieval if deterministic solver is uncertain.")
    parser.add_argument("--no-retrieval", dest="retrieval", action="store_false", help="Disable retrieval.")
    parser.add_argument("--qwen-real", action="store_true", help="Enable guarded Qwen planner/runtime using models/ or the configured structured endpoint.")
    parser.add_argument("--polish", dest="polish", action="store_true", default=None, help="Enable guarded Qwen explanation polish.")
    parser.add_argument("--no-polish", dest="polish", action="store_false", help="Disable guarded Qwen explanation polish.")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="Local Qwen model directory.")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="Request timeout seconds.")
    parser.add_argument("--telemetry", default=str(TELEMETRY_PATH), help="Telemetry JSONL path.")
    parser.add_argument("--no-telemetry", action="store_true", help="Disable telemetry file writes.")
    args = parser.parse_args()

    question = " ".join(args.question).strip() or QUESTION
    use_llm = ENABLE_LLM if args.llm is None else args.llm
    use_retrieval = USE_RETRIEVAL if args.retrieval is None else args.retrieval
    use_polish = ENABLE_QWEN_POLISH if args.polish is None else args.polish

    if args.qwen_real:
        use_llm = True
        os.environ["XAI_ENABLE_LOCAL_QWEN"] = "1"
    os.environ["XAI_QWEN_MODEL_DIR"] = str(Path(args.model_dir).expanduser())
    os.environ["XAI_ENABLE_QWEN_POLISH"] = "1" if use_polish else "0"
    if ENABLE_TELEMETRY and not args.no_telemetry and args.telemetry:
        os.environ["XAI_TELEMETRY_PATH"] = str(Path(args.telemetry).expanduser())
    else:
        os.environ.pop("XAI_TELEMETRY_PATH", None)

    response = handle_request(
        {"question": question},
        data_path=DATA_PATH if use_retrieval else None,
        enable_llm=use_llm,
        timeout_seconds=args.timeout,
    )

    if args.json or OUTPUT_MODE == "json":
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return

    print("QUESTION:")
    print(question)
    print()
    print("RUNTIME:")
    readiness = check_qwen_model_readiness(os.environ["XAI_QWEN_MODEL_DIR"])
    qwen_config = resolve_qwen_runtime_config(os.environ["XAI_QWEN_MODEL_DIR"])
    print(f"- deterministic_first: True")
    print(f"- retrieval: {use_retrieval}")
    print(f"- qwen_planner_enabled: {use_llm}")
    print(f"- local_qwen_enabled: {qwen_config.enabled}")
    print(f"- qwen_model_ready: {readiness.ready}")
    print(f"- qwen_model_dir: {readiness.model_path}")
    print(f"- polish_enabled: {use_polish}")
    print(f"- telemetry_path: {os.environ.get('XAI_TELEMETRY_PATH')}")
    print(f"- timeout_seconds: {args.timeout}")
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
    print("SOLVER:")
    solver = response.get("solver", {})
    print(f"- solved: {solver.get('solved')}")
    print(f"- formula_id: {solver.get('formula_id')}")
    print(f"- principle_id: {solver.get('principle_id')}")
    print(f"- trace_reason: {solver.get('trace', {}).get('reason')}")
    print(f"- executor_mode: {solver.get('trace', {}).get('executor_mode', {}).get('mode')}")
    print()
    print("VERIFIER:")
    verifier = response.get("verifier", {})
    print(f"- ok: {verifier.get('ok')}")
    print(f"- issues: {verifier.get('issues')}")
    print()
    print("EXPLANATION:")
    print(response.get("explanation"))
    print()
    print("PRODUCTION TRACE:")
    planner = response.get("planner", {})
    cache = response.get("cache", {})
    trace = response.get("trace", {})
    print(f"- cache_hit: {cache.get('hit')}")
    print(f"- planner_reason: {planner.get('reason')}")
    print(f"- planner_budget: {planner.get('budget')}")
    print(f"- polish: {response.get('polish')}")
    print(f"- telemetry_store: {trace.get('telemetry_store')}")
    print(f"- target_unit_conversion: {trace.get('target_unit_conversion')}")


if __name__ == "__main__":
    main()
