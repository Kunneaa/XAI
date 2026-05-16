import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from training.benchmark_core import run_logic_eval, run_physics_eval
from src.xai_pipeline.pipeline import XAIPipeline


def compute_composite(logic_stats, physics_stats):
    p1_logic = logic_stats.get("acc", 0.0)
    p1_phys = physics_stats.get("strict_rate", 0.0)
    p2 = 0.5 * logic_stats.get("explanation_nonempty_rate", 0.0) + 0.5 * physics_stats.get(
        "explanation_formula_like_rate", 0.0
    )
    p3 = 0.5 * logic_stats.get("explanation_nonempty_rate", 0.0) + 0.5 * physics_stats.get(
        "unit_match", 0.0
    ) / max(physics_stats.get("total", 1), 1)
    # Competition-style weighted blend proxy: P1 dominates.
    return 0.6 * (0.5 * p1_logic + 0.5 * p1_phys) + 0.25 * p2 + 0.15 * p3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--max-logic-records", type=int, default=50)
    ap.add_argument("--max-physics-samples", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ["XAI_MODEL_PATH"] = args.model_path
    os.environ.setdefault("XAI_USE_LLM", "1")
    os.environ.setdefault("XAI_DATA_SPLIT", "test")

    pipe = XAIPipeline(ROOT)
    logic_stats = run_logic_eval(pipe, ROOT, max_records=args.max_logic_records)
    physics_stats = run_physics_eval(pipe, ROOT, max_samples=args.max_physics_samples)
    composite = compute_composite(logic_stats, physics_stats)

    report = {
        "model_path": args.model_path,
        "logic": logic_stats,
        "physics": physics_stats,
        "composite_score": composite,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

