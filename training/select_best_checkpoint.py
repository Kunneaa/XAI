import argparse
import json
import subprocess
from pathlib import Path


def list_checkpoints(out_dir: Path):
    cps = []
    for p in out_dir.glob("checkpoint-*"):
        if p.is_dir():
            cps.append(p)
    cps.sort(key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else -1)
    return cps


def eval_model(model_path: Path, max_logic_records: int, max_physics_samples: int):
    cmd = [
        "python3",
        "training/competition_eval.py",
        "--model-path",
        str(model_path),
        "--max-logic-records",
        str(max_logic_records),
        "--max-physics-samples",
        str(max_physics_samples),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None, p.stderr
    try:
        report = json.loads(p.stdout)
        return report, ""
    except Exception:
        return None, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-logic-records", type=int, default=50)
    ap.add_argument("--max-physics-samples", type=int, default=200)
    ap.add_argument("--report-out", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    candidates = [out_dir] + list_checkpoints(out_dir)
    results = []

    for c in candidates:
        report, err = eval_model(c, args.max_logic_records, args.max_physics_samples)
        if report is not None:
            results.append(report)
        else:
            print(f"[warn] failed evaluating {c}: {err}")

    if not results:
        raise SystemExit("No checkpoint evaluated successfully.")

    best = max(results, key=lambda x: x.get("composite_score", 0.0))
    final = {"best": best, "all": results}
    print(json.dumps(final, ensure_ascii=False, indent=2))
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

