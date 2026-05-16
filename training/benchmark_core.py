import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.xai_pipeline.normalizer import extract_number, normalize_logic_answer, normalize_unit
from src.xai_pipeline.pipeline import XAIPipeline

def _to_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _unit_in_answer(ans: str, unit: str) -> bool:
    au = str(ans)
    u = normalize_unit(unit)
    if not u:
        return True
    return u.lower() in au.lower()


def run_logic_eval(pipe: XAIPipeline, root: Path, max_records: int = 50):
    rows = json.loads((root / "Logic_Based_Educational_Queries.test.json").read_text(encoding="utf-8"))
    total = 0
    correct = 0
    explanation_nonempty = 0
    for row in rows[:max_records]:
        premises = [str(x) for x in row.get("premises-NL", [])]
        for q, a in zip(row.get("questions", []), row.get("answers", [])):
            total += 1
            pred = pipe.predict(str(q), premises)
            if normalize_logic_answer(str(pred["answer"])) == normalize_logic_answer(str(a)):
                correct += 1
            if str(pred.get("explanation", "")).strip():
                explanation_nonempty += 1
    return {
        "total": total,
        "correct": correct,
        "acc": (correct / total if total else 0.0),
        "explanation_nonempty_rate": (explanation_nonempty / total if total else 0.0),
    }


def run_physics_eval(pipe: XAIPipeline, root: Path, max_samples: int = 200):
    import pandas as pd

    df = pd.read_csv(root / "Physics_Problems_Text_Only.test.csv").head(max_samples)
    total = len(df)
    exact_numeric = 0
    numeric_tol = 0
    unit_match = 0
    strict_match = 0
    explanation_nonempty = 0
    explanation_formula_like = 0
    for _, r in df.iterrows():
        pred = pipe.predict(str(r["question"]))
        pred_answer = str(pred["answer"])
        pred_expl = str(pred.get("explanation", "")).strip()
        if pred_expl:
            explanation_nonempty += 1
        if "=" in pred_expl or "apply" in pred_expl.lower() or "substitute" in pred_expl.lower():
            explanation_formula_like += 1
        gt_num = _to_float(r["answer"])
        pred_num = extract_number(pred_answer)
        if gt_num is not None and pred_num is not None:
            if pred_num == gt_num:
                exact_numeric += 1
            if math.isclose(pred_num, gt_num, rel_tol=1e-2, abs_tol=1e-6):
                numeric_tol += 1
        if _unit_in_answer(pred_answer, str(r.get("unit", ""))):
            unit_match += 1
        if (gt_num is not None and pred_num is not None and math.isclose(pred_num, gt_num, rel_tol=1e-2, abs_tol=1e-6)
                and _unit_in_answer(pred_answer, str(r.get("unit", "")))):
            strict_match += 1
    return {
        "total": total,
        "exact_numeric": exact_numeric,
        "numeric_tol_1pct": numeric_tol,
        "unit_match": unit_match,
        "strict_num_and_unit": strict_match,
        "strict_rate": (strict_match / total if total else 0.0),
        "explanation_nonempty_rate": (explanation_nonempty / total if total else 0.0),
        "explanation_formula_like_rate": (explanation_formula_like / total if total else 0.0),
    }


def main():
    root = ROOT
    pipe = XAIPipeline(root)
    print("logic:", run_logic_eval(pipe, root))
    print("physics:", run_physics_eval(pipe, root))


if __name__ == "__main__":
    main()
