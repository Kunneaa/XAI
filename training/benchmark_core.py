import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.xai_pipeline.pipeline import XAIPipeline


def run_logic_eval(pipe: XAIPipeline, root: Path, max_records: int = 50):
    rows = json.loads((root / "Logic_Based_Educational_Queries.test.json").read_text(encoding="utf-8"))
    total = 0
    correct = 0
    for row in rows[:max_records]:
        premises = [str(x) for x in row.get("premises-NL", [])]
        for q, a in zip(row.get("questions", []), row.get("answers", [])):
            total += 1
            pred = pipe.predict(str(q), premises)
            if str(pred["answer"]).strip().lower() == str(a).strip().lower():
                correct += 1
    return {"total": total, "correct": correct, "acc": (correct / total if total else 0.0)}


def run_physics_eval(pipe: XAIPipeline, root: Path, max_samples: int = 200):
    import pandas as pd

    df = pd.read_csv(root / "Physics_Problems_Text_Only.test.csv").head(max_samples)
    total = len(df)
    contain_match = 0
    for _, r in df.iterrows():
        pred = pipe.predict(str(r["question"]))
        if str(r["answer"]) in str(pred["answer"]):
            contain_match += 1
    return {"total": total, "contain_match": contain_match, "rate": (contain_match / total if total else 0.0)}


def main():
    root = ROOT
    pipe = XAIPipeline(root)
    print("logic:", run_logic_eval(pipe, root))
    print("physics:", run_physics_eval(pipe, root))


if __name__ == "__main__":
    main()
