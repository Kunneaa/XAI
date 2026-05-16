import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.xai_pipeline.pipeline import XAIPipeline


def eval_logic(pipe: XAIPipeline, rows, max_records=30):
    total, correct = 0, 0
    for row in rows[:max_records]:
        premises = [str(x) for x in row.get("premises-NL", [])]
        for q, a in zip(row.get("questions", []), row.get("answers", [])):
            total += 1
            pred = pipe.predict(str(q), premises)
            if str(pred["answer"]).strip().lower() == str(a).strip().lower():
                correct += 1
    return correct / total if total else 0.0


def run(version: str):
    os.environ["XAI_PROMPT_VERSION"] = version
    pipe = XAIPipeline(ROOT)
    rows = json.loads((ROOT / "Logic_Based_Educational_Queries.test.json").read_text(encoding="utf-8"))
    acc = eval_logic(pipe, rows)
    return {"prompt_version": version, "logic_acc": acc}


if __name__ == "__main__":
    for v in ["v1", "v2"]:
        print(run(v))
