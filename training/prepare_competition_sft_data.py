import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGIC_IN = ROOT / "Logic_Based_Educational_Queries.train.json"
PHYSICS_IN = ROOT / "Physics_Problems_Text_Only.train.csv"
OUT = ROOT / "training" / "competition_sft.jsonl"


def build_logic_samples():
    rows = json.loads(LOGIC_IN.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        premises = [str(x) for x in r.get("premises-NL", [])]
        answers = [str(x) for x in r.get("answers", [])]
        questions = [str(x) for x in r.get("questions", [])]
        explanations = [str(x) for x in r.get("explanation", [])]
        for i, (q, a) in enumerate(zip(questions, answers)):
            exp = explanations[i] if i < len(explanations) else ""
            output = {
                "answer": a,
                "explanation": exp,
                "premises": premises[:8],
                "cot": ["Use premises and formal entailment checks before concluding."],
                "confidence": 0.9,
            }
            out.append(
                {
                    "instruction": "Solve the logic query from natural-language premises and return strict JSON.",
                    "input": json.dumps({"premises_nl": premises, "question": q}, ensure_ascii=False),
                    "output": json.dumps(output, ensure_ascii=False),
                }
            )
    return out


def build_physics_samples():
    out = []
    with PHYSICS_IN.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            unit = str(row.get("unit", "")).strip()
            cot = str(row.get("cot", "")).strip()
            answer_with_unit = f"{a} {unit}".strip()
            output = {
                "answer": answer_with_unit,
                "explanation": "Apply the relevant electricity formula and substitute known values carefully.",
                "premises": ["Use only electric-circuits/electrostatics formulas and unit-consistent computation."],
                "cot": [x.strip() for x in cot.split("\n") if x.strip()][:8],
                "confidence": 0.9,
            }
            out.append(
                {
                    "instruction": "Solve the physics problem and return strict JSON with final numeric answer including unit.",
                    "input": json.dumps({"question": q}, ensure_ascii=False),
                    "output": json.dumps(output, ensure_ascii=False),
                }
            )
    return out


def main():
    logic = build_logic_samples()
    physics = build_physics_samples()
    merged = logic + physics
    with OUT.open("w", encoding="utf-8") as w:
        for row in merged:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(merged))
    print("logic samples:", len(logic))
    print("physics samples:", len(physics))


if __name__ == "__main__":
    main()

