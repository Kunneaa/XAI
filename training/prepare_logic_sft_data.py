import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "Logic_Based_Educational_Queries.train.json"
OUT_FOL = ROOT / "training" / "logic_nl2fol_sft.jsonl"
OUT_EXPL = ROOT / "training" / "logic_explainer_sft.jsonl"


def main() -> None:
    rows = json.loads(IN_PATH.read_text(encoding="utf-8"))
    n_fol = 0
    n_expl = 0

    with OUT_FOL.open("w", encoding="utf-8") as wf, OUT_EXPL.open("w", encoding="utf-8") as we:
        for r in rows:
            premises_nl = [str(x) for x in r.get("premises-NL", [])]
            premises_fol = [str(x) for x in r.get("premises-FOL", [])]
            questions = [str(x) for x in r.get("questions", [])]
            answers = [str(x) for x in r.get("answers", [])]
            exps = [str(x) for x in r.get("explanation", [])]

            if premises_nl and premises_fol:
                fol_item = {
                    "instruction": "Convert natural-language premises into compact formal logic statements.",
                    "input": json.dumps({"premises_nl": premises_nl}, ensure_ascii=False),
                    "output": json.dumps({"fol": premises_fol}, ensure_ascii=False),
                }
                wf.write(json.dumps(fol_item, ensure_ascii=False) + "\n")
                n_fol += 1

            for q, a, e in zip(questions, answers, exps):
                expl_item = {
                    "instruction": "Answer the logic question and provide concise verifiable explanation.",
                    "input": json.dumps({"premises_nl": premises_nl, "question": q}, ensure_ascii=False),
                    "output": json.dumps({"answer": a, "explanation": e}, ensure_ascii=False),
                }
                we.write(json.dumps(expl_item, ensure_ascii=False) + "\n")
                n_expl += 1

    print("wrote", OUT_FOL, n_fol)
    print("wrote", OUT_EXPL, n_expl)


if __name__ == "__main__":
    main()
