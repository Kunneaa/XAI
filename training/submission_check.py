import argparse
import json
from pathlib import Path


REQUIRED = {"answer": str, "explanation": str}
OPTIONAL = {"fol": (str, type(None)), "cot": list, "premises": list, "confidence": (float, int)}


def validate_item(obj):
    errors = []
    for k, t in REQUIRED.items():
        if k not in obj:
            errors.append(f"missing required field: {k}")
            continue
        if not isinstance(obj[k], t):
            errors.append(f"field `{k}` must be {t.__name__}")
    for k, t in OPTIONAL.items():
        if k in obj and not isinstance(obj[k], t):
            errors.append(f"field `{k}` has wrong type")
    if "confidence" in obj:
        c = float(obj["confidence"])
        if c < 0.0 or c > 1.0:
            errors.append("confidence must be in [0,1]")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="JSON or JSONL file containing API outputs")
    args = ap.parse_args()

    p = Path(args.file)
    text = p.read_text(encoding="utf-8").strip()
    items = []
    if p.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                items.append(json.loads(line))
    else:
        obj = json.loads(text)
        if isinstance(obj, list):
            items = obj
        else:
            items = [obj]

    total = len(items)
    bad = 0
    for i, it in enumerate(items):
        errs = validate_item(it)
        if errs:
            bad += 1
            print(f"[invalid] item {i}: " + "; ".join(errs))
    print(f"checked={total} invalid={bad} valid={total-bad}")
    if bad > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

