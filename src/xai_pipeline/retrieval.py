"""Safe metadata-only retrieval helper."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from .front_pipeline import process_question_front
from .registries import FORMULA_REGISTRY
from .router import route


TOKEN_RE = re.compile(r"[a-zA-Z0-9_μΩ]+")


@dataclass(frozen=True)
class RetrievalHit:
    problem_id: str
    score: float
    task_metadata: dict

    def to_dict(self):
        return {"problem_id": self.problem_id, "score": self.score, "task_metadata": dict(self.task_metadata)}


def retrieve_metadata(question: str, data_path: Path, k: int = 5) -> List[RetrievalHit]:
    query_tokens = _tokens(question)
    hits: List[RetrievalHit] = []
    for problem_id, doc_tokens, metadata in _load_retrieval_index(str(data_path)):
        score = _cosine(query_tokens, Counter(dict(doc_tokens)))
        if score > 0:
            hits.append(RetrievalHit(problem_id, score, metadata))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:k]


@lru_cache(maxsize=4)
def _load_retrieval_index(data_path: str) -> Tuple[Tuple[str, Tuple[Tuple[str, int], ...], dict], ...]:
    path = Path(data_path)
    if not path.exists():
        return tuple()
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            front = process_question_front(row["question"])
            route_result = route(front)
            formula_ids = [
                formula_id
                for formula_id, spec in FORMULA_REGISTRY.items()
                if spec.task_type == route_result.task_type
            ]
            metadata = {
                "concepts": front["concepts"],
                "answer_type_hint": front["answer_type_hint"],
                "task_type": route_result.task_type,
                "target_hints": front["target_hints"],
                "quantity_dimensions": [q["dimension"] for q in front["quantities"]],
                "formula_ids": formula_ids,
                "principle_ids": sorted({FORMULA_REGISTRY[formula_id].principle_id for formula_id in formula_ids}),
                "safe_fields_only": True,
            }
            rows.append((row["id"], tuple(_tokens(row["question"]).items()), metadata))
    return tuple(rows)


def _tokens(text: str) -> Counter:
    return Counter(token.lower() for token in TOKEN_RE.findall(text.replace("µ", "μ")))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[token] * b.get(token, 0) for token in a)
    an = math.sqrt(sum(v * v for v in a.values()))
    bn = math.sqrt(sum(v * v for v in b.values()))
    return dot / max(an * bn, 1e-12)
