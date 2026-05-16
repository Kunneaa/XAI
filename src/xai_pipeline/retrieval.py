import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOKEN_RE = re.compile(r"[a-zA-Z0-9_μµΩ]+")


def _norm(s: str) -> str:
    s = s.replace("µ", "μ").lower().strip()
    return re.sub(r"\s+", " ", s)


def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(_norm(s))


def build_idf(docs: List[List[str]]) -> dict:
    n = len(docs)
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    return {t: math.log((n - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}


def bm25_score(query_tokens: List[str], doc_tokens: List[str], idf: dict, avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for t in query_tokens:
        if t not in tf:
            continue
        denom = tf[t] + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
        score += idf.get(t, 0.0) * (tf[t] * (k1 + 1)) / denom
    return score


@dataclass
class RetrievalHit:
    text: str
    bm25: float
    semantic: float
    score: float


class HybridRetriever:
    def __init__(self, docs: List[str]) -> None:
        self.docs = docs
        self.tokens = [tokenize(x) for x in docs]
        self.idf = build_idf(self.tokens)
        self.avgdl = sum(len(x) for x in self.tokens) / max(len(self.tokens), 1)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.doc_matrix = self.vectorizer.fit_transform(docs)

        self.use_embedder = os.getenv("XAI_USE_EMBEDDER", "0") == "1"
        self.embedder = None
        self.doc_embeddings = None
        if self.use_embedder:
            self._init_embedder()

    def _init_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
            self.doc_embeddings = self.embedder.encode(self.docs, normalize_embeddings=True)
        except Exception:
            self.embedder = None
            self.doc_embeddings = None

    def _semantic_scores(self, query: str) -> List[float]:
        if self.embedder is not None and self.doc_embeddings is not None:
            q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
            return (self.doc_embeddings @ q_emb).tolist()
        qv = self.vectorizer.transform([query])
        return cosine_similarity(qv, self.doc_matrix)[0].tolist()

    def search(self, query: str, k: int = 5, w_bm25: float = 0.5, w_sem: float = 0.5) -> List[RetrievalHit]:
        q_tokens = tokenize(query)
        bm25_scores = [bm25_score(q_tokens, dt, self.idf, self.avgdl) for dt in self.tokens]
        sem_scores = self._semantic_scores(query)

        def norm(arr: List[float]) -> List[float]:
            if not arr:
                return arr
            mn, mx = min(arr), max(arr)
            if mx - mn < 1e-12:
                return [0.0 for _ in arr]
            return [(x - mn) / (mx - mn) for x in arr]

        b_norm = norm(bm25_scores)
        s_norm = norm(sem_scores)

        hits = []
        for i, text in enumerate(self.docs):
            score = w_bm25 * b_norm[i] + w_sem * s_norm[i]
            hits.append(RetrievalHit(text=text, bm25=bm25_scores[i], semantic=sem_scores[i], score=score))

        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:k]
