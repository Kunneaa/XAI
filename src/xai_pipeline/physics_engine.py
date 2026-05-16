import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .data import PhysicsQA
from .llm_client import LLMClient
from .prompt_registry import get_prompt_pack
from .retrieval import HybridRetriever


@dataclass
class PhysicsResult:
    answer: str
    explanation: str
    cot: List[str]
    premises: List[str]
    confidence: float


NUM_UNIT_RE = re.compile(r"([-+]?\d*\.?\d+)\s*([a-zA-ZμµΩ]+)")


def _to_base(value: float, unit: str) -> Tuple[float, str]:
    u = unit.replace("µ", "μ").lower()
    scale = {
        "mv": (1e-3, "V"),
        "kv": (1e3, "V"),
        "ma": (1e-3, "A"),
        "μa": (1e-6, "A"),
        "ka": (1e3, "A"),
        "mohm": (1e-3, "ohm"),
        "kω": (1e3, "ohm"),
        "kohm": (1e3, "ohm"),
        "μf": (1e-6, "F"),
        "nf": (1e-9, "F"),
        "pf": (1e-12, "F"),
        "mc": (1e-3, "C"),
        "μc": (1e-6, "C"),
    }
    if u in scale:
        s, b = scale[u]
        return value * s, b
    if u in ["v", "volt", "volts"]:
        return value, "V"
    if u in ["a", "amp", "amps"]:
        return value, "A"
    if u in ["ohm", "ω"]:
        return value, "ohm"
    if u in ["f"]:
        return value, "F"
    if u in ["c"]:
        return value, "C"
    return value, unit


def parse_quantities(question: str) -> Dict[str, float]:
    found: Dict[str, float] = {}
    for m in NUM_UNIT_RE.finditer(question.replace("=", " ")):
        v = float(m.group(1))
        b, u = _to_base(v, m.group(2))
        found[u] = b
    return found


def fmt(v: float) -> str:
    if abs(v) >= 1 and abs(v) < 1000:
        return f"{v:.4g}"
    return f"{v:.4e}"


class PhysicsEngine:
    def __init__(self, qa_items: List[PhysicsQA], llm: Optional[LLMClient] = None) -> None:
        self.qa_items = qa_items
        self.llm = llm

        formula_bank = [
            "Ohm's law: V = I * R",
            "Power: P = V * I = I^2 * R = V^2 / R",
            "Series resistance: R_eq = R1 + R2 + ...",
            "Parallel resistance: 1/R_eq = 1/R1 + 1/R2 + ...",
            "Capacitance: C = Q / V",
            "Capacitor energy: E = 0.5 * C * V^2",
        ]
        dataset_knowledge = [f"Q: {x.question} | A: {x.answer} {x.unit} | CoT: {x.cot}" for x in qa_items[:1200]]
        self.knowledge_docs = formula_bank + dataset_knowledge
        self.retriever = HybridRetriever(self.knowledge_docs)

    def _plan(self, question: str, retrieved: List[str]) -> List[str]:
        return [
            "Extract known quantities and normalize units.",
            "Select formulas from retrieved knowledge.",
            "Compute target variable deterministically.",
            "Format answer with unit and verify plausibility.",
        ]

    def _deterministic_solve(self, question: str) -> Optional[Dict[str, object]]:
        q = question.lower()
        vals = parse_quantities(question)
        premises: List[str] = []
        cot = ["Extracted values and normalized units."]

        if "current" in q and "V" in vals and "ohm" in vals and vals["ohm"] != 0:
            i = vals["V"] / vals["ohm"]
            premises = ["Ohm's law: V=IR"]
            cot.append("Used I=V/R.")
            return {"answer": f"{fmt(i)} A", "premises": premises, "cot": cot}

        if "voltage" in q and "A" in vals and "ohm" in vals:
            v = vals["A"] * vals["ohm"]
            premises = ["Ohm's law: V=IR"]
            cot.append("Used V=IR.")
            return {"answer": f"{fmt(v)} V", "premises": premises, "cot": cot}

        if "energy" in q and "capacitor" in q and "F" in vals and "V" in vals:
            e = 0.5 * vals["F"] * vals["V"] * vals["V"]
            premises = ["Capacitor energy: E=0.5CV^2"]
            cot.append("Used E=0.5CV^2.")
            return {"answer": f"{fmt(e)} J", "premises": premises, "cot": cot}

        if "capacitance" in q and "C" in vals and "V" in vals and vals["V"] != 0:
            c = vals["C"] / vals["V"]
            premises = ["Capacitance: C=Q/V"]
            cot.append("Used C=Q/V.")
            return {"answer": f"{fmt(c)} F", "premises": premises, "cot": cot}

        return None

    def _llm_explain(self, question: str, answer: str, premises: List[str], cot: List[str]) -> str:
        if not self.llm or not self.llm.ready:
            return f"Computed answer is {answer} using retrieved formulas and execution trace."
        pack = get_prompt_pack()
        prompt = pack.physics_explain + f"Question: {question}\nAnswer: {answer}\nPremises: {premises}\nTrace: {cot}\n"
        txt = self.llm.generate(prompt, max_new_tokens=120, temperature=0.0)
        return txt.strip() if txt else f"Computed answer is {answer} using retrieved formulas and execution trace."

    @staticmethod
    def _extract_retrieved_answer(text: str) -> Optional[str]:
        m = re.search(r"\|\s*A:\s*([^|]+)\|", text)
        if not m:
            return None
        ans = m.group(1).strip()
        return ans if ans else None

    def _safe_retrieval_fallback(self, hits) -> Optional[PhysicsResult]:
        if not hits:
            return None

        best = hits[0]
        second = hits[1] if len(hits) > 1 else None

        # Only fallback when best hit is clearly dominant.
        if best.score < 0.82:
            return None
        if second and (best.score - second.score) < 0.12:
            return None

        ra = self._extract_retrieved_answer(best.text)
        if not ra:
            return None

        return PhysicsResult(
            answer=ra,
            explanation="Used a high-confidence nearest solved example fallback.",
            cot=["Deterministic solver not applicable.", "Retrieved nearest solved sample with strong score margin."],
            premises=[best.text],
            confidence=0.35,
        )

    def solve(self, question: str) -> PhysicsResult:
        hits = self.retriever.search(question, k=5)
        retrieved = [h.text for h in hits]
        plan = self._plan(question, retrieved)

        out = self._deterministic_solve(question)
        if out:
            answer = str(out.get("answer", "Uncertain"))
            premises = [str(x) for x in out.get("premises", [])][:5] or retrieved[:2]
            cot = [str(x) for x in out.get("cot", [])][:8] or plan
            explanation = self._llm_explain(question, answer, premises, cot)
            conf = 0.9 if answer.lower() != "uncertain" else 0.45
            return PhysicsResult(answer=answer, explanation=explanation, cot=cot, premises=premises, confidence=conf)

        fb = self._safe_retrieval_fallback(hits)
        if fb:
            return fb

        if self.llm and self.llm.ready:
            obj = self.llm.generate_json(
                f"Solve and return JSON keys answer, explanation, cot, premises, confidence. Question: {question}",
                ["answer", "explanation", "cot", "premises", "confidence"],
            )
            if obj:
                cot = obj.get("cot") if isinstance(obj.get("cot"), list) else [str(obj.get("cot", ""))]
                prem = obj.get("premises") if isinstance(obj.get("premises"), list) else [str(obj.get("premises", ""))]
                try:
                    conf = float(obj.get("confidence", 0.45))
                except Exception:
                    conf = 0.45
                return PhysicsResult(
                    str(obj.get("answer", "Uncertain")),
                    str(obj.get("explanation", "")),
                    [str(x) for x in cot][:8],
                    [str(x) for x in prem][:8],
                    max(0.0, min(1.0, conf)),
                )

        return PhysicsResult("Uncertain", "Could not solve via deterministic path or safe fallback.", plan, retrieved[:3], 0.15)
