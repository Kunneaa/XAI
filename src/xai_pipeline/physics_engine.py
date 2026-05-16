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


NUM_UNIT_RE = re.compile(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-ZμµΩ/\-\^0-9]+)")
K_COULOMB = 8.9875517923e9


def _to_base(value: float, unit: str) -> Tuple[float, str]:
    raw = unit.strip().replace("µ", "μ")
    u = raw.lower()
    u = u.replace("ω", "ohm")
    u = u.replace(" ", "")
    u = u.replace("**", "^")
    # normalize common ASCII forms
    u = u.replace("nc^-1", "n/c")
    u = u.replace("v*m^-1", "v/m")
    u = u.replace("v/m", "vper_m")
    u = u.replace("n/c", "vper_m")

    scale = {
        "mv": (1e-3, "V"),
        "kv": (1e3, "V"),
        "ma": (1e-3, "A"),
        "μa": (1e-6, "A"),
        "ua": (1e-6, "A"),
        "ka": (1e3, "A"),
        "mohm": (1e-3, "ohm"),
        "kohm": (1e3, "ohm"),
        "μf": (1e-6, "F"),
        "uf": (1e-6, "F"),
        "nf": (1e-9, "F"),
        "pf": (1e-12, "F"),
        "mc": (1e-3, "C"),
        "μc": (1e-6, "C"),
        "uc": (1e-6, "C"),
        "nc": (1e-9, "C"),
        "pc": (1e-12, "C"),
        "mj": (1e-3, "J"),
        "kj": (1e3, "J"),
        "mw": (1e-3, "W"),
        "kw": (1e3, "W"),
        "mvper_m": (1e-3, "V/m"),
        "kvper_m": (1e3, "V/m"),
        "cm": (1e-2, "m"),
        "mm": (1e-3, "m"),
        "km": (1e3, "m"),
        "nm": (1e-9, "m"),
    }
    if u in scale:
        s, b = scale[u]
        return value * s, b
    if u in ["v", "volt", "volts"]:
        return value, "V"
    if u in ["a", "amp", "amps"]:
        return value, "A"
    if u in ["ohm"]:
        return value, "ohm"
    if u in ["f"]:
        return value, "F"
    if u in ["c"]:
        return value, "C"
    if u in ["w"]:
        return value, "W"
    if u in ["j"]:
        return value, "J"
    if u in ["n"]:
        return value, "N"
    if u in ["m"]:
        return value, "m"
    if u in ["vper_m", "vm-1"]:
        return value, "V/m"
    return value, raw


def parse_quantities(question: str) -> Dict[str, List[float]]:
    found: Dict[str, List[float]] = {}
    for m in NUM_UNIT_RE.finditer(question.replace("=", " ")):
        v = float(m.group(1))
        b, u = _to_base(v, m.group(2))
        found.setdefault(u, []).append(b)
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
            "Capacitor energy equivalents: E = Q^2/(2C) = 0.5*Q*V",
            "Charge on capacitor: Q = C * V",
            "Series capacitance: 1/C_eq = 1/C1 + 1/C2 + ...",
            "Parallel capacitance: C_eq = C1 + C2 + ...",
            "Electric field-force relation: F = q * E",
            "Electric field from force: E = F / q",
            "Potential difference in uniform field: V = E * d",
            "Point-charge electric field: E = k * |Q| / r^2",
            "Electric potential of point charge: V = k * Q / r",
            "Coulomb force magnitude: F = k * |q1*q2| / r^2",
            "Electrostatic potential energy: U = k * q1 * q2 / r",
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
        scalar_vals = {k: v[-1] for k, v in vals.items() if v}
        has = lambda *keys: all(k in scalar_vals for k in keys)
        mentions = lambda *terms: any(t in q for t in terms)
        premises: List[str] = []
        cot = ["Extracted values and normalized units."]
        explanation = ""

        def done(answer: str, formula: str, substitution: str) -> Dict[str, object]:
            nonlocal explanation
            explanation = f"Apply {formula}. Substitute {substitution}. Therefore the answer is {answer}."
            return {"answer": answer, "premises": premises, "cot": cot, "explanation": explanation}

        if mentions("current") and has("V", "ohm") and scalar_vals["ohm"] != 0:
            i = scalar_vals["V"] / scalar_vals["ohm"]
            premises = ["Ohm's law: V=IR"]
            cot.append("Used I=V/R.")
            ans = f"{fmt(i)} A"
            return done(ans, "I = V / R", f"V={fmt(scalar_vals['V'])}, R={fmt(scalar_vals['ohm'])}")

        if mentions("voltage", "potential difference") and has("A", "ohm"):
            v = scalar_vals["A"] * scalar_vals["ohm"]
            premises = ["Ohm's law: V=IR"]
            cot.append("Used V=IR.")
            ans = f"{fmt(v)} V"
            return done(ans, "V = I * R", f"I={fmt(scalar_vals['A'])}, R={fmt(scalar_vals['ohm'])}")

        if mentions("resistance") and has("V", "A") and scalar_vals["A"] != 0:
            r = scalar_vals["V"] / scalar_vals["A"]
            premises = ["Ohm's law: R=V/I"]
            cot.append("Used R=V/I.")
            ans = f"{fmt(r)} ohm"
            return done(ans, "R = V / I", f"V={fmt(scalar_vals['V'])}, I={fmt(scalar_vals['A'])}")

        if mentions("power") and has("V", "A"):
            p = scalar_vals["V"] * scalar_vals["A"]
            premises = ["Power: P=VI"]
            cot.append("Used P=VI.")
            ans = f"{fmt(p)} W"
            return done(ans, "P = V * I", f"V={fmt(scalar_vals['V'])}, I={fmt(scalar_vals['A'])}")

        if mentions("power") and has("A", "ohm"):
            p = scalar_vals["A"] * scalar_vals["A"] * scalar_vals["ohm"]
            premises = ["Power: P=I^2R"]
            cot.append("Used P=I^2R.")
            ans = f"{fmt(p)} W"
            return done(ans, "P = I^2 * R", f"I={fmt(scalar_vals['A'])}, R={fmt(scalar_vals['ohm'])}")

        if mentions("power") and has("V", "ohm") and scalar_vals["ohm"] != 0:
            p = (scalar_vals["V"] * scalar_vals["V"]) / scalar_vals["ohm"]
            premises = ["Power: P=V^2/R"]
            cot.append("Used P=V^2/R.")
            ans = f"{fmt(p)} W"
            return done(ans, "P = V^2 / R", f"V={fmt(scalar_vals['V'])}, R={fmt(scalar_vals['ohm'])}")

        if mentions("series") and mentions("resistance") and "ohm" in vals and len(vals["ohm"]) >= 2:
            req = sum(vals["ohm"])
            premises = ["Series resistance: R_eq = R1 + R2 + ..."]
            cot.append("Summed all resistor values in series.")
            ans = f"{fmt(req)} ohm"
            return done(ans, "R_eq = sum(Ri)", f"R={', '.join(fmt(x) for x in vals['ohm'])}")

        if mentions("parallel") and mentions("resistance") and "ohm" in vals and len(vals["ohm"]) >= 2:
            denom = sum(1.0 / x for x in vals["ohm"] if x != 0)
            if denom != 0:
                req = 1.0 / denom
                premises = ["Parallel resistance: 1/R_eq = 1/R1 + 1/R2 + ..."]
                cot.append("Computed reciprocal sum for parallel resistors.")
                ans = f"{fmt(req)} ohm"
                return done(ans, "R_eq = 1 / sum(1/Ri)", f"R={', '.join(fmt(x) for x in vals['ohm'])}")

        if mentions("energy") and mentions("capacitor") and has("F", "V"):
            e = 0.5 * scalar_vals["F"] * scalar_vals["V"] * scalar_vals["V"]
            premises = ["Capacitor energy: E=0.5CV^2"]
            cot.append("Used E=0.5CV^2.")
            ans = f"{fmt(e)} J"
            return done(ans, "E = 0.5 * C * V^2", f"C={fmt(scalar_vals['F'])}, V={fmt(scalar_vals['V'])}")

        if mentions("capacitance") and has("C", "V") and scalar_vals["V"] != 0:
            c = scalar_vals["C"] / scalar_vals["V"]
            premises = ["Capacitance: C=Q/V"]
            cot.append("Used C=Q/V.")
            ans = f"{fmt(c)} F"
            return done(ans, "C = Q / V", f"Q={fmt(scalar_vals['C'])}, V={fmt(scalar_vals['V'])}")

        if mentions("charge") and has("F", "V"):
            qv = scalar_vals["F"] * scalar_vals["V"]
            premises = ["Capacitor charge: Q=CV"]
            cot.append("Used Q=CV.")
            ans = f"{fmt(qv)} C"
            return done(ans, "Q = C * V", f"C={fmt(scalar_vals['F'])}, V={fmt(scalar_vals['V'])}")

        if mentions("energy") and mentions("capacitor") and has("C", "V"):
            ee = 0.5 * scalar_vals["C"] * scalar_vals["V"]
            premises = ["Capacitor energy: E=0.5QV"]
            cot.append("Used E=0.5QV.")
            ans = f"{fmt(ee)} J"
            return done(ans, "E = 0.5 * Q * V", f"Q={fmt(scalar_vals['C'])}, V={fmt(scalar_vals['V'])}")

        if mentions("energy") and mentions("capacitor") and has("C", "F") and scalar_vals["F"] != 0:
            ee = (scalar_vals["C"] * scalar_vals["C"]) / (2.0 * scalar_vals["F"])
            premises = ["Capacitor energy: E=Q^2/(2C)"]
            cot.append("Used E=Q^2/(2C).")
            ans = f"{fmt(ee)} J"
            return done(ans, "E = Q^2 / (2C)", f"Q={fmt(scalar_vals['C'])}, C={fmt(scalar_vals['F'])}")

        if mentions("series") and mentions("capacitance") and "F" in vals and len(vals["F"]) >= 2:
            denom = sum(1.0 / x for x in vals["F"] if x != 0)
            if denom != 0:
                ceq = 1.0 / denom
                premises = ["Series capacitance: 1/C_eq = 1/C1 + 1/C2 + ..."]
                cot.append("Computed reciprocal sum for series capacitors.")
                ans = f"{fmt(ceq)} F"
                return done(ans, "C_eq = 1 / sum(1/Ci)", f"C={', '.join(fmt(x) for x in vals['F'])}")

        if mentions("parallel") and mentions("capacitance") and "F" in vals and len(vals["F"]) >= 2:
            ceq = sum(vals["F"])
            premises = ["Parallel capacitance: C_eq = C1 + C2 + ..."]
            cot.append("Summed all capacitor values in parallel.")
            ans = f"{fmt(ceq)} F"
            return done(ans, "C_eq = sum(Ci)", f"C={', '.join(fmt(x) for x in vals['F'])}")

        if mentions("electric field") and mentions("force") and has("C", "N") and scalar_vals["C"] != 0:
            ef = scalar_vals["N"] / scalar_vals["C"]
            premises = ["Electric field: E=F/q"]
            cot.append("Used E=F/q.")
            ans = f"{fmt(ef)} V/m"
            return done(ans, "E = F / q", f"F={fmt(scalar_vals['N'])}, q={fmt(scalar_vals['C'])}")

        if mentions("force") and mentions("electric field") and has("C", "V/m"):
            ff = scalar_vals["C"] * scalar_vals["V/m"]
            premises = ["Electric force: F=qE"]
            cot.append("Used F=qE.")
            ans = f"{fmt(ff)} N"
            return done(ans, "F = q * E", f"q={fmt(scalar_vals['C'])}, E={fmt(scalar_vals['V/m'])}")

        if mentions("potential difference", "voltage") and has("V/m", "m"):
            vv = scalar_vals["V/m"] * scalar_vals["m"]
            premises = ["Uniform field relation: V=Ed"]
            cot.append("Used V=Ed.")
            ans = f"{fmt(vv)} V"
            return done(ans, "V = E * d", f"E={fmt(scalar_vals['V/m'])}, d={fmt(scalar_vals['m'])}")

        if mentions("electric field") and mentions("point charge") and has("C", "m") and scalar_vals["m"] != 0:
            ee = K_COULOMB * abs(scalar_vals["C"]) / (scalar_vals["m"] ** 2)
            premises = ["Point-charge field: E=k|Q|/r^2"]
            cot.append("Used E=k|Q|/r^2.")
            ans = f"{fmt(ee)} V/m"
            return done(ans, "E = k * |Q| / r^2", f"Q={fmt(scalar_vals['C'])}, r={fmt(scalar_vals['m'])}")

        if mentions("electric potential") and mentions("point charge") and has("C", "m") and scalar_vals["m"] != 0:
            vp = K_COULOMB * scalar_vals["C"] / scalar_vals["m"]
            premises = ["Point-charge potential: V=kQ/r"]
            cot.append("Used V=kQ/r.")
            ans = f"{fmt(vp)} V"
            return done(ans, "V = k * Q / r", f"Q={fmt(scalar_vals['C'])}, r={fmt(scalar_vals['m'])}")

        if mentions("coulomb") and mentions("force") and "C" in vals and len(vals["C"]) >= 2 and has("m") and scalar_vals["m"] != 0:
            fmag = K_COULOMB * abs(vals["C"][0] * vals["C"][1]) / (scalar_vals["m"] ** 2)
            premises = ["Coulomb law: F=k|q1q2|/r^2"]
            cot.append("Used Coulomb's law.")
            ans = f"{fmt(fmag)} N"
            return done(
                ans,
                "F = k * |q1*q2| / r^2",
                f"q1={fmt(vals['C'][0])}, q2={fmt(vals['C'][1])}, r={fmt(scalar_vals['m'])}",
            )

        if mentions("potential energy", "electrostatic energy") and "C" in vals and len(vals["C"]) >= 2 and has("m") and scalar_vals["m"] != 0:
            uu = K_COULOMB * vals["C"][0] * vals["C"][1] / scalar_vals["m"]
            premises = ["Electrostatic potential energy: U=kq1q2/r"]
            cot.append("Used U=kq1q2/r.")
            ans = f"{fmt(uu)} J"
            return done(
                ans,
                "U = k * q1 * q2 / r",
                f"q1={fmt(vals['C'][0])}, q2={fmt(vals['C'][1])}, r={fmt(scalar_vals['m'])}",
            )

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
            explanation = str(out.get("explanation", "")).strip() or self._llm_explain(question, answer, premises, cot)
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
