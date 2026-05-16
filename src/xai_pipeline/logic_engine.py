import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .data import LogicQA
from .llm_client import LLMClient
from .prompt_registry import get_prompt_pack

try:
    from z3 import Bool, Implies, Not, Solver, unsat
    HAS_Z3 = True
except Exception:
    HAS_Z3 = False


@dataclass
class LogicResult:
    answer: str
    explanation: str
    fol: str
    cot: List[str]
    premises: List[str]
    confidence: float


IF_THEN_RE = re.compile(r"^if (.+), then (.+)\.?$", re.IGNORECASE)
ALL_RE = re.compile(r"^all (.+?) (?:are|is) (.+)\.?$", re.IGNORECASE)


def _norm(text: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).strip()
    return re.sub(r"\s+", "_", t)


def _to_atom(phrase: str) -> str:
    p = phrase.strip().lower().rstrip("?.")
    p = re.sub(r"^all\s+.+?\s+(?:are|is)\s+", "", p)
    return _norm(p)


def _predicate_from_clause(text: str) -> str:
    s = text.strip().lower().rstrip("?.")
    s = re.sub(r"^(?:a|an|the)\s+", "", s)
    m = re.search(r"\b(?:is|are)\s+(.+)$", s)
    if m:
        s = m.group(1)
    return _norm(s)


def parse_rules_regex(premises: List[str]) -> Tuple[Set[str], List[Tuple[str, str]]]:
    facts: Set[str] = set()
    rules: List[Tuple[str, str]] = []
    for p in premises:
        s = p.strip()
        m = IF_THEN_RE.match(s)
        if m:
            left, right = m.group(1), m.group(2)
            rules.append((_to_atom(left), _to_atom(right)))
            # Predicate abstraction improves quantified entailment matching.
            pa, pb = _predicate_from_clause(left), _predicate_from_clause(right)
            if pa and pb:
                rules.append((pa, pb))
            continue
        m = ALL_RE.match(s)
        if m:
            a, b = _to_atom(m.group(1)), _to_atom(m.group(2))
            facts.add(a)
            rules.append((a, b))
            # For quantified statements, also retain predicate-only abstractions.
            pa = _predicate_from_clause(m.group(1))
            pb = _predicate_from_clause(m.group(2))
            if pa:
                facts.add(pa)
            if pa and pb:
                rules.append((pa, pb))
    return facts, rules


def parse_rules_llm(llm: Optional[LLMClient], premises: List[str]) -> Optional[Tuple[Set[str], List[Tuple[str, str]], str]]:
    if not llm or not llm.ready:
        return None
    pack = get_prompt_pack()
    prompt = pack.logic_translate + f"Premises: {premises}\n"
    obj = llm.generate_json(prompt, required_keys=["facts", "rules"])
    if not obj:
        return None
    facts = set(str(x) for x in obj.get("facts", []))
    raw_rules = obj.get("rules", [])
    rules: List[Tuple[str, str]] = []
    for r in raw_rules:
        if isinstance(r, list) and len(r) == 2:
            rules.append((str(r[0]), str(r[1])))
    if not rules and not facts:
        return None
    fol = " ∧ ".join([f"Fact({f})" for f in sorted(facts)] + [f"({a}->{b})" for a, b in rules])
    return facts, rules, fol


def z3_entail(question: str, facts: Set[str], rules: List[Tuple[str, str]]) -> Tuple[str, str, List[str], float]:
    if not HAS_Z3:
        return "Unknown", "", ["z3-solver not installed."], 0.35

    atoms = set(facts)
    for a, b in rules:
        atoms.add(a)
        atoms.add(b)
    if not atoms:
        return "Unknown", "", ["No symbolic atoms extracted."], 0.3

    var = {a: Bool(a) for a in atoms}
    base = Solver()
    for a, b in rules:
        base.add(Implies(var[a], var[b]))
    for f in facts:
        base.add(var[f])

    q = question.lower().strip()
    cot = [
        "Built symbolic facts/rules.",
        "Constructed Z3 constraints.",
    ]

    m = re.search(r"does it follow that if (.+), then (.+)\?", q)
    if m:
        left, right = m.group(1), m.group(2)
        a, b = _to_atom(left), _to_atom(right)
        pa, pb = _predicate_from_clause(left), _predicate_from_clause(right)
        if a not in var:
            var[a] = Bool(a)
        if b not in var:
            var[b] = Bool(b)
        claim = Implies(var[a], var[b])
        if pa and pb and pa in var and pb in var:
            # Stronger proxy for quantified language.
            claim = Implies(var[pa], var[pb])
        s = Solver()
        s.add(base.assertions())
        s.add(Not(claim))
        if s.check() == unsat:
            return "Yes", f"Entails(Implies({a},{b}))", cot + ["Negated claim is UNSAT."], 0.86
        return "Unknown", f"NotEntails(Implies({a},{b}))", cot + ["Negated claim is SAT."], 0.5

    m = re.search(r"is it true that (.+)\?", q)
    if m:
        a = _to_atom(m.group(1))
        if a not in var:
            var[a] = Bool(a)
        s = Solver()
        s.add(base.assertions())
        s.add(Not(var[a]))
        if s.check() == unsat:
            return "Yes", f"Entails({a})", cot + ["Negated proposition is UNSAT."], 0.84
        return "Unknown", f"NotEntails({a})", cot + ["Negated proposition is SAT."], 0.48

    # MCQ evaluation: choose option whose proposition is entailed.
    opts = re.findall(r"([A-D])\.\s*(.+)", question, flags=re.IGNORECASE)
    if opts:
        entailed: List[str] = []
        for label, text in opts:
            prop = _predicate_from_clause(text)
            if not prop:
                continue
            if prop not in var:
                var[prop] = Bool(prop)
            s = Solver()
            s.add(base.assertions())
            s.add(Not(var[prop]))
            if s.check() == unsat:
                entailed.append(label.upper())
        if len(entailed) == 1:
            lb = entailed[0]
            return lb, f"EntailsOption({lb})", cot + [f"Exactly one option is entailed: {lb}."], 0.76
        if len(entailed) > 1:
            return "Unknown", "MultipleEntailedOptions", cot + ["Multiple options appear entailed."], 0.4
        return "Unknown", "NoEntailedOption", cot + ["No option could be formally entailed."], 0.4

    # Open question neutral fallback.
    return "Unknown", "", cot + ["Question not in direct entailment form."], 0.4


class LogicEngine:
    def __init__(self, qa_items: List[LogicQA], llm: Optional[LLMClient] = None) -> None:
        self.qa_items = qa_items
        self.llm = llm

    def _llm_fallback(self, question: str, premises_nl: List[str]) -> Optional[LogicResult]:
        if not self.llm or not self.llm.ready:
            return None
        pack = get_prompt_pack()
        prompt = pack.logic_answer + f"Premises: {premises_nl}\nQuestion: {question}\n"
        obj = self.llm.generate_json(prompt, ["answer", "explanation", "fol", "cot", "premises", "confidence"])
        if not obj:
            return None
        cot = obj.get("cot") if isinstance(obj.get("cot"), list) else [str(obj.get("cot", ""))]
        prem = obj.get("premises") if isinstance(obj.get("premises"), list) else premises_nl[:8]
        try:
            conf = float(obj.get("confidence", 0.45))
        except Exception:
            conf = 0.45
        return LogicResult(
            answer=str(obj.get("answer", "Unknown")),
            explanation=str(obj.get("explanation", "")),
            fol=str(obj.get("fol", "")),
            cot=[str(x) for x in cot][:8],
            premises=[str(x) for x in prem][:8],
            confidence=max(0.0, min(1.0, conf)),
        )

    def solve(self, question: str, premises_nl: List[str]) -> LogicResult:
        llm_parse = parse_rules_llm(self.llm, premises_nl)
        if llm_parse:
            facts, rules, fol = llm_parse
            parse_step = "Parsed NL->FOL rules with LLM translator."
        else:
            facts, rules = parse_rules_regex(premises_nl)
            fol = " ∧ ".join([f"Fact({f})" for f in sorted(facts)] + [f"({a}->{b})" for a, b in rules])
            parse_step = "Parsed NL->FOL rules with regex fallback."

        answer, fol_entail, cot, conf = z3_entail(question, facts, rules)
        cot = [parse_step] + cot
        fol_final = fol_entail or fol
        explanation = f"Used symbolic translation and Z3 entailment checking; decision is `{answer}`."

        if conf < 0.55:
            fb = self._llm_fallback(question, premises_nl)
            if fb and fb.confidence >= conf:
                return fb

        return LogicResult(
            answer=answer,
            explanation=explanation,
            fol=fol_final,
            cot=cot,
            premises=premises_nl[:8],
            confidence=conf,
        )
