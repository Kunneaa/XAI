import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class PromptPack:
    logic_translate: str
    logic_answer: str
    physics_code: str
    physics_explain: str


PROMPTS: Dict[str, PromptPack] = {
    "v1": PromptPack(
        logic_translate=(
            "Convert premises into JSON with keys facts and rules.\n"
            "- facts: list of atomic predicates (snake_case strings)\n"
            "- rules: list of [antecedent, consequent] pairs\n"
            "Return JSON only.\n"
        ),
        logic_answer=(
            "Given premises and question, return JSON with keys: answer, explanation, fol, cot, premises, confidence.\n"
            "Answer should be Yes/No/Unknown or option label A/B/C/D for MCQ.\n"
        ),
        physics_code=(
            "Write Python code that solves the physics question.\n"
            "Return JSON with key code only. Code must set OUTPUT dict with keys: answer, premises, cot.\n"
        ),
        physics_explain="Write a concise, verifiable explanation (<=70 words).\n",
    ),
    "v2": PromptPack(
        logic_translate=(
            "You are a formal-logic parser.\n"
            "Transform premises into machine-usable JSON only.\n"
            "Schema: {\"facts\": [str], \"rules\": [[str,str]]}.\n"
            "Prefer normalized predicate names in snake_case.\n"
        ),
        logic_answer=(
            "You are a strict logic judge.\n"
            "Return ONLY JSON: answer, explanation, fol, cot, premises, confidence.\n"
            "Keep explanation concise and verifiable from premises.\n"
        ),
        physics_code=(
            "You are a physics coding agent.\n"
            "Generate Python code only via JSON {\"code\": \"...\"}.\n"
            "Code must compute final numeric answer with unit in OUTPUT['answer'].\n"
        ),
        physics_explain="Produce concise explanation grounded in formulas and execution trace (<=70 words).\n",
    ),
}


def get_prompt_pack() -> PromptPack:
    version = os.getenv("XAI_PROMPT_VERSION", "v1").strip().lower()
    return PROMPTS.get(version, PROMPTS["v1"])
