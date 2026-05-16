import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data import load_split
from .llm_client import LLMClient, load_llm_config
from .logic_engine import LogicEngine
from .normalizer import normalize_logic_answer, normalize_unit, split_answer_number_unit
from .physics_engine import PhysicsEngine


class XAIPipeline:
    def __init__(self, root_dir: Path) -> None:
        self.root = root_dir
        self.data_split = os.getenv("XAI_DATA_SPLIT", "train").strip().lower()
        self.llm = LLMClient(load_llm_config())

        logic_qa, physics_qa, self.data_paths = load_split(self.root, self.data_split)
        self.logic_engine = LogicEngine(logic_qa, llm=self.llm)
        self.physics_engine = PhysicsEngine(physics_qa, llm=self.llm)

    @staticmethod
    def route(question: str, premises_nl: Optional[List[str]]) -> str:
        if premises_nl and len(premises_nl) > 0:
            return "logic"
        q = (question or "").lower()
        logic_cues = [
            "does it follow that",
            "is it true that",
            "therefore",
            "premise",
            "conclusion",
            "all ",
            " if ",
            " then ",
            "a.",
            "b.",
            "c.",
            "d.",
        ]
        if any(cue in q for cue in logic_cues):
            return "logic"
        return "physics"

    @staticmethod
    def _verify(result: Dict[str, Any]) -> Dict[str, Any]:
        ans = str(result.get("answer", ""))
        exp = str(result.get("explanation", "")).lower()
        issues: List[str] = []

        if not ans:
            issues.append("empty_answer")
        if not exp:
            issues.append("empty_explanation")

        if re.search(r"\d", ans):
            if not re.search(r"\b(v|a|ohm|j|w|f|c)\b", ans.lower()) and not re.search(
                r"\b(voltage|current|resistance|energy|power|capacitance)\b", exp
            ):
                issues.append("missing_unit_signal")

        result.setdefault("meta", {})
        result["meta"]["verification_issues"] = issues
        return result

    @staticmethod
    def _calibrate(result: Dict[str, Any]) -> Dict[str, Any]:
        conf = float(result.get("confidence", 0.5))
        issues = result.get("meta", {}).get("verification_issues", [])
        if issues:
            conf = max(0.05, conf - 0.15 * len(issues))
        if str(result.get("answer", "")).lower() in ["uncertain", "unknown"]:
            conf = min(conf, 0.45)
        result["confidence"] = max(0.0, min(1.0, conf))
        return result

    def predict(self, question: str, premises_nl: Optional[List[str]] = None) -> Dict[str, Any]:
        task = self.route(question, premises_nl)
        if task == "logic":
            result = asdict(self.logic_engine.solve(question, premises_nl or []))
        else:
            result = asdict(self.physics_engine.solve(question))

        result.setdefault("answer", "Uncertain")
        result.setdefault("explanation", "No explanation available.")
        result.setdefault("fol", None)
        result.setdefault("cot", [])
        result.setdefault("premises", [])
        result.setdefault("confidence", 0.0)
        result.setdefault("meta", {})
        if task == "logic":
            result["answer"] = normalize_logic_answer(str(result.get("answer", "")))
        else:
            num, unit = split_answer_number_unit(str(result.get("answer", "")))
            if num is not None:
                nu = normalize_unit(unit)
                result["answer"] = f"{num:g} {nu}".strip()
            elif unit:
                result["answer"] = f"{str(result.get('answer', '')).strip()} {normalize_unit(unit)}".strip()
        result["meta"].update(
            {
                "task": task,
                "split": self.data_split,
                "llm_enabled": self.llm.config.enabled,
                "llm_ready": self.llm.ready,
                "model_id": self.llm.config.model_id,
                "model_path": self.llm.config.local_model_path,
                "mode": "hybrid_solver_first",
            }
        )

        result = self._verify(result)
        result = self._calibrate(result)
        return result
