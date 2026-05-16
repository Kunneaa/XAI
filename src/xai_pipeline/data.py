import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class LogicQA:
    question: str
    answer: str
    explanation: str
    premises_nl: List[str]


@dataclass
class PhysicsQA:
    question: str
    answer: str
    unit: str
    cot: str


@dataclass
class DataPaths:
    logic_train: Path
    logic_test: Path
    physics_train: Path
    physics_test: Path


def resolve_data_paths(root: Path) -> DataPaths:
    return DataPaths(
        logic_train=root / "Logic_Based_Educational_Queries.train.json",
        logic_test=root / "Logic_Based_Educational_Queries.test.json",
        physics_train=root / "Physics_Problems_Text_Only.train.csv",
        physics_test=root / "Physics_Problems_Text_Only.test.csv",
    )


def load_logic_dataset(path: Path) -> List[LogicQA]:
    with path.open("r", encoding="utf-8") as f:
        raw: List[Dict[str, Any]] = json.load(f)

    out: List[LogicQA] = []
    for row in raw:
        premises = [str(x) for x in row.get("premises-NL", [])]
        questions = row.get("questions", [])
        answers = row.get("answers", [])
        exps = row.get("explanation", [])
        n = min(len(questions), len(answers))
        for i in range(n):
            out.append(
                LogicQA(
                    question=str(questions[i]),
                    answer=str(answers[i]),
                    explanation=str(exps[i]) if i < len(exps) else "",
                    premises_nl=premises,
                )
            )
    return out


def load_physics_dataset(path: Path) -> List[PhysicsQA]:
    df = pd.read_csv(path, usecols=["question", "answer", "unit", "cot"])
    rows = df.to_dict("records")
    return [
        PhysicsQA(
            question=str(r.get("question", "")),
            answer=str(r.get("answer", "")),
            unit=str(r.get("unit", "")),
            cot=str(r.get("cot", "")),
        )
        for r in rows
    ]


def load_split(root: Path, split: str) -> Tuple[List[LogicQA], List[PhysicsQA], DataPaths]:
    paths = resolve_data_paths(root)
    split_name = split.strip().lower()
    if split_name == "test":
        logic = load_logic_dataset(paths.logic_test)
        physics = load_physics_dataset(paths.physics_test)
    else:
        logic = load_logic_dataset(paths.logic_train)
        physics = load_physics_dataset(paths.physics_train)
    return logic, physics, paths


def load_train_test(root: Path) -> Tuple[List[LogicQA], List[LogicQA], List[PhysicsQA], List[PhysicsQA], DataPaths]:
    paths = resolve_data_paths(root)
    logic_train = load_logic_dataset(paths.logic_train)
    logic_test = load_logic_dataset(paths.logic_test)
    physics_train = load_physics_dataset(paths.physics_train)
    physics_test = load_physics_dataset(paths.physics_test)
    return logic_train, logic_test, physics_train, physics_test, paths
