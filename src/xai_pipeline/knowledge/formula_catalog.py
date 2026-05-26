"""Prompt-safe formula catalog synthesized from dataset coverage and registry cards.

The dataset is used offline to identify recurring formula families. At runtime
this module exposes only code-owned registry metadata: IDs, dimensions, target
types, and premises. It never exposes dataset answers, dataset CoT text, or row
IDs to the local LLM.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Iterable

from .registries import FORMULA_IDS, FORMULA_REGISTRY, formula_execution_branch, formula_family_for_id, formula_logic_catalog


DATASET_FORMULA_CATALOG_VERSION = "physics-problems-text-only-v1-registry-synthesis"
DATASET_FORMULA_CATALOG_SOURCE = {
    "file": "Physics_Problems_Text_Only.csv",
    "rows": "current_local_csv",
    "usage": "offline formula-family coverage synthesis only",
    "inference_leakage_policy": "no dataset answers, row ids, or dataset cot are exposed",
}


def formula_catalog_for_prompt(
    *,
    route_task_type: str | None = None,
    candidate_formula_ids: Iterable[str] | None = None,
    max_route_cards: int = 48,
    route_local_only: bool = False,
) -> dict[str, Any]:
    """Return a compact catalog that lets the LLM choose registry formula IDs.

    By default this can expose the complete executable ID list for audits and
    tests. Runtime LLM calls should use ``formula_prompt_pack`` instead.
    """

    candidate_ids = [fid for fid in dict.fromkeys(candidate_formula_ids or []) if fid in FORMULA_IDS]
    route_ids = [
        fid
        for fid, spec in FORMULA_REGISTRY.items()
        if fid in FORMULA_IDS and route_task_type and spec.task_type == route_task_type
    ]
    executable_ids = sorted(FORMULA_IDS)
    full_catalog = not route_local_only or os.environ.get("XAI_LLM_FULL_FORMULA_CATALOG", "").strip().lower() in {"1", "true", "yes", "on"}
    route_local_ids = candidate_ids or route_ids[:max_route_cards]
    allowed_ids = executable_ids if full_catalog or not route_local_ids else route_local_ids
    detailed_ids = allowed_ids if full_catalog else (candidate_ids or route_ids[: min(6, max_route_cards)])
    logic_catalog = formula_logic_catalog()
    return {
        "catalog_version": DATASET_FORMULA_CATALOG_VERSION,
        "source": DATASET_FORMULA_CATALOG_SOURCE,
        "formula_count": len(allowed_ids),
        "global_formula_count": len(executable_ids),
        "all_formula_ids": allowed_ids,
        "allowed_formula_ids": allowed_ids,
        "candidate_formula_ids": candidate_ids,
        "route_task_type": route_task_type,
        "route_formula_ids": route_ids,
        "detailed_formula_cards": [_card(fid) for fid in detailed_ids],
        "family_index": formula_family_index() if full_catalog else {route_task_type: route_ids} if route_task_type else {},
        "logic_families": _catalog_logic_families(logic_catalog, allowed_ids, full_catalog=full_catalog),
        "catalog_scope": "global" if full_catalog or not route_local_ids else "route_local",
        "llm_rule": "LLM may cite formula_id values only; formulas are executed only by deterministic engines.",
    }


def formula_prompt_pack(
    *,
    route_task_type: str | None = None,
    candidate_formula_ids: Iterable[str] | None = None,
    available_dimensions: Iterable[str] | None = None,
    target_dimensions: Iterable[str] | None = None,
    route_reasons: Iterable[str] | None = None,
    max_formula_ids: int = 12,
) -> dict[str, Any]:
    """Return the tiny formula menu that is safe to put in every LLM prompt.

    This is intentionally smaller than ``formula_catalog_for_prompt``. The full
    registry stays in code for compiler/verifier checks. The LLM receives only
    route-local IDs and compact dimension cards.
    """

    candidate_ids = [fid for fid in dict.fromkeys(candidate_formula_ids or []) if fid in FORMULA_IDS]
    route_ids = [
        fid
        for fid, spec in FORMULA_REGISTRY.items()
        if fid in FORMULA_IDS and route_task_type and spec.task_type == route_task_type
    ]
    expand_route = os.environ.get("XAI_LLM_EXPAND_ROUTE_FORMULAS", "").strip().lower() in {"1", "true", "yes", "on"}
    allowed_ids = list(dict.fromkeys([*(candidate_ids or []), *(route_ids if expand_route else [])]))[:max_formula_ids]
    if not allowed_ids:
        allowed_ids = candidate_ids[:max_formula_ids]
    if not allowed_ids:
        allowed_ids = route_ids[:max_formula_ids]
    logic_catalog = formula_logic_catalog()
    family_ids = sorted({formula_family_for_id(fid) for fid in allowed_ids if formula_family_for_id(fid)})
    family_hints = {
        family_id: {
            "domain": logic_catalog["families"][family_id]["domain"],
            "plan_logic": logic_catalog["families"][family_id]["plan_logic"],
        }
        for family_id in family_ids
        if family_id in logic_catalog.get("families", {})
    }
    return {
        "scope": "route_local_compact",
        "route_task_type": route_task_type,
        "global_formula_count": len(FORMULA_IDS),
        "hidden_route_formula_count": len(route_ids),
        "allowed_formula_ids": allowed_ids,
        "candidate_formula_ids": candidate_ids,
        "cards": [_compact_card(fid) for fid in allowed_ids],
        "decision_evidence": _decision_evidence(
            allowed_ids=allowed_ids,
            candidate_ids=candidate_ids,
            available_dimensions=list(available_dimensions or []),
            target_dimensions=list(target_dimensions or []),
            route_reasons=list(route_reasons or []),
        ),
        "family_hints": family_hints,
        "policy": "The full registry is hidden from the LLM; compiler and verifier check chosen IDs against code-owned FORMULA_REGISTRY.",
    }


@lru_cache(maxsize=1)
def formula_family_index() -> dict[str, list[str]]:
    """Group all known formula IDs by route/task family."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for formula_id, spec in FORMULA_REGISTRY.items():
        if formula_id not in FORMULA_IDS:
            continue
        grouped[spec.task_type].append(formula_id)
    return {task_type: sorted(ids) for task_type, ids in sorted(grouped.items())}


def _card(formula_id: str) -> dict[str, Any]:
    spec = FORMULA_REGISTRY[formula_id]
    return {
        "formula_id": spec.formula_id,
        "task_type": spec.task_type,
        "principle_id": spec.principle_id,
        "required_dimensions": list(spec.required_dimensions),
        "target_dimension": spec.target_dimension,
        "target_unit": spec.target_unit,
        "expression": spec.expression,
        "premise": spec.premise,
        "logic_family": formula_family_for_id(formula_id),
        "execution_branch": formula_execution_branch(formula_id),
    }


def _compact_card(formula_id: str) -> dict[str, Any]:
    spec = FORMULA_REGISTRY[formula_id]
    return {
        "id": spec.formula_id,
        "principle": spec.principle_id,
        "need": list(spec.required_dimensions),
        "out": spec.target_dimension,
        "unit": spec.target_unit,
        "family": formula_family_for_id(formula_id),
        "branch": formula_execution_branch(formula_id),
    }


def _decision_evidence(
    *,
    allowed_ids: list[str],
    candidate_ids: list[str],
    available_dimensions: list[str],
    target_dimensions: list[str],
    route_reasons: list[str],
) -> dict[str, Any]:
    """Explain why each small prompt formula is a candidate.

    The evidence is dimension-only and registry-owned. It helps the LLM choose
    among IDs without exposing equations, dataset examples, or answers.
    """

    return {
        "available_dimensions": _ordered_dimensions(available_dimensions),
        "target_dimensions": _ordered_dimensions(target_dimensions),
        "route_reasons": route_reasons[:4],
        "candidates": [
            _candidate_evidence(
                formula_id,
                selected_by_graph=formula_id in candidate_ids,
                available_dimensions=available_dimensions,
                target_dimensions=target_dimensions,
            )
            for formula_id in allowed_ids
        ],
        "selection_rule": "Prefer selected_by_graph candidates whose missing_dimensions is empty and target_match is true.",
    }


def _candidate_evidence(
    formula_id: str,
    *,
    selected_by_graph: bool,
    available_dimensions: list[str],
    target_dimensions: list[str],
) -> dict[str, Any]:
    spec = FORMULA_REGISTRY[formula_id]
    missing = _missing_dimensions(available_dimensions, list(spec.required_dimensions))
    return {
        "formula_id": formula_id,
        "selected_by_graph": selected_by_graph,
        "need": list(spec.required_dimensions),
        "out": spec.target_dimension,
        "branch": formula_execution_branch(formula_id),
        "target_match": spec.target_dimension in set(target_dimensions) if target_dimensions else False,
        "missing_dimensions": missing,
        "input_match": not missing,
    }


def _missing_dimensions(available_dimensions: list[str], required_dimensions: list[str]) -> list[str]:
    pool = Counter(available_dimensions)
    missing: list[str] = []
    for dimension in required_dimensions:
        if pool[dimension] <= 0:
            missing.append(dimension)
        else:
            pool[dimension] -= 1
    return missing


def _ordered_dimensions(dimensions: Iterable[str]) -> list[str]:
    return [dimension for dimension in dict.fromkeys(dimensions) if dimension]


def _catalog_logic_families(logic_catalog: dict[str, Any], allowed_ids: list[str], *, full_catalog: bool) -> dict[str, Any]:
    allowed = set(allowed_ids)
    families = {}
    for family_id, family in (logic_catalog.get("families") or {}).items():
        family_formula_ids = [formula_id for formula_id in family.get("formula_ids", []) if formula_id in allowed]
        if not full_catalog and not family_formula_ids:
            continue
        payload = dict(family)
        payload["formula_ids"] = family_formula_ids if not full_catalog else list(family.get("formula_ids", []))
        families[family_id] = payload
    return {
        "families": families,
        "policy": logic_catalog.get("policy"),
    }
