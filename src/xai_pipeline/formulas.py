"""Formula registry facade.

The core describes formulas as a separate module. The project keeps the
canonical data in ``registries.py`` and exposes this facade so future executors
can depend on a stable formulas API without duplicating IDs.
"""

from __future__ import annotations

from .registries import FORMULA_IDS, FORMULA_REGISTRY, FormulaSpec


def get_formula(formula_id: str) -> FormulaSpec | None:
    return FORMULA_REGISTRY.get(formula_id)


def formulas_for_task(task_type: str) -> list[FormulaSpec]:
    return [spec for spec in FORMULA_REGISTRY.values() if spec.task_type == task_type]


def formula_ids_for_task(task_type: str) -> list[str]:
    return [spec.formula_id for spec in formulas_for_task(task_type)]
