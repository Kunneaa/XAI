"""Schema and registry validation for deterministic pipeline states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .registries import ANSWER_TYPES, FORMULA_IDS, FORMULA_REGISTRY, PRINCIPLE_IDS, TASK_TYPES
from .units import unit_info


@dataclass(frozen=True)
class RegistryValidationResult:
    ok: bool
    issues: List[str]
    trace: dict

    def to_dict(self):
        return {"ok": self.ok, "issues": list(self.issues), "trace": dict(self.trace)}


def validate_registry_state(front_payload: dict, route_result, solver_result: Optional[object] = None) -> RegistryValidationResult:
    """Validate code-owned IDs, units, and formula compatibility.

    This intentionally validates only deterministic artifacts. Qwen plans are
    checked separately by ``verifier.validate_plan``.
    """

    issues: List[str] = []
    answer_type = front_payload.get("answer_type_hint")
    if answer_type not in ANSWER_TYPES:
        issues.append(f"unknown_answer_type:{answer_type}")
    if route_result.task_type not in TASK_TYPES:
        issues.append(f"unknown_task_type:{route_result.task_type}")

    for index, quantity in enumerate(front_payload.get("quantities", [])):
        unit = quantity.get("unit")
        info = unit_info(unit or "")
        if info is None:
            issues.append(f"unknown_unit:{index}:{unit}")
            continue
        if quantity.get("dimension") != info.dimension:
            issues.append(f"dimension_unit_mismatch:{index}:{quantity.get('dimension')}:{info.dimension}")
        try:
            value = float(quantity.get("value"))
        except (TypeError, ValueError):
            issues.append(f"non_numeric_quantity:{index}")
            continue
        dimension = quantity.get("dimension")
        if dimension in {"resistance", "capacitance", "inductance", "mass", "time", "frequency", "area", "turn_density"} and value < 0:
            issues.append(f"impossible_negative_{dimension}:{index}")
        if dimension in {"length", "time", "resistance", "capacitance", "inductance", "mass"} and value == 0:
            context = str(quantity.get("context") or "").lower()
            if dimension == "length" and any(cue in context for cue in ["distance", "separation", "apart", "between"]):
                issues.append(f"impossible_zero_distance:{index}")
            elif dimension in {"time", "resistance", "capacitance", "inductance", "mass"}:
                issues.append(f"impossible_zero_{dimension}:{index}")

    if solver_result is not None and getattr(solver_result, "solved", False):
        formula_id = getattr(solver_result, "formula_id", None)
        principle_id = getattr(solver_result, "principle_id", None)
        if formula_id not in FORMULA_IDS:
            issues.append(f"unknown_formula_id:{formula_id}")
        if principle_id not in PRINCIPLE_IDS:
            issues.append(f"unknown_principle_id:{principle_id}")
        formula = FORMULA_REGISTRY.get(formula_id or "")
        if formula is not None:
            if formula.task_type != route_result.task_type:
                issues.append(f"formula_route_mismatch:{formula.formula_id}:{route_result.task_type}")
            if formula.formula_id != "multi_output_direct" and formula.principle_id != principle_id:
                issues.append(f"formula_principle_mismatch:{formula.formula_id}:{principle_id}")
            if formula.formula_id != "multi_output_direct" and getattr(solver_result, "unit", None) != formula.target_unit:
                issues.append(f"formula_unit_mismatch:{formula.formula_id}:{getattr(solver_result, 'unit', None)}")

    trace = {
        "stage": "schema_registry_validator",
        "quantity_count": len(front_payload.get("quantities", [])),
        "route_task_type": route_result.task_type,
        "solver_formula_id": getattr(solver_result, "formula_id", None) if solver_result is not None else None,
    }
    return RegistryValidationResult(not issues, issues, trace)
