"""Per-target deterministic orchestration for multi-output questions."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from .equation_engine import SolverResult, solve_fast
from .logic_engine import solve_conceptual
from .spatial_engine import solve_spatial_from_front
from ..knowledge.constraint_graph import route


def solve_multi_output(front_payload: dict) -> SolverResult:
    """Solve ordered targets as independent proof branches when safe."""

    if front_payload.get("answer_type_hint") != "multi_output":
        return _unsolved("not_multi_output")
    direct_spatial = _solve_direct_spatial_multi_output(front_payload)
    if direct_spatial is not None:
        return direct_spatial
    target_fragments = _target_fragments(front_payload)
    if len(target_fragments) < 2:
        return _unsolved("multi_output_targets_not_grounded", {"target_hints": front_payload.get("target_hints", [])})

    branch_results = []
    branch_traces = []
    premises = []
    for index, fragment in enumerate(target_fragments, start=1):
        branch_front = dict(front_payload)
        branch_front["answer_type_hint"] = "numeric"
        branch_front["target_hints"] = [fragment]
        branch_front["goals"] = [goal for goal in front_payload.get("goals", []) if str(goal.get("text") or "").lower() in fragment.lower()]
        branch_route = route(branch_front)
        if branch_route.task_type == "multi_output":
            branch_route = SimpleNamespace(task_type="unknown", answer_type="numeric", confidence=0.0, reasons=["nested_multi_output_rejected"])
        result = solve_conceptual(branch_front, branch_route)
        if not result.solved:
            result = solve_fast(branch_front, branch_route)
        if not result.solved:
            result = solve_spatial_from_front(branch_front, branch_route)
        branch_traces.append(
            {
                "target_index": index,
                "target_text": fragment,
                "route": branch_route.to_dict() if hasattr(branch_route, "to_dict") else dict(branch_route.__dict__),
                "solver": result.to_dict(),
            }
        )
        if not result.solved:
            return _unsolved(
                "multi_output_branch_failed",
                {"failed_index": index, "failed_target": fragment, "branches": branch_traces},
            )
        branch_results.append(result)
        premises.extend(item for item in result.premises if item not in premises)

    values = [
        {
            "target_index": index,
            "answer": result.answer,
            "value": result.value,
            "unit": result.unit,
            "formula_id": result.formula_id,
        }
        for index, result in enumerate(branch_results, start=1)
    ]
    answer = "; ".join(result.answer for result in branch_results)
    confidence = min(float(result.confidence) for result in branch_results)
    return SolverResult(
        solved=True,
        answer=answer,
        value=values,
        unit="-",
        formula_id="multi_output_direct",
        principle_id="conceptual_core",
        premises=premises,
        trace={
            "stage": "multi_output_orchestrator",
            "target_dimension": "multi_output",
            "expression": "ordered independent target proof branches",
            "branch_count": len(branch_results),
            "branches": branch_traces,
            "binding_audit": {"policy": "ordered_target_branches"},
        },
        confidence=min(0.7, confidence),
    )


def _target_fragments(front_payload: dict) -> list[str]:
    goals = [goal.get("text") for goal in front_payload.get("goals", []) if goal.get("text")]
    if len(goals) >= 2:
        return [str(goal) for goal in goals]
    hints = [str(hint) for hint in front_payload.get("target_hints", []) if hint]
    if len(hints) >= 2:
        return hints
    if not hints:
        return []
    text = hints[0]
    for separator in [" respectively ", ";", " and ", ","]:
        if separator in text:
            return [part.strip(" .?") for part in text.split(separator) if part.strip(" .?")]
    return hints


def _solve_direct_spatial_multi_output(front_payload: dict) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "electric field" not in text or "zero" not in text or "distance" not in text:
        return None
    branch_front = dict(front_payload)
    branch_front["answer_type_hint"] = "numeric"
    branch_front["target_hints"] = [", ".join(str(item) for item in front_payload.get("target_hints") or [])]
    branch_route = route(branch_front)
    result = solve_spatial_from_front(branch_front, branch_route)
    if not result.solved:
        return None
    return replace(
        result,
        formula_id=result.formula_id or "multi_output_direct",
        trace={
            **result.trace,
            "stage": "multi_output_orchestrator",
            "substage": result.trace.get("stage"),
            "binding_audit": {
                **(result.trace.get("binding_audit") or {}),
                "multi_output_policy": "single spatial zero-field construction satisfies point and distance targets",
            },
        },
        confidence=min(0.7, result.confidence),
    )


def _unsolved(reason: str, extra: dict | None = None) -> SolverResult:
    trace = {"stage": "multi_output_orchestrator", "reason": reason}
    if extra:
        trace.update(extra)
    return SolverResult(False, "", None, None, None, None, [], trace, 0.0)
