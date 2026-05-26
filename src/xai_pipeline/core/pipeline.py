"""NSP-Core deterministic pipeline composition."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict

from ..engines.algebraic_engine import solve_algebraic_plan
from ..engines.equation_engine import SolverResult, solve_fast
from ..engines.logic_engine import solve_conceptual
from ..engines.multi_output import solve_multi_output
from ..engines.spatial_engine import geometry_recoverability, solve_spatial_from_front
from ..frontend.semantic_parser import process_question_front
from ..knowledge.constraint_graph import build_registry_symbolic_plan, route, select_minimal_equation_subset
from ..knowledge.registries import FORMULA_REGISTRY, formula_family_for_id
from ..knowledge.units import convert_si_to_target, detect_requested_target_unit
from ..planning.local_llm import propose_solve_plan_if_enabled, refine_front_ir_if_enabled, repair_front_ir_once, repair_solve_plan_once
from ..planning.plan_compiler import CompiledPlan, build_plan_error_packet, compile_solve_plan
from ..planning.solve_plan import build_deterministic_solve_plan
from ..runtime.cache import get_verified_response, put_verified_response
from ..runtime.telemetry import build_pipeline_telemetry, persist_telemetry_event
from ..verification.answer_check import check_answer
from ..verification.verifier import verify_solver
from ..xai.explanation import build_explanation
from ..xai.trace import build_proof_dag


PLANNING_MODES = frozenset({"deterministic", "hybrid", "llm_required"})


@dataclass(frozen=True)
class Deadline:
    started_at: float
    timeout_seconds: float

    def expired(self) -> bool:
        if self.timeout_seconds < 0:
            return False
        return self.remaining_seconds() <= 0

    def remaining_seconds(self) -> float | None:
        if self.timeout_seconds < 0:
            return None
        return max(0.0, self.timeout_seconds - (time.monotonic() - self.started_at))

    def to_dict(self) -> dict:
        return {
            "timeout_seconds": self.timeout_seconds,
            "remaining_seconds": self.remaining_seconds(),
            "disabled": self.timeout_seconds < 0,
        }


def process_question(
    question: str,
    data_path: Path | None = None,
    enable_llm: bool = False,
    timeout_seconds: float = 55.0,
    planning_mode: str | None = None,
) -> Dict[str, Any]:
    """Run the core: semantic IR -> constraint graph -> equation engine -> verifier."""

    del data_path
    mode = _normalize_planning_mode(planning_mode, enable_llm=enable_llm)
    enable_llm = enable_llm or mode == "llm_required" or os.environ.get("XAI_ENABLE_LOCAL_LLM") == "1"
    deadline = Deadline(time.monotonic(), timeout_seconds)
    cached = get_verified_response(question, namespace=_cache_namespace(mode))
    if cached is not None:
        cached["cache"] = {"hit": True, "policy": "verified_and_answer_checked_only"}
        cached.setdefault("trace", {})["cache"] = {"checked": True, "hit": True}
        cached["trace"]["deadline"] = deadline.to_dict()
        cached.setdefault("metadata", {}).setdefault("planning_mode", mode)
        return cached

    if deadline.expired():
        return _timeout_response(deadline)

    front = process_question_front(question)
    front, refinement_trace = refine_front_ir_if_enabled(front, enable_llm=enable_llm)
    front.setdefault("trace", {})["local_llm_refinement"] = refinement_trace
    route_result = route(front)
    graph_selection = select_minimal_equation_subset(front, route_result)
    compiled_plan = _build_compiled_plan(front, route_result, graph_selection, enable_llm=enable_llm, planning_mode=mode)
    solver_result = _dispatch_solver(front, route_result, graph_selection, compiled_plan)
    solver_result = _attach_core_trace(front, solver_result, graph_selection, compiled_plan)
    verification = verify_solver(front, route_result, solver_result)
    if not verification.ok and _should_attempt_front_repair(front, verification, enable_llm=enable_llm):
        repaired_front, repair_trace = repair_front_ir_once(front, verification, enable_llm=enable_llm)
        solver_result = replace(solver_result, trace={**solver_result.trace, "local_llm_repair": repair_trace})
        if repair_trace.get("used"):
            front = repaired_front
            route_result = route(front)
            graph_selection = select_minimal_equation_subset(front, route_result)
            compiled_plan = _build_compiled_plan(front, route_result, graph_selection, enable_llm=enable_llm, planning_mode=mode)
            solver_result = _attach_core_trace(front, _dispatch_solver(front, route_result, graph_selection, compiled_plan), graph_selection, compiled_plan)
            verification = verify_solver(front, route_result, solver_result)
    elif not verification.ok:
        repair_trace = {
            "stage": "local_llm_repair",
            "used": False,
            "reason": "local_llm_disabled_or_front_repair_disabled",
            "policy": "default LLM budget is one structured solve-plan call; set XAI_LLM_ENABLE_FRONT_REPAIR=1 to spend one semantic repair call",
        }
        solver_result = replace(solver_result, trace={**solver_result.trace, "local_llm_repair": repair_trace})
    if not verification.ok and solver_result.solved:
        solver_result = _mark_unverified_candidate(solver_result, verification)

    if deadline.expired():
        return _timeout_response(deadline)

    unit_trace = {"stage": "units.target_conversion", "applied": False}
    if verification.ok:
        solver_result, unit_trace = _apply_requested_target_unit(front, solver_result)
        solver_result = _refresh_proof_dag(solver_result, graph_selection)
        answer = solver_result.answer
        explanation = build_explanation(solver_result)
        confidence = verification.confidence
    else:
        answer = "Uncertain"
        explanation = _fallback_explanation(front, route_result, solver_result, verification)
        confidence = 0.0

    answer_check = check_answer(answer, explanation, solver_result, verification)
    if verification.ok and not answer_check.ok:
        answer = "Uncertain"
        explanation = "The deterministic answer checker rejected the computed answer."
        confidence = 0.0

    telemetry = build_pipeline_telemetry(
        front=front,
        route_result=route_result,
        solver_result=solver_result,
        verification=verification,
        compiled_plan=compiled_plan,
        deadline=deadline,
    )
    telemetry_store = persist_telemetry_event(telemetry)

    response = _format_response(
        front=front,
        route_result=route_result,
        graph_selection=graph_selection,
        compiled_plan=compiled_plan,
        solver_result=solver_result,
        verification=verification,
        answer_check=answer_check,
        answer=answer,
        explanation=explanation,
        confidence=confidence,
        cache_hit=False,
        telemetry=telemetry,
        telemetry_store=telemetry_store,
        deadline=deadline,
        unit_trace=unit_trace,
        planning_mode=mode,
    )
    if verification.ok and answer_check.ok:
        put_verified_response(question, response, namespace=_cache_namespace(mode))
    return response


def _normalize_planning_mode(planning_mode: str | None, *, enable_llm: bool) -> str:
    raw = (planning_mode or os.environ.get("XAI_PLANNING_MODE") or "").strip().lower()
    if not raw:
        raw = "llm_required" if enable_llm or os.environ.get("XAI_ENABLE_LOCAL_LLM") == "1" else "deterministic"
    raw = raw.replace("-", "_")
    if raw not in PLANNING_MODES:
        return "deterministic"
    return raw


def _cache_namespace(planning_mode: str) -> str:
    return f"physics-xai:{planning_mode}"


def _build_compiled_plan(front: dict, route_result, graph_selection, enable_llm: bool = False, planning_mode: str = "deterministic") -> CompiledPlan:
    if planning_mode == "deterministic":
        deterministic_plan = build_deterministic_solve_plan(front, route_result, graph_selection).to_dict()
        compiled = compile_solve_plan(deterministic_plan, front, route_result, graph_selection)
        trace = dict(compiled.trace)
        trace["planning_mode"] = planning_mode
        trace["planning_authority"] = "deterministic_debug_fallback"
        return replace(compiled, trace=trace)

    llm_plan, llm_trace = propose_solve_plan_if_enabled(front, route_result, graph_selection, enable_llm=enable_llm)
    front.setdefault("trace", {})["local_llm_solve_plan"] = llm_trace
    if llm_plan:
        llm_compiled = compile_solve_plan(llm_plan, front, route_result, graph_selection)
        if llm_compiled.ok:
            return _mark_llm_plan_applied(llm_compiled, llm_trace, source="local_llm", planning_mode=planning_mode)
        error_packet = build_plan_error_packet(llm_compiled, front, route_result, graph_selection)
        repair_trace = _disabled_plan_repair_trace(error_packet)
        if _should_attempt_plan_repair(planning_mode):
            repaired_plan, repair_trace = repair_solve_plan_once(
                front,
                error_packet,
                route_result,
                graph_selection,
                enable_llm=enable_llm,
            )
            front.setdefault("trace", {})["local_llm_solve_plan_repair"] = repair_trace
            if repaired_plan:
                repaired_compiled = compile_solve_plan(repaired_plan, front, route_result, graph_selection)
                if repaired_compiled.ok:
                    trace = dict(repaired_compiled.trace)
                    trace["original_llm_plan_rejected"] = llm_compiled.to_dict()
                    trace["plan_repair_trace"] = {**repair_trace, "applied": True, "reason": "accepted_by_plan_compiler"}
                    return _mark_llm_plan_applied(repaired_compiled, repair_trace, source="local_llm_repair", extra_trace=trace, planning_mode=planning_mode)
                llm_compiled = _attach_repair_rejection(llm_compiled, repaired_compiled, repair_trace)
        else:
            front.setdefault("trace", {})["local_llm_solve_plan_repair"] = repair_trace
        if planning_mode == "llm_required":
            return _llm_required_rejection(
                front,
                route_result,
                graph_selection,
                reason="llm_plan_rejected_by_compiler",
                llm_trace=llm_trace,
                rejected_compiled=llm_compiled,
                error_packet=error_packet,
                repair_trace=front.get("trace", {}).get("local_llm_solve_plan_repair"),
            )
        deterministic_plan = build_deterministic_solve_plan(front, route_result, graph_selection).to_dict()
        deterministic_compiled = compile_solve_plan(deterministic_plan, front, route_result, graph_selection)
        trace = dict(deterministic_compiled.trace)
        trace["planning_mode"] = planning_mode
        trace["planning_authority"] = "deterministic_fallback_after_llm_rejection"
        trace["llm_plan_rejected"] = llm_compiled.to_dict()
        trace["llm_plan_trace"] = llm_trace
        trace["plan_error_packet"] = error_packet
        trace["plan_repair_trace"] = front.get("trace", {}).get("local_llm_solve_plan_repair")
        return replace(deterministic_compiled, trace=trace)

    if planning_mode == "llm_required":
        missing_compiled = compile_solve_plan(None, front, route_result, graph_selection)
        error_packet = build_plan_error_packet(missing_compiled, front, route_result, graph_selection)
        repair_trace = _disabled_plan_repair_trace(error_packet)
        if _should_attempt_plan_repair(planning_mode):
            repaired_plan, repair_trace = repair_solve_plan_once(
                front,
                error_packet,
                route_result,
                graph_selection,
                enable_llm=enable_llm,
            )
            front.setdefault("trace", {})["local_llm_solve_plan_repair"] = repair_trace
            if repaired_plan:
                repaired_compiled = compile_solve_plan(repaired_plan, front, route_result, graph_selection)
                if repaired_compiled.ok:
                    trace = dict(repaired_compiled.trace)
                    trace["original_llm_plan_missing"] = True
                    trace["plan_repair_trace"] = {**repair_trace, "applied": True, "reason": "accepted_by_plan_compiler"}
                    return _mark_llm_plan_applied(repaired_compiled, repair_trace, source="local_llm_repair", extra_trace=trace, planning_mode=planning_mode)
                missing_compiled = _attach_repair_rejection(missing_compiled, repaired_compiled, repair_trace)
        else:
            front.setdefault("trace", {})["local_llm_solve_plan_repair"] = repair_trace
        return _llm_required_rejection(
            front,
            route_result,
            graph_selection,
            reason="llm_plan_missing_or_invalid_json",
            llm_trace=llm_trace,
            rejected_compiled=missing_compiled,
            error_packet=error_packet,
            repair_trace=front.get("trace", {}).get("local_llm_solve_plan_repair"),
        )

    deterministic_plan = build_deterministic_solve_plan(front, route_result, graph_selection).to_dict()
    compiled = compile_solve_plan(deterministic_plan, front, route_result, graph_selection)
    trace = dict(compiled.trace)
    trace["planning_mode"] = planning_mode
    trace["planning_authority"] = "deterministic_fallback_after_missing_llm_plan"
    trace["llm_plan_trace"] = llm_trace
    return replace(compiled, trace=trace)


def _mark_llm_plan_applied(
    compiled: CompiledPlan,
    llm_trace: dict,
    source: str,
    extra_trace: dict | None = None,
    planning_mode: str = "hybrid",
) -> CompiledPlan:
    plan = dict(compiled.plan)
    plan["source"] = source
    trace = dict(extra_trace or compiled.trace)
    trace["planning_mode"] = planning_mode
    trace["planning_authority"] = "local_llm_structured_step_dag"
    trace["llm_plan_trace"] = {**llm_trace, "applied": True, "reason": "accepted_by_plan_compiler"}
    return replace(compiled, plan=plan, trace=trace)


def _attach_repair_rejection(original: CompiledPlan, repaired: CompiledPlan, repair_trace: dict) -> CompiledPlan:
    trace = dict(original.trace)
    trace["repair_plan_rejected"] = repaired.to_dict()
    trace["plan_repair_trace"] = repair_trace
    return replace(original, trace=trace)


def _llm_required_rejection(
    front: dict,
    route_result,
    graph_selection,
    *,
    reason: str,
    llm_trace: dict,
    rejected_compiled: CompiledPlan,
    error_packet: dict,
    repair_trace: dict | None,
) -> CompiledPlan:
    envelope = build_deterministic_solve_plan(front, route_result, graph_selection).to_dict()
    plan = {
        "status": "needs_fallback",
        "task_type": envelope.get("task_type"),
        "answer_type": envelope.get("answer_type"),
        "targets": envelope.get("targets") or [],
        "assumptions": envelope.get("assumptions") or [],
        "steps": [],
        "output_format": {
            "format_kind": "controlled_fallback",
            "ordered_targets": [
                target.get("id")
                for target in envelope.get("targets") or []
                if isinstance(target, dict) and target.get("id")
            ],
            "target_count": len(envelope.get("targets") or []),
        },
        "source": "local_llm_required",
        "notes": ["LLM-required planning mode abstained because no compiler-accepted LLM step DAG was available."],
    }
    return CompiledPlan(
        ok=False,
        plan=plan,
        selected_formula_ids=[],
        preferred_engine_order=["logic", "fast_formula", "spatial", "algebraic"],
        issues=[reason, *list(rejected_compiled.issues)],
        trace={
            "stage": "plan_compiler",
            "summary": {"present": True, "status": "needs_fallback", "source": "local_llm_required", "step_count": 0},
            "planning_mode": "llm_required",
            "planning_authority": "local_llm_required_fail_closed",
            "llm_plan_trace": {**llm_trace, "applied": False, "reason": reason},
            "llm_plan_rejected": rejected_compiled.to_dict(),
            "plan_error_packet": error_packet,
            "plan_repair_trace": repair_trace,
            "policy": "llm_required_mode_never_executes_deterministic_fallback_plan",
        },
    )


def _should_attempt_plan_repair(planning_mode: str = "hybrid") -> bool:
    if planning_mode == "llm_required":
        return True
    return os.environ.get("XAI_LLM_ENABLE_PLAN_REPAIR", "").strip().lower() in {"1", "true", "yes", "on"}


def _disabled_plan_repair_trace(error_packet: dict) -> dict:
    return {
        "stage": "local_llm_solve_plan_repair",
        "used": False,
        "applied": False,
        "reason": "plan_repair_disabled_by_default",
        "error_packet": error_packet,
        "policy": "default LLM budget is one structured solve-plan call; set XAI_LLM_ENABLE_PLAN_REPAIR=1 for one repair call",
    }


def _dispatch_solver(front: dict, route_result, graph_selection, compiled_plan: CompiledPlan | None = None) -> SolverResult:
    if compiled_plan is not None and not compiled_plan.ok:
        reason = "unsupported_plan" if any(str(issue).startswith("plan_status:unsupported") for issue in compiled_plan.issues) else "plan_compilation_failed"
        return SolverResult(
            False,
            "",
            None,
            None,
            None,
            None,
            [],
            {
                "stage": "plan_compiler",
                "reason": reason,
                "compiled_plan": compiled_plan.to_dict(),
            },
            0.0,
        )
    order = list(compiled_plan.preferred_engine_order if compiled_plan and compiled_plan.ok else [])
    if not order:
        order = ["multi_output", "logic", "fast_formula", "spatial", "algebraic"]

    attempts: dict[str, dict] = {}
    attempt_results: dict[str, SolverResult] = {}
    spatial_required = _compiled_plan_requires_spatial(compiled_plan)
    for engine in order:
        if engine == "multi_output":
            if route_result.task_type != "multi_output" and front.get("answer_type_hint") != "multi_output":
                continue
            result = solve_multi_output(front)
        elif engine == "logic":
            result = solve_conceptual(front, route_result)
        elif engine == "fast_formula":
            result = solve_fast(
                front,
                route_result,
                allowed_formula_ids=compiled_plan.selected_formula_ids if compiled_plan and compiled_plan.ok else None,
            )
        elif engine == "spatial":
            result = solve_spatial_from_front(front, route_result)
        elif engine == "algebraic":
            symbolic_plan = build_registry_symbolic_plan(front, route_result, graph_selection)
            result = solve_algebraic_plan(symbolic_plan, route_result)
        else:
            continue
        if result.solved:
            off_plan_issue = _off_plan_solver_result_issue(result, compiled_plan)
            if off_plan_issue:
                result = _reject_off_plan_solver_result(result, off_plan_issue, compiled_plan)
                attempts[engine] = result.trace
                attempt_results[engine] = result
                if engine == "spatial" and spatial_required:
                    trace = dict(result.trace)
                    trace["engine_attempts"] = attempts
                    trace["solver_dispatch_order"] = order
                    trace["policy"] = "required_spatial_plan_failed_or_off_plan"
                    return replace(result, trace=trace)
                continue
        attempts[engine] = result.trace
        attempt_results[engine] = result
        if result.solved:
            return result
        if engine == "spatial" and spatial_required:
            trace = dict(result.trace)
            trace["engine_attempts"] = attempts
            trace["solver_dispatch_order"] = order
            trace["reason"] = trace.get("reason") or "required_spatial_plan_failed"
            trace["policy"] = "scalar_formula_fallback_blocked_for_geometry_plan"
            return replace(result, trace=trace)

    base = _best_failed_attempt(attempt_results) or SolverResult(False, "", None, None, None, None, [], {"stage": "solver_dispatch", "reason": "no_engine_attempted"}, 0.0)
    trace = dict(base.trace)
    trace["engine_attempts"] = attempts
    trace["solver_dispatch_order"] = order
    return replace(base, trace=trace)


def _off_plan_solver_result_issue(result: SolverResult, compiled_plan: CompiledPlan | None) -> str | None:
    """Return a governance issue when an engine solved outside the compiled plan.

    Some deterministic engines contain internal specialized branches for broad
    physical families. That is fine only when the branch corresponds to the
    compiler-accepted formula set. This gate keeps execution plan-owned in
    `llm_required` mode and also protects deterministic debug plans from stale
    solver shortcuts.
    """

    if compiled_plan is None or not compiled_plan.ok:
        return None
    if result.formula_id in {"conceptual_direct", "yes_no_direct"}:
        operations = {
            step.get("operation")
            for step in (compiled_plan.plan or {}).get("steps") or []
            if isinstance(step, dict)
        }
        if operations & {"apply_logic_rule", "check_condition"}:
            return None
    selected = {str(formula_id) for formula_id in compiled_plan.selected_formula_ids if formula_id}
    if not selected or not result.formula_id:
        return None
    if result.formula_id in selected:
        return None
    if any(_same_formula_family(result.formula_id, selected_id) for selected_id in selected):
        return None
    return f"solver_formula_not_in_compiled_plan:{result.formula_id}"


def _same_formula_family(formula_id: str, selected_formula_id: str) -> bool:
    family = formula_family_for_id(formula_id)
    selected_family = formula_family_for_id(selected_formula_id)
    return family is not None and family == selected_family


def _reject_off_plan_solver_result(result: SolverResult, issue: str, compiled_plan: CompiledPlan | None) -> SolverResult:
    trace = dict(result.trace)
    trace["off_plan_result_rejected"] = {
        "issue": issue,
        "solver_formula_id": result.formula_id,
        "compiled_formula_ids": list(compiled_plan.selected_formula_ids if compiled_plan else []),
        "policy": "executor_result_must_match_compiler_selected_registry_path",
    }
    return replace(
        result,
        solved=False,
        answer="",
        value=None,
        unit=None,
        trace=trace,
        confidence=0.0,
    )


def _compiled_plan_requires_spatial(compiled_plan: CompiledPlan | None) -> bool:
    if compiled_plan is None or not compiled_plan.ok:
        return False
    plan = compiled_plan.plan or {}
    operations = {
        step.get("operation")
        for step in plan.get("steps") or []
        if isinstance(step, dict) and step.get("operation")
    }
    return bool(operations & {"construct_geometry", "compute_pairwise_force", "resolve_vector_components", "vector_sum"})


def _best_failed_attempt(attempt_results: dict[str, SolverResult]) -> SolverResult | None:
    for engine in ("fast_formula", "spatial", "algebraic", "multi_output", "logic"):
        if engine in attempt_results:
            return attempt_results[engine]
    return next(iter(attempt_results.values()), None)


def _attach_core_trace(front: dict, solver_result: SolverResult, graph_selection, compiled_plan: CompiledPlan | None = None) -> SolverResult:
    trace = dict(solver_result.trace)
    trace["constraint_graph"] = graph_selection.to_dict()
    if compiled_plan is not None:
        trace["structured_solve_plan"] = compiled_plan.to_dict()
    if _should_attach_geometry_recoverability(front, solver_result):
        trace["geometry"] = geometry_recoverability(front)
    attached = replace(solver_result, trace=trace)
    trace["proof_dag"] = _proof_dag(attached, graph_selection)
    return replace(attached, trace=trace)


def _should_attempt_front_repair(front: dict, verification, *, enable_llm: bool) -> bool:
    if not enable_llm:
        return False
    if os.environ.get("XAI_LLM_ENABLE_FRONT_REPAIR", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if os.environ.get("XAI_LLM_ENABLE_SEMANTIC_AUDIT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    trace = front.setdefault("trace", {})
    trace["local_llm_repair_gate"] = {
        "enabled": False,
        "reason": "front_repair_opt_in_not_enabled",
        "policy": "default LLM budget is one structured solve-plan call; set XAI_LLM_ENABLE_FRONT_REPAIR=1 for semantic repair",
        "issues": list(getattr(verification, "issues", [])),
    }
    return False


def _should_attach_geometry_recoverability(front: dict, solver_result: SolverResult) -> bool:
    if solver_result.trace.get("geometry") is not None:
        return False
    if solver_result.trace.get("geometry_engine") is not None or solver_result.trace.get("geometry_audit") is not None:
        return True
    if any(isinstance(relation, dict) and relation.get("relation_type") == "geometry" for relation in front.get("relations") or []):
        return True
    geometry_words = ("point", "vertex", "triangle", "midpoint", "bisector", "center")
    for goal in front.get("goals") or []:
        if not isinstance(goal, dict):
            continue
        text = str(goal.get("text") or "").lower()
        if any(word in text for word in geometry_words):
            return True
    if not solver_result.solved and front.get("answer_type_hint") == "symbolic":
        return True
    return False


def _proof_dag(solver_result: SolverResult, graph_selection=None) -> dict:
    return build_proof_dag(solver_result, graph_selection)


def _refresh_proof_dag(solver_result: SolverResult, graph_selection) -> SolverResult:
    trace = dict(solver_result.trace)
    refreshed = replace(solver_result, trace=trace)
    trace["proof_dag"] = build_proof_dag(refreshed, graph_selection)
    return replace(refreshed, trace=trace)


def _mark_unverified_candidate(solver_result: SolverResult, verification) -> SolverResult:
    """Expose rejected executor output as an audit artifact, not as a solved result."""

    trace = dict(solver_result.trace)
    trace["unverified_candidate"] = solver_result.to_dict()
    trace["candidate_proof_dag"] = trace.get("proof_dag", {})
    trace["proof_dag"] = build_proof_dag(
        replace(
            solver_result,
            solved=False,
            trace={"reason": "verifier_rejected_candidate", "verifier_issues": list(verification.issues)},
        )
    )
    trace["verifier_rejection"] = verification.to_dict()
    return replace(
        solver_result,
        solved=False,
        answer="",
        value=None,
        unit=None,
        trace=trace,
        confidence=0.0,
    )


def _version_metadata() -> dict:
    return {
        "semantic_parser_version": "semantic-parser-v4-qualitative-change-and-contextual-hidden-si",
        "canonical_structure_version": "canonical-structure-v2-local-right-triangle-frame",
        "logic_engine_version": "logic-engine-v4-factorized-proportional-reasoning",
        "constraint_graph_version": "constraint-graph-v5-branch-pruned-structural-multiplicity",
        "formula_catalog_version": "physics-problems-text-only-v1-registry-synthesis",
        "equation_engine_version": "equation-engine-v4-cas-lite-topology",
        "spatial_engine_version": "spatial-engine-v6-role-grounded-line-and-triangle-dispatch",
        "multi_output_version": "multi-output-v1-ordered-branches",
        "solve_plan_version": "solve-plan-v4-graph-selected-spatial-branching",
        "plan_compiler_version": "plan-compiler-v3-formula-branch-dispatch",
        "local_llm_boundary_version": "local-llm-boundary-v2-structured-public-cot",
        "verifier_version": "verifier-v4-residual-domain-uncertainty",
        "proof_dag_version": "proof-dag-v3-structural-constraint-certificate",
        "explanation_version": "trace-explanation-v2-structural-proof",
        "unit_registry_version": "unit-registry-v1",
    }


def _apply_requested_target_unit(front: dict, solver_result: SolverResult) -> tuple[SolverResult, dict]:
    if not solver_result.solved or solver_result.value is None or isinstance(solver_result.value, (str, list)):
        return solver_result, {"stage": "units.target_conversion", "applied": False, "reason": "unsupported_value_shape"}
    formula = FORMULA_REGISTRY.get(solver_result.formula_id or "")
    target_dimension = (formula.target_dimension if formula else None) or solver_result.trace.get("target_dimension")
    requested = detect_requested_target_unit(front, str(target_dimension or ""))
    if not requested:
        return solver_result, {"stage": "units.target_conversion", "applied": False, "reason": "no_requested_target_unit"}
    converted = convert_si_to_target(float(solver_result.value), str(target_dimension), requested)
    if not converted.ok:
        return solver_result, {"stage": "units.target_conversion", "applied": False, "issues": converted.issues, "requested_unit": requested}
    answer = f"{converted.value:.6g} {converted.unit}".strip()
    trace = dict(solver_result.trace)
    trace["target_unit_conversion"] = converted.to_dict()
    return (
        replace(
            solver_result,
            answer=answer,
            value=converted.value,
            unit=converted.unit,
            trace=trace,
        ),
        {"stage": "units.target_conversion", "applied": True, "requested_unit": requested, "converted": converted.to_dict()},
    )


def _format_response(
    *,
    front: dict,
    route_result,
    graph_selection,
    compiled_plan: CompiledPlan,
    solver_result: SolverResult,
    verification,
    answer_check,
    answer: str,
    explanation: str,
    confidence: float,
    cache_hit: bool,
    telemetry: dict,
    telemetry_store: dict,
    deadline: Deadline,
    unit_trace: dict,
    planning_mode: str,
) -> Dict[str, Any]:
    return {
        "answer": answer,
        "explanation": explanation,
        "cot": _public_steps(solver_result, verification),
        "premises": list(solver_result.premises),
        "confidence": confidence,
        "metadata": {
            "status": "ok" if verification.ok and answer_check.ok else _fallback_status(front, solver_result, verification),
            "planning_mode": planning_mode,
            "versions": _version_metadata(),
            "xai_policy": {
                "explanation_source": "proof_dag_and_execution_trace",
                "reasoning_style": "structural_public_trace",
                "planning_authority": _planning_authority(compiled_plan),
                "llm_free_form_reasoning_used": False,
            },
        },
        "front": front,
        "route": route_result.to_dict(),
        "solve_plan": compiled_plan.to_dict(),
        "constraint_graph": graph_selection.to_dict(),
        "solver": solver_result.to_dict(),
        "verifier": verification.to_dict(),
        "answer_checker": answer_check.to_dict(),
        "cache": {"hit": cache_hit, "policy": "verified_and_answer_checked_only"},
        "telemetry": telemetry,
        "trace": {
            "stages": [
                "api",
                "cache",
                "semantic_parser",
                "solve_plan",
                "plan_compiler",
                "logic_engine",
                "constraint_graph",
                "equation_engine",
                "spatial_engine",
                "verifier",
                "explanation",
                "answer_check",
                "response",
            ],
            "cache": {"checked": True, "hit": cache_hit},
            "deadline": deadline.to_dict(),
            "planning_mode": planning_mode,
            "structured_solve_plan": compiled_plan.to_dict(),
            "target_unit_conversion": unit_trace,
            "proof_dag": solver_result.trace.get("proof_dag", {}),
            "telemetry_store": telemetry_store,
        },
    }


def _planning_authority(compiled_plan: CompiledPlan) -> str:
    trace = compiled_plan.trace or {}
    return str(trace.get("planning_authority") or "unknown")


def _public_steps(solver_result: SolverResult, verification) -> list[str]:
    if not verification.ok:
        reason = str((solver_result.trace or {}).get("reason") or "")
        if _trace_is_llm_required_plan_failure(solver_result.trace):
            return [
                "Parse and normalize semantic facts deterministically.",
                "Ask the local LLM for a structured executable solve plan.",
                "Abstain because no compiler-accepted LLM step DAG was available.",
            ]
        if reason == "multicharge_force_goal_requires_spatial_grounding":
            return [
                "Parse the question into semantic facts.",
                "Detect a multi-charge force target.",
                "Reject scalar Coulomb execution because the target point is not spatially grounded.",
            ]
        if reason == "no_registry_formula_executed":
            return [
                "Parse the question into semantic facts.",
                "Build a registry-backed solve plan.",
                "Abstain because no executable, target-compatible constraint path was verified.",
            ]
        return ["Parse the question into semantic facts.", "No verified constraint path was found."]
    plan_steps = _accepted_public_cot_steps(solver_result)
    proof_audit = (solver_result.trace.get("proof_dag") or {}).get("audit") or {}
    constraint_id = proof_audit.get("constraint_id") or "registry constraint"
    if plan_steps:
        fact_phrase = "accepted symbolic facts" if isinstance(solver_result.value, str) else "SI-normalized facts"
        return [
            "Build a typed semantic IR from quantities, entities, constraints, and goals.",
            *plan_steps,
            f"Execute the compiled registry path through `{constraint_id}` with {fact_phrase}.",
            "Verify the proof graph, unit compatibility, and final answer string.",
        ]
    return [
        "Build a typed semantic IR from quantities, entities, constraints, and goals.",
        "Compile a schema-bound public solve plan into registry-backed executable steps.",
        f"Select `{constraint_id}` through deterministic constraint-graph reachability.",
        "Execute the selected constraint with SI-normalized facts, not dataset examples.",
        "Verify the proof graph, unit compatibility, and final answer string.",
    ]


def _accepted_public_cot_steps(solver_result: SolverResult) -> list[str]:
    compiled = solver_result.trace.get("structured_solve_plan") if isinstance(solver_result.trace, dict) else None
    if not isinstance(compiled, dict) or not compiled.get("ok"):
        return []
    plan = compiled.get("plan") or {}
    out: list[str] = []
    for step in plan.get("steps") or []:
        if not isinstance(step, dict):
            continue
        public_cot = step.get("public_cot")
        if isinstance(public_cot, str) and public_cot.strip():
            out.append(public_cot.strip())
    return out


def _fallback_status(front: dict, solver_result: SolverResult, verification) -> str:
    reason = str((solver_result.trace or {}).get("reason") or "")
    if _trace_is_llm_required_plan_failure(solver_result.trace):
        return "llm_plan_unavailable"
    if reason == "multicharge_force_goal_requires_spatial_grounding":
        return "underspecified_geometry"
    if reason == "no_registry_formula_executed" and not front.get("target_hints"):
        return "missing_explicit_target"
    return "unsupported_or_unverified"


def _fallback_explanation(front: dict, route_result, solver_result: SolverResult, verification) -> str:
    reason = str((solver_result.trace or {}).get("reason") or "")
    if _trace_is_llm_required_plan_failure(solver_result.trace):
        return (
            "LLM-required planning mode was active, but the local LLM did not produce a compiler-accepted structured step DAG. "
            "NSP-Core abstained instead of executing a deterministic fallback plan behind the model."
        )
    if reason == "multicharge_force_goal_requires_spatial_grounding":
        return (
            "The semantic frontend detected a multi-charge force target, but the question does not provide enough spatial grounding "
            "for the target charge point, such as coordinates, distances from that point to each source charge, or a valid geometry "
            "constructor. NSP-Core abstained instead of binding a scalar Coulomb formula to an arbitrary distance."
        )
    if reason == "no_registry_formula_executed" and not front.get("target_hints"):
        return (
            "The question states physics conditions but does not contain an explicit requested target. NSP-Core requires a grounded goal "
            "before executing registry equations, so it abstained instead of inferring a target from the dataset answer."
        )
    if reason == "no_registry_formula_executed":
        return (
            "The semantic frontend and constraint graph found the physics domain, but no validated registry-backed execution path matched "
            "the requested target and available facts. The system returned Uncertain rather than using an unverified formula binding."
        )
    issues = ", ".join(getattr(verification, "issues", []) or ["solver_not_solved"])
    return f"The deterministic solver could not verify an answer for this question. Verification issues: {issues}."


def _trace_is_llm_required_plan_failure(trace: dict | None) -> bool:
    if not isinstance(trace, dict):
        return False
    compiled = trace.get("compiled_plan") or trace.get("structured_solve_plan") or {}
    compiled_trace = compiled.get("trace") if isinstance(compiled, dict) else {}
    return isinstance(compiled_trace, dict) and compiled_trace.get("planning_authority") == "local_llm_required_fail_closed"


def _timeout_response(deadline: Deadline) -> Dict[str, Any]:
    return {
        "answer": "Uncertain",
        "explanation": "The problem exceeded the current Physics solving time budget.",
        "confidence": 0.0,
        "metadata": {"status": "timeout", "timeout_seconds": deadline.timeout_seconds},
        "front": {},
        "route": {"task_type": "unknown", "answer_type": "unknown", "confidence": 0.0, "reasons": []},
        "constraint_graph": {"ok": False, "formula_ids": [], "issues": ["deadline_expired"], "trace": {}},
        "solver": {
            "solved": False,
            "answer": "",
            "value": None,
            "unit": None,
            "formula_id": None,
            "principle_id": None,
            "premises": [],
            "trace": {"reason": "deadline_expired"},
            "confidence": 0.0,
        },
        "verifier": {"ok": False, "confidence": 0.0, "issues": ["deadline_expired"]},
        "answer_checker": {"ok": True, "issues": [], "trace": {"stage": "answer_checker", "mode": "controlled_fallback"}},
        "cache": {"hit": False, "policy": "verified_and_answer_checked_only"},
        "telemetry": {},
        "trace": {"stages": ["api", "deadline", "controlled_timeout_response"], "deadline": deadline.to_dict()},
    }
