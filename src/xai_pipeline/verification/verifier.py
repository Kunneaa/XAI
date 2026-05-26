"""Deterministic verification for NSP-Core solver outputs."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..engines.equation_engine import _safe_eval, _sanitize_equation, solve_candidate_paths
from ..engines.logic_engine import allowed_implicit_rule_ids
from ..knowledge.language import has_change_factor_cue
from ..knowledge.registries import (
    ANSWER_TYPES,
    FORMULA_IDS,
    FORMULA_REGISTRY,
    GEOMETRY_TEMPLATE_IDS,
    PROPOSAL_STATUSES,
    PRINCIPLE_IDS,
    SOLVE_STRATEGIES,
    TASK_TYPES,
    formula_family_for_id,
)
from ..knowledge.units import unit_info


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    confidence: float
    issues: List[str]
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "ok": self.ok,
            "confidence": self.confidence,
            "issues": list(self.issues),
            "conflicts": list(self.conflicts),
            "audit": dict(self.audit),
        }


def verify_solver(front_payload: dict, route_result, solver_result, registry_validation=None, unit_conversion=None) -> VerificationResult:
    """Accept only finite, registry-owned, unit-compatible solver output."""

    issues: List[str] = []
    conflicts = _detect_quantity_contradictions(front_payload)
    issues.extend(conflict["issue"] for conflict in conflicts)
    audit: Dict[str, Any] = {}
    if registry_validation is not None and getattr(registry_validation, "ok", True) is not True:
        issues.extend(f"registry:{issue}" for issue in getattr(registry_validation, "issues", []))
    if unit_conversion is not None and getattr(unit_conversion, "ok", True) is not True:
        issues.extend(f"unit_conversion:{issue}" for issue in getattr(unit_conversion, "issues", []))
    plan_audit = _compiled_plan_audit(solver_result)
    if plan_audit:
        audit["structured_solve_plan"] = plan_audit
        if plan_audit.get("status") == "rejected":
            issues.extend(plan_audit.get("issues", []))
    if not solver_result.solved:
        issues.append("solver_not_solved")
        return VerificationResult(False, 0.0, issues, conflicts, audit)

    if solver_result.formula_id not in FORMULA_IDS:
        issues.append("unknown_formula_id")
    if plan_audit.get("status") == "accepted" and solver_result.formula_id:
        selected = set(plan_audit.get("selected_formula_ids") or [])
        if selected and solver_result.formula_id not in selected and not _same_compiled_formula_family(solver_result.formula_id, selected):
            issues.append("solver_formula_not_in_compiled_plan")
    formula = FORMULA_REGISTRY.get(solver_result.formula_id or "")
    if (
        formula
        and solver_result.formula_id != "multi_output_direct"
        and formula.target_unit != "dynamic"
        and solver_result.unit != formula.target_unit
    ):
        issues.append("target_unit_mismatch")

    answer_type = front_payload.get("answer_type_hint")
    is_conceptual_result = isinstance(solver_result.value, str) or solver_result.formula_id in {"conceptual_direct", "yes_no_direct"}
    if is_conceptual_result:
        if not isinstance(solver_result.value, str) or not solver_result.answer:
            issues.append("conceptual_answer_missing")
        if solver_result.principle_id not in PRINCIPLE_IDS:
            issues.append("conceptual_answer_without_principle")
        if answer_type == "yes_no" and solver_result.answer not in {"Yes", "No"}:
            issues.append("yes_no_answer_not_normalized")
    elif solver_result.value is None:
        issues.append("non_finite_value")
    elif isinstance(solver_result.value, list):
        for item in solver_result.value:
            if not math.isfinite(float(item.get("value", float("nan")))):
                issues.append("non_finite_value")
                break
    elif not math.isfinite(float(solver_result.value)):
        issues.append("non_finite_value")

    _verify_vector_trace_consistency(solver_result, issues)
    residual_audit = _residual_verification(solver_result)
    if residual_audit:
        audit["residual"] = residual_audit
        if residual_audit.get("status") == "failed":
            issues.append("residual_verification_failed")
    domain_audit = _physical_domain_verification(solver_result)
    if domain_audit:
        audit["physical_domain"] = domain_audit
        issues.extend(domain_audit.get("issues", []))
    topology_audit = _topology_formula_audit(front_payload, solver_result)
    if topology_audit:
        audit["topology"] = topology_audit
        if topology_audit.get("status") == "rejected":
            issues.append(topology_audit["issue"])

    multi_path = _multi_path_verification(front_payload, route_result, solver_result)
    if multi_path:
        audit["multi_path"] = multi_path
        if multi_path.get("status") == "disagreement":
            issues.append("multi_path_disagreement")

    if not solver_result.answer:
        issues.append("answer_missing_unit")
    elif is_conceptual_result:
        pass
    elif isinstance(solver_result.value, list):
        for unit in [item.get("unit") for item in solver_result.value if item.get("unit") and item.get("unit") != "-"]:
            if unit not in solver_result.answer:
                issues.append("answer_missing_unit")
                break
    elif solver_result.unit not in solver_result.answer:
        issues.append("answer_missing_unit")
    if route_result.task_type == "unknown":
        issues.append("unknown_route")

    confidence = min(float(route_result.confidence), float(solver_result.confidence))
    if solver_result.trace.get("geometry", {}).get("recoverable"):
        confidence = min(confidence, 0.7)
    if _uses_inferred_si_units(front_payload):
        confidence = min(confidence, 0.82)
    if issues:
        confidence = min(confidence, 0.45)
    audit["uncertainty"] = _uncertainty_graph(front_payload, route_result, solver_result, confidence, issues)
    return VerificationResult(not issues, confidence if not issues else 0.0, issues, conflicts, audit)


def _topology_formula_audit(front_payload: dict, solver_result) -> Dict[str, Any]:
    topology = front_payload.get("topology_graph") or {}
    if not topology.get("is_complex"):
        return {}
    if solver_result.formula_id in {"current_sum", "current_branch_difference", "power_sum"}:
        return {
            "status": "accepted",
            "canonical_form": topology.get("canonical_form"),
            "formula_id": solver_result.formula_id,
            "reason": "branch conservation formula uses explicit current/power facts rather than unresolved component topology",
        }
    risky_principles = {"dc_circuit_core"}
    if solver_result.principle_id not in risky_principles:
        return {}
    binding_audit = solver_result.trace.get("binding_audit") if hasattr(solver_result, "trace") else None
    exact_or_single = True
    if isinstance(binding_audit, dict):
        for payload in binding_audit.values():
            if not isinstance(payload, dict):
                continue
            if payload.get("policy") not in {"exact_symbol_match", "single_candidate_dimension"}:
                exact_or_single = False
                break
    ambiguity = list(topology.get("ambiguity") or [])
    if topology.get("canonical_form") == "complex_circuit_topology_unresolved" or ambiguity:
        return {
            "status": "rejected",
            "issue": "topology_not_canonicalized_for_direct_formula",
            "canonical_form": topology.get("canonical_form"),
            "ambiguity": ambiguity,
            "formula_id": solver_result.formula_id,
            "principle_id": solver_result.principle_id,
            "binding_is_exact_or_single": exact_or_single,
        }
    return {
        "status": "accepted",
        "canonical_form": topology.get("canonical_form"),
        "formula_id": solver_result.formula_id,
    }


def validate_plan(plan: dict, front_payload: dict) -> VerificationResult:
    """Validate optional neural semantic proposals without importing LLM modules."""

    if isinstance(plan, dict) and "steps" in plan:
        from ..planning.plan_compiler import validate_structured_solve_plan

        result = validate_structured_solve_plan(plan, front_payload)
        return VerificationResult(result.ok, 0.85 if result.ok else 0.0, result.issues, audit=result.audit)

    issues: List[str] = []
    required = {
        "status",
        "task_type",
        "answer_type",
        "targets",
        "formula_ids",
        "principle_ids",
        "geometry_template_ids",
        "implicit_rule_ids",
        "solve_strategy",
    }
    if not isinstance(plan, dict):
        return VerificationResult(False, 0.0, ["proposal_not_object"])
    for field in sorted(required):
        if field not in plan:
            issues.append(f"missing_field:{field}")
    allowed_implicit_ids = set(allowed_implicit_rule_ids())
    if plan.get("status") not in PROPOSAL_STATUSES:
        issues.append("unknown_status")
    if plan.get("task_type") not in TASK_TYPES:
        issues.append("unknown_task_type")
    if plan.get("answer_type") not in ANSWER_TYPES:
        issues.append("unknown_answer_type")
    if plan.get("solve_strategy") not in SOLVE_STRATEGIES:
        issues.append("unknown_solve_strategy")
    if not isinstance(plan.get("targets"), list) or not plan.get("targets"):
        issues.append("empty_targets")
    for formula_id in plan.get("formula_ids", []):
        if formula_id not in FORMULA_IDS:
            issues.append(f"unknown_formula_id:{formula_id}")
    for principle_id in plan.get("principle_ids", []):
        if principle_id not in PRINCIPLE_IDS:
            issues.append(f"unknown_principle_id:{principle_id}")
    for template_id in plan.get("geometry_template_ids", []):
        if template_id not in GEOMETRY_TEMPLATE_IDS:
            issues.append(f"unknown_geometry_template_id:{template_id}")
    for rule_id in plan.get("implicit_rule_ids", []):
        if rule_id not in allowed_implicit_ids:
            issues.append(f"unknown_implicit_rule_id:{rule_id}")
        elif front_payload and rule_id not in {fact.get("rule_id") for fact in front_payload.get("implicit_facts", [])}:
            issues.append(f"implicit_rule_not_triggered:{rule_id}")
    for target in plan.get("targets", []):
        if not isinstance(target, dict) or not target.get("symbol"):
            issues.append("invalid_target")
    if plan.get("formula_ids") and plan.get("task_type") in TASK_TYPES:
        task_type = plan.get("task_type")
        available_dimensions = [quantity.get("dimension") for quantity in (front_payload or {}).get("quantities", [])]
        for formula_id in plan.get("formula_ids", []):
            formula = FORMULA_REGISTRY.get(formula_id)
            if formula is not None and formula.task_type != task_type:
                issues.append(f"formula_task_mismatch:{formula_id}:{task_type}")
            if formula is not None and formula.required_dimensions:
                missing = _missing_required_dimensions(available_dimensions, formula.required_dimensions)
                if missing:
                    issues.append(f"formula_missing_required_dimensions:{formula_id}:{','.join(missing)}")
    if plan.get("numeric_answer") is not None:
        issues.append("proposal_supplied_numeric_answer")
    if plan.get("answer_type") == "numeric" and plan.get("conceptual_answer"):
        issues.append("numeric_task_with_conceptual_answer")
    if plan.get("answer_type") in {"conceptual", "yes_no"} and plan.get("conceptual_answer") and not plan.get("principle_ids"):
        issues.append("conceptual_answer_without_principle")
    return VerificationResult(not issues, 0.8 if not issues else 0.0, issues)


def _compiled_plan_audit(solver_result) -> Dict[str, Any]:
    trace = solver_result.trace if hasattr(solver_result, "trace") else {}
    compiled = trace.get("structured_solve_plan")
    if not isinstance(compiled, dict):
        return {}
    if not compiled.get("ok"):
        return {"status": "rejected", "issues": [f"compiled_plan:{issue}" for issue in compiled.get("issues", [])]}
    return {
        "status": "accepted",
        "source": (compiled.get("plan") or {}).get("source"),
        "step_count": len((compiled.get("plan") or {}).get("steps") or []),
        "selected_formula_ids": list(compiled.get("selected_formula_ids") or []),
        "engine_order": list(compiled.get("preferred_engine_order") or []),
    }


def _same_compiled_formula_family(formula_id: str, selected_formula_ids: set[str]) -> bool:
    """Allow deterministic lowering to a sibling registry card in the same family."""

    compatibility_groups = (
        {"lc_energy_complement", "inductor_energy", "capacitor_energy_voltage", "capacitor_energy_charge"},
        {"capacitor_energy_voltage_percent", "capacitor_energy_voltage", "capacitor_voltage_charge", "capacitor_voltage_energy"},
        {
            "rlc_resonance_reactance_from_current_ratio",
            "rlc_resonance_frequency_multiplier",
            "rlc_resonance_resistance_from_impedance",
            "rlc_resonance_resistor_voltage",
            "rlc_component_voltage",
            "rlc_impedance",
            "rlc_impedance_from_rlcf",
            "rlc_current_from_rlcf_voltage",
            "ohm_current",
            "ohm_voltage",
            "ohm_resistance",
            "inductive_reactance",
            "capacitive_reactance",
            "ac_source_rms_from_sinusoid",
            "ac_source_angular_frequency",
        },
        {
            "rlc_quadrature_unknown_resistance",
            "rlc_quadrature_segment_current",
            "rlc_quadrature_segment_power",
            "rlc_quadrature_power_transfer_same_voltage",
            "rlc_quadrature_segment_voltage",
            "ohm_current",
            "ohm_voltage",
            "ohm_resistance",
            "power_ui",
            "power_i2r",
            "power_u2r",
        },
        {
            "magnetic_flux",
            "solenoid_flux_one_turn",
            "magnetic_flux_linkage",
            "solenoid_magnetic_field",
            "solenoid_magnetic_field_turns_length",
        },
    )
    for group in compatibility_groups:
        if formula_id in group and selected_formula_ids & group:
            return True
    family_id = formula_family_for_id(formula_id)
    if family_id is None:
        return False
    return any(formula_family_for_id(selected_id) == family_id for selected_id in selected_formula_ids)


def _verify_vector_trace_consistency(solver_result, issues: List[str]) -> None:
    geometry_engine = solver_result.trace.get("geometry_engine") if hasattr(solver_result, "trace") else None
    if not isinstance(geometry_engine, dict):
        return
    vector = geometry_engine.get("vector") or geometry_engine.get("components")
    value = geometry_engine.get("value")
    if value is None and isinstance(vector, dict):
        value = vector.get("magnitude")
    if not isinstance(vector, dict) or value is None:
        return
    try:
        x = float(vector.get("x", 0.0))
        y = float(vector.get("y", 0.0))
        magnitude = float(vector.get("magnitude", math.hypot(x, y)))
        reported = float(value)
        solver_value = float(solver_result.value) if solver_result.value is not None and not isinstance(solver_result.value, (str, list)) else reported
    except (TypeError, ValueError):
        issues.append("vector_trace_not_numeric")
        return
    if not math.isclose(math.hypot(x, y), magnitude, rel_tol=1e-6, abs_tol=1e-9):
        issues.append("vector_component_magnitude_mismatch")
    if not math.isclose(reported, solver_value, rel_tol=1e-6, abs_tol=1e-9):
        issues.append("vector_trace_solver_value_mismatch")


def _residual_verification(solver_result) -> Dict[str, Any]:
    trace = solver_result.trace if hasattr(solver_result, "trace") else {}
    if trace.get("stage") == "algebraic_constraint_engine":
        residuals = trace.get("residuals") or []
        failed = [item for item in residuals if not item.get("ok")]
        return {"status": "failed" if failed else "passed", "residuals": residuals, "failed_count": len(failed)}
    if trace.get("stage") not in {"registry_formula_solver", "registry_formula_redundant_path"}:
        return {}
    expression = trace.get("expression")
    sanitized = _sanitize_equation(str(expression or ""))
    if sanitized is None or solver_result.value is None or isinstance(solver_result.value, (str, list)):
        return {"status": "skipped", "reason": "unsupported_expression_or_value_shape"}
    try:
        lhs, rhs = sanitized.split("=", 1)
        values = {
            symbol: float(payload["si_value"])
            for symbol, payload in (trace.get("inputs") or {}).items()
            if isinstance(payload, dict) and payload.get("si_value") is not None
        }
        values.update({symbol: float(value) for symbol, value in (trace.get("constants") or {}).items()})
        lhs_symbol = lhs.strip()
        if lhs_symbol and lhs_symbol not in values:
            values[lhs_symbol] = float(solver_result.value)
        lhs_value = _safe_eval(lhs, values)
        rhs_value = _safe_eval(rhs, values)
        residual = lhs_value - rhs_value
        tolerance = 1e-7 * max(1.0, abs(lhs_value), abs(rhs_value))
        return {
            "status": "passed" if abs(residual) <= tolerance else "failed",
            "equation": sanitized,
            "residual": residual,
            "tolerance": tolerance,
        }
    except Exception as exc:
        return {"status": "skipped", "reason": f"residual_eval_failed:{type(exc).__name__}"}


def _physical_domain_verification(solver_result) -> Dict[str, Any]:
    trace = solver_result.trace if hasattr(solver_result, "trace") else {}
    issues: list[str] = []
    checked: list[dict] = []
    target_dimension = trace.get("target_dimension")
    if solver_result.value is not None and not isinstance(solver_result.value, (str, list)):
        issue = _domain_issue(str(target_dimension or ""), float(solver_result.value), "target")
        if issue:
            issues.append(issue)
        checked.append({"role": "target", "dimension": target_dimension, "value": solver_result.value})
    for symbol, payload in (trace.get("inputs") or {}).items():
        if not isinstance(payload, dict) or payload.get("si_value") is None:
            continue
        issue = _domain_issue(str(payload.get("dimension") or ""), float(payload["si_value"]), f"input:{symbol}")
        if issue:
            issues.append(issue)
        checked.append({"role": f"input:{symbol}", "dimension": payload.get("dimension"), "value": payload.get("si_value")})
    for index, payload in enumerate(trace.get("components") or [], start=1):
        if not isinstance(payload, dict) or payload.get("si_value") is None:
            continue
        issue = _domain_issue(str(payload.get("dimension") or ""), float(payload["si_value"]), f"component:{index}")
        if issue:
            issues.append(issue)
        checked.append({"role": f"component:{index}", "dimension": payload.get("dimension"), "value": payload.get("si_value")})
    return {"status": "failed" if issues else "passed", "issues": issues, "checked": checked}


def _domain_issue(dimension: str, value: float, role: str) -> str | None:
    positive = {"area", "capacitance", "count", "inductance", "length", "resistance", "resistivity", "turn_density"}
    nonnegative = {
        "electric_field",
        "energy",
        "force",
        "frequency",
        "magnetic_field",
        "magnetic_flux",
        "power",
        "time",
        "angular_frequency",
    }
    if dimension in positive and value <= 0:
        return f"physical_domain_violation:{role}:{dimension}:positive_required"
    if dimension in nonnegative and value < -1e-12:
        return f"physical_domain_violation:{role}:{dimension}:nonnegative_required"
    return None


def _uses_inferred_si_units(front_payload: dict) -> bool:
    return any(quantity.get("raw_unit") == "implicit_base_SI" for quantity in front_payload.get("quantities", []))


def _uncertainty_graph(front_payload: dict, route_result, solver_result, confidence: float, issues: list[str]) -> Dict[str, Any]:
    parse_confidence = float(front_payload.get("parse_confidence") or 0.0)
    route_confidence = float(getattr(route_result, "confidence", 0.0))
    solver_confidence = float(getattr(solver_result, "confidence", 0.0))
    inferred_si = _uses_inferred_si_units(front_payload)
    geometry_recoverable = bool(solver_result.trace.get("geometry", {}).get("recoverable")) if hasattr(solver_result, "trace") else False
    nodes = [
        {"id": "semantic_frontend", "confidence": parse_confidence, "factors": ["deterministic_parse"]},
        {"id": "constraint_graph", "confidence": route_confidence, "factors": list(getattr(route_result, "reasons", []))},
        {"id": "solver", "confidence": solver_confidence, "factors": [solver_result.trace.get("stage")] if hasattr(solver_result, "trace") else []},
        {"id": "verifier", "confidence": 0.0 if issues else 1.0, "factors": list(issues) or ["all_checks_passed"]},
    ]
    caps = []
    if inferred_si:
        caps.append({"cap": 0.82, "reason": "inferred_si_unit_used"})
    if geometry_recoverable:
        caps.append({"cap": 0.70, "reason": "geometry_template_assumption"})
    return {
        "status": "failed" if issues else "passed",
        "nodes": nodes,
        "caps": caps,
        "aggregation": "min(route,solver,caps); zero_on_issue",
        "answer_confidence": confidence,
    }


def _missing_required_dimensions(available_dimensions: list[str | None], required_dimensions: tuple[str, ...]) -> list[str]:
    pool = list(available_dimensions)
    missing: list[str] = []
    for dimension in required_dimensions:
        if dimension not in pool:
            missing.append(dimension)
        else:
            pool.remove(dimension)
    return missing


def _detect_quantity_contradictions(front_payload: dict) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []
    by_key: dict[str, dict] = {}
    text = str(front_payload.get("canonical_question") or "").lower()
    for quantity in front_payload.get("quantities", []):
        symbol = str(quantity.get("symbol") or "").strip().lower()
        if not symbol:
            continue
        state_id = str(quantity.get("state_id") or "state:base")
        entity_id = str(quantity.get("entity_id") or "")
        dimension = quantity.get("dimension")
        key = f"{_quantity_identity(quantity)}|{entity_id}|{state_id}"
        info = unit_info(quantity.get("unit") or "")
        if info is None:
            continue
        try:
            si_value = float(quantity.get("value")) * info.si_factor
        except (TypeError, ValueError):
            continue
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = {
                "symbol": symbol,
                "dimension": dimension,
                "si_value": si_value,
                "raw_text": quantity.get("raw_text"),
                "span": quantity.get("span"),
                "state_id": state_id,
                "entity_id": entity_id,
            }
            continue
        if previous["dimension"] != dimension:
            conflicts.append(
                {
                    "type": "contradictory_quantity_dimension",
                    "issue": f"contradictory_quantity_dimension:{symbol}:{previous['dimension']}:{dimension}",
                    "symbol": symbol,
                    "entity_id": entity_id or None,
                    "state_id": state_id,
                    "previous": previous["raw_text"],
                    "current": quantity.get("raw_text"),
                }
            )
            continue
        if not math.isclose(previous["si_value"], si_value, rel_tol=1e-9, abs_tol=1e-12):
            if _quantity_rebinding_allowed(text, previous.get("span"), quantity.get("span")):
                by_key[key] = {
                    "symbol": symbol,
                    "dimension": dimension,
                    "si_value": si_value,
                    "raw_text": quantity.get("raw_text"),
                    "span": quantity.get("span"),
                    "state_id": state_id,
                    "entity_id": entity_id,
                }
                continue
            conflicts.append(
                {
                    "type": "contradictory_quantity_value",
                    "issue": f"contradictory_quantity_value:{symbol}:{previous['raw_text']}:{quantity.get('raw_text')}",
                    "symbol": symbol,
                    "entity_id": entity_id or None,
                    "state_id": state_id,
                    "previous": previous["raw_text"],
                    "current": quantity.get("raw_text"),
                    "previous_si_value": previous["si_value"],
                    "current_si_value": si_value,
                }
            )
    return conflicts


def _quantity_identity(quantity: dict) -> str:
    symbol = str(quantity.get("symbol") or "").strip()
    dimension = str(quantity.get("dimension") or "")
    if dimension == "length" and re.fullmatch(r"[A-Z]{2}", symbol):
        return f"side:{symbol}"
    if dimension == "charge" and re.fullmatch(r"q[_-]?[A-Za-z]", symbol, flags=re.IGNORECASE):
        return f"charge:{symbol.lower()}"
    return f"symbol:{symbol.lower()}"


def _multi_path_verification(front_payload: dict, route_result, solver_result) -> Dict[str, Any]:
    if solver_result.value is None or isinstance(solver_result.value, (str, list)):
        return {}
    if solver_result.formula_id in {
        "symmetric_zero_force",
        "electric_field_symmetric_zero",
        "electric_field_square_three_equal_vertex",
        "electric_field_square_center_cancel_charge",
        "point_charge_field_midpoint_from_two_fields",
        "electric_field_two_charge_angle",
        "electric_field_resultant_two_vectors",
        "electric_equilibrium_mass_angle",
        "power_sum",
        "current_sum",
        "current_branch_difference",
        "rlc_quadrature_segment_current",
        "rlc_quadrature_segment_power",
        "rlc_quadrature_power_transfer_same_voltage",
        "rlc_quadrature_unknown_resistance",
        "rlc_quadrature_segment_voltage",
        "rlc_resonance_reactance_from_current_ratio",
        "rlc_resonance_resistor_voltage",
        "rlc_current_from_rlcf_voltage",
        "rlc_impedance_from_rlcf",
        "rlc_component_voltage",
        "rlc_power_impedance",
        "rlc_power_resonance",
        "ac_source_rms_from_sinusoid",
        "ac_source_angular_frequency",
        "lc_energy_complement",
        "inductor_energy_current_scaled",
        "magnetic_flux_linkage",
        "solenoid_flux_one_turn",
        "magnetic_energy_density_field",
        "solenoid_magnetic_energy_density",
        "solenoid_magnetic_energy",
    }:
        return {
            "status": "single_path",
            "path_count": 1,
            "paths": [
                {
                    "formula_id": solver_result.formula_id,
                    "value": float(solver_result.value),
                    "unit": solver_result.unit,
                    "role": "primary",
                }
            ],
            "reason": "structural geometry/vector proof is not comparable to direct scalar one-source paths",
        }
    if isinstance(getattr(solver_result, "trace", None), dict) and solver_result.trace.get("transformed_reactances"):
        return {
            "status": "single_path",
            "path_count": 1,
            "paths": [
                {
                    "formula_id": solver_result.formula_id,
                    "value": float(solver_result.value),
                    "unit": solver_result.unit,
                    "role": "primary",
                }
            ],
            "reason": "frequency-transformed RLC state is not comparable to direct DC Ohm paths",
        }
    formula = FORMULA_REGISTRY.get(solver_result.formula_id or "")
    target_dimension = (formula.target_dimension if formula else None) or solver_result.trace.get("target_dimension")
    if not target_dimension:
        return {}
    try:
        primary_value = float(solver_result.value)
    except (TypeError, ValueError):
        return {}
    alternates = solve_candidate_paths(
        front_payload,
        route_result,
        target_dimension=str(target_dimension),
        exclude_formula_id=solver_result.formula_id,
    )
    paths = [
        {
            "formula_id": solver_result.formula_id,
            "value": primary_value,
            "unit": solver_result.unit,
            "role": "primary",
        }
    ]
    for alternate in alternates:
        if alternate.unit != solver_result.unit:
            continue
        if not _alternate_path_is_comparable(front_payload, solver_result.formula_id, alternate.formula_id):
            continue
        try:
            value = float(alternate.value)
        except (TypeError, ValueError):
            continue
        paths.append(
            {
                "formula_id": alternate.formula_id,
                "value": value,
                "unit": alternate.unit,
                "role": "redundant",
            }
        )
    if len(paths) < 2:
        return {"status": "single_path", "path_count": 1, "paths": paths}
    disagreements = [
        path
        for path in paths[1:]
        if not math.isclose(primary_value, float(path["value"]), rel_tol=1e-6, abs_tol=1e-9)
    ]
    return {
        "status": "disagreement" if disagreements else "agreement",
        "path_count": len(paths),
        "paths": paths,
        "disagreements": disagreements,
    }


def _alternate_path_is_comparable(front_payload: dict, primary_formula_id: str | None, alternate_formula_id: str | None) -> bool:
    primary = FORMULA_REGISTRY.get(primary_formula_id or "")
    alternate = FORMULA_REGISTRY.get(alternate_formula_id or "")
    if not primary or not alternate:
        return True
    primary_required = Counter(primary.required_dimensions)
    alternate_required = Counter(alternate.required_dimensions)
    if alternate_required - primary_required:
        return True
    extra = primary_required - alternate_required
    if not extra:
        return True
    available = Counter(
        quantity.get("dimension")
        for quantity in front_payload.get("quantities", [])
        if quantity.get("dimension")
    )
    explicitly_present_extra = Counter({dimension: min(count, available[dimension]) for dimension, count in extra.items() if available[dimension] > 0})
    return not explicitly_present_extra


def _quantity_rebinding_allowed(text: str, first_span, second_span) -> bool:
    if not first_span or not second_span:
        return False
    try:
        _, first_end = first_span
        second_start, _ = second_span
    except (TypeError, ValueError):
        return False
    between = text[min(first_end, second_start) : max(first_end, second_start)]
    if has_change_factor_cue(between):
        return True
    return any(
        cue in between
        for cue in [
            "when",
            "then",
            "after",
            "before",
            "initial",
            "final",
            "becomes",
            "changed",
            "increases",
            "decreases",
            "at resonance",
            "at a frequency",
            "frequency is",
        ]
    )
