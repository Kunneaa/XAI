"""Public primitives for the NSP-Core Physics XAI pipeline.

The package root stays intentionally lazy. Importing ``xai_pipeline.core.api``
should not also import every engine, registry, and local-LLM helper.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "handle_request": ("xai_pipeline.core.api", "handle_request"),
    "process_question": ("xai_pipeline.core.pipeline", "process_question"),
    "solve_algebraic_plan": ("xai_pipeline.engines.algebraic_engine", "solve_algebraic_plan"),
    "apply_logic_rules": ("xai_pipeline.engines.logic_engine", "apply_logic_rules"),
    "solve_multi_output": ("xai_pipeline.engines.multi_output", "solve_multi_output"),
    "normalize_question": ("xai_pipeline.frontend.semantic_parser", "normalize_question"),
    "process_question_front": ("xai_pipeline.frontend.semantic_parser", "process_question_front"),
    "formula_catalog_for_prompt": ("xai_pipeline.knowledge.formula_catalog", "formula_catalog_for_prompt"),
    "check_local_llm_readiness": ("xai_pipeline.planning.local_llm", "check_local_llm_readiness"),
    "propose_solve_plan_if_enabled": ("xai_pipeline.planning.local_llm", "propose_solve_plan_if_enabled"),
    "repair_solve_plan_once": ("xai_pipeline.planning.local_llm", "repair_solve_plan_once"),
    "build_deterministic_solve_plan": ("xai_pipeline.planning.solve_plan", "build_deterministic_solve_plan"),
    "compile_solve_plan": ("xai_pipeline.planning.plan_compiler", "compile_solve_plan"),
    "validate_structured_solve_plan": ("xai_pipeline.planning.plan_compiler", "validate_structured_solve_plan"),
    "build_plan_error_packet": ("xai_pipeline.planning.plan_compiler", "build_plan_error_packet"),
    "CompiledPlan": ("xai_pipeline.planning.plan_compiler", "CompiledPlan"),
    "PlanValidation": ("xai_pipeline.planning.plan_compiler", "PlanValidation"),
    "SolvePlanStep": ("xai_pipeline.planning.solve_plan", "SolvePlanStep"),
    "StructuredSolvePlan": ("xai_pipeline.planning.solve_plan", "StructuredSolvePlan"),
    "Constraint": ("xai_pipeline.frontend.semantic_ir", "Constraint"),
    "DerivedFact": ("xai_pipeline.frontend.semantic_ir", "DerivedFact"),
    "Entity": ("xai_pipeline.frontend.semantic_ir", "Entity"),
    "Event": ("xai_pipeline.frontend.semantic_ir", "Event"),
    "Goal": ("xai_pipeline.frontend.semantic_ir", "Goal"),
    "ImplicitFact": ("xai_pipeline.frontend.semantic_ir", "ImplicitFact"),
    "LogicEngineResult": ("xai_pipeline.frontend.semantic_ir", "LogicEngineResult"),
    "NormalizedQuestion": ("xai_pipeline.frontend.semantic_ir", "NormalizedQuestion"),
    "NumericConstant": ("xai_pipeline.frontend.semantic_ir", "NumericConstant"),
    "Quantity": ("xai_pipeline.frontend.semantic_ir", "Quantity"),
    "Relation": ("xai_pipeline.frontend.semantic_ir", "Relation"),
    "State": ("xai_pipeline.frontend.semantic_ir", "State"),
    "SymbolicQuantity": ("xai_pipeline.frontend.semantic_ir", "SymbolicQuantity"),
    "SymbolicRelation": ("xai_pipeline.frontend.semantic_ir", "SymbolicRelation"),
    "TopologyGraph": ("xai_pipeline.frontend.semantic_ir", "TopologyGraph"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
