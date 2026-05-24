"""Executor mode registry and deterministic mode selection."""

from __future__ import annotations


EXECUTOR_MODES = {
    "direct_numeric_formula",
    "principle_equation_system",
    "symbolic_expression",
    "vector_geometry",
    "yes_no_condition",
    "conceptual",
    "multi_output",
}


def select_executor_mode(front_payload: dict, route_result) -> dict:
    answer_type = front_payload.get("answer_type_hint")
    task_type = route_result.task_type
    if answer_type == "multi_output":
        mode = "multi_output"
    elif answer_type == "symbolic":
        mode = "symbolic_expression"
    elif answer_type == "yes_no":
        mode = "yes_no_condition"
    elif task_type in {"coulomb_force", "electric_field_point", "resultant_force"}:
        mode = "vector_geometry"
    elif answer_type == "conceptual" or task_type == "conceptual":
        mode = "conceptual"
    else:
        mode = "direct_numeric_formula"
    return {"stage": "executor_mode_selector", "mode": mode, "supported_modes": sorted(EXECUTOR_MODES)}
