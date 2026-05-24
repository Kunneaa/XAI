"""Deterministic explanation builder from execution trace."""

from __future__ import annotations


def build_trace_explanation(solver_result, unit_conversion_result) -> str:
    if not solver_result.solved:
        return "The deterministic executor did not produce a verified trace for this question."

    premise = solver_result.premises[0] if solver_result.premises else "Use the verified formula."
    expression = solver_result.trace.get("expression")
    converted_count = unit_conversion_result.trace.get("converted_quantity_count", 0)
    parts = [premise]
    if expression:
        parts.append(f"Formula used: {expression}.")
    parts.append(f"The solver converted {converted_count} extracted quantity/quantities to SI units before substitution.")
    parts.append(f"The verified result is {solver_result.answer}.")
    return " ".join(parts)
