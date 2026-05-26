"""Trace-based XAI explanation boundary."""

from __future__ import annotations

from .trace import build_trace_explanation


def build_explanation(solver_result, unit_conversion_result=None) -> str:
    """Translate the executable trace into a concise public explanation."""

    return build_trace_explanation(solver_result, unit_conversion_result)
