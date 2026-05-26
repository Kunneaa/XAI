"""Small API-facing wrapper for the Physics XAI pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .pipeline import process_question


def handle_request(
    payload: Dict[str, Any],
    *,
    data_path: str | Path | None = None,
    enable_llm: bool = False,
    timeout_seconds: float = 55.0,
    planning_mode: str | None = None,
) -> Dict[str, Any]:
    """Handle the core API shape: {"question": "..."}.

    Expected validation failures return controlled responses instead of raising
    web-layer exceptions. The caller can still choose how to map this response
    to HTTP status codes.
    """

    question = payload.get("question") if isinstance(payload, dict) else None
    if not isinstance(question, str) or not question.strip():
        return {
            "answer": "Uncertain",
            "explanation": "The request must contain a non-empty string field named `question`.",
            "confidence": 0.0,
            "metadata": {"status": "invalid_request"},
        }
    return process_question(
        question,
        data_path=Path(data_path) if data_path is not None else None,
        enable_llm=enable_llm,
        timeout_seconds=timeout_seconds,
        planning_mode=planning_mode,
    )
