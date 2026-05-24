"""Structured-output backend boundary for Qwen planner JSON."""

from __future__ import annotations

import os
from dataclasses import dataclass


SUPPORTED_BACKENDS = {"vllm_guided_json", "sglang", "outlines", "lm_format_enforcer", "local_transformers_json_guarded"}


@dataclass(frozen=True)
class StructuredOutputBackend:
    name: str
    available: bool
    issues: list[str]

    def to_dict(self) -> dict:
        return {"name": self.name, "available": self.available, "issues": list(self.issues)}


def select_structured_output_backend(preferred: str | None = None) -> StructuredOutputBackend:
    """Select a constrained decoding backend if installed.

    Server-backed engines are reported unavailable unless their packages are
    importable in this process. Local Transformers generation is a guarded JSON
    fallback: it is not true constrained decoding, so validator confidence caps
    and schema rejection still apply.
    """

    name = preferred or "vllm_guided_json"
    if name not in SUPPORTED_BACKENDS:
        return StructuredOutputBackend(name, False, [f"unknown_structured_output_backend:{name}"])
    if name == "local_transformers_json_guarded":
        return StructuredOutputBackend(name, True, ["guarded_json_generation_not_constrained_decoding"])
    if name == "vllm_guided_json" and os.environ.get("XAI_QWEN_STRUCTURED_ENDPOINT"):
        return StructuredOutputBackend(name, True, ["server_guided_json_endpoint_configured"])
    if name == "sglang" and os.environ.get("XAI_QWEN_STRUCTURED_ENDPOINT"):
        return StructuredOutputBackend(name, True, ["server_structured_output_endpoint_configured"])
    if name == "outlines":
        return _importable_backend(name, "outlines")
    if name == "lm_format_enforcer":
        return _importable_backend(name, "lmformatenforcer")
    if name == "sglang":
        return _importable_backend(name, "sglang")
    return StructuredOutputBackend(name, False, ["structured_output_backend_not_connected"])


def _importable_backend(name: str, module_name: str) -> StructuredOutputBackend:
    try:
        __import__(module_name)
    except Exception:
        return StructuredOutputBackend(name, False, [f"structured_output_backend_not_installed:{module_name}"])
    return StructuredOutputBackend(name, True, [])
