"""Optional constrained decoding adapters for Qwen planner JSON."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from .planner_schema import REQUIRED_PLANNER_FIELDS
from .qwen_config import QwenRuntimeConfig


@dataclass(frozen=True)
class ConstrainedGenerationResult:
    ok: bool
    text: str
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {"ok": self.ok, "text": self.text, "issues": list(self.issues), "trace": dict(self.trace)}


def planner_json_schema() -> dict:
    properties = {field: _json_schema_type(expected) for field, expected in REQUIRED_PLANNER_FIELDS.items()}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(REQUIRED_PLANNER_FIELDS),
        "properties": properties,
    }


def generate_constrained_planner_text(prompt: str, config: QwenRuntimeConfig, backend_name: str) -> ConstrainedGenerationResult:
    """Call a configured structured-output server.

    The endpoint is expected to be OpenAI-compatible. For vLLM, ``guided_json``
    is sent in the request body; for SGLang-compatible servers, the same schema
    is also included under ``response_format``.
    """

    if not config.structured_endpoint:
        return ConstrainedGenerationResult(False, "", ["structured_endpoint_missing"], {"stage": "constrained_decoding", "backend": backend_name})
    if backend_name not in {"vllm_guided_json", "sglang"}:
        return ConstrainedGenerationResult(False, "", [f"unsupported_server_backend:{backend_name}"], {"stage": "constrained_decoding", "backend": backend_name})
    payload = {
        "model": config.model_path,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": config.max_new_tokens,
        "guided_json": planner_json_schema(),
        "response_format": {"type": "json_schema", "json_schema": {"name": "physics_planner", "schema": planner_json_schema()}},
    }
    try:
        request = urllib.request.Request(
            config.structured_endpoint.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        return ConstrainedGenerationResult(True, text, [], {"stage": "constrained_decoding", "backend": backend_name, "endpoint": config.structured_endpoint})
    except Exception as exc:
        return ConstrainedGenerationResult(False, "", [f"constrained_decoding_error:{type(exc).__name__}"], {"stage": "constrained_decoding", "backend": backend_name, "endpoint": config.structured_endpoint, "error": str(exc)[:300]})


def _json_schema_type(expected) -> dict:
    if expected is str:
        return {"type": "string"}
    if expected is list:
        return {"type": "array"}
    if expected is int or expected is float or expected == (int, float):
        return {"type": "number"}
    if expected is type(None):
        return {"type": "null"}
    if isinstance(expected, tuple):
        types = []
        for item in expected:
            schema = _json_schema_type(item)
            schema_type = schema.get("type")
            if isinstance(schema_type, list):
                types.extend(schema_type)
            else:
                types.append(schema_type)
        return {"type": sorted(set(types))}
    return {}
