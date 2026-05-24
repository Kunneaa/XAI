"""Lazy local Qwen runtime.

This module is only imported into the hot path when the user explicitly enables
local Qwen. Tests and deterministic batch solving should not load the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .qwen_config import QwenRuntimeConfig


_TOKENIZER: Any = None
_MODEL: Any = None
_LOADED_PATH: str | None = None


@dataclass(frozen=True)
class QwenGenerationResult:
    ok: bool
    text: str
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {"ok": self.ok, "text": self.text, "issues": list(self.issues), "trace": dict(self.trace)}


def generate_planner_text(prompt: str, config: QwenRuntimeConfig) -> QwenGenerationResult:
    """Generate raw planner JSON text from local Qwen with strict runtime guards."""

    started = time.monotonic()
    if not config.enabled:
        return QwenGenerationResult(False, "", ["local_qwen_disabled"], {"stage": "qwen_runtime"})
    if not config.readiness.ready:
        return QwenGenerationResult(False, "", ["local_qwen_not_ready", *config.readiness.issues], {"stage": "qwen_runtime", "readiness": config.readiness.to_dict()})
    if config.timeout_seconds <= 0:
        return QwenGenerationResult(False, "", ["qwen_timeout_budget_empty"], {"stage": "qwen_runtime"})
    try:
        tokenizer, model = _load_model(config)
        if hasattr(tokenizer, "apply_chat_template"):
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            chat_prompt = prompt
        inputs = tokenizer(chat_prompt, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        do_sample = config.temperature > 0.0
        generate_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        if do_sample:
            generate_kwargs["temperature"] = config.temperature
            generate_kwargs["top_p"] = config.top_p
        outputs = model.generate(**inputs, **generate_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        elapsed = time.monotonic() - started
        return QwenGenerationResult(
            True,
            text,
            [],
            {
                "stage": "qwen_runtime",
                "model_path": config.model_path,
                "elapsed_seconds": elapsed,
                "max_new_tokens": config.max_new_tokens,
                "guarded_json_only": True,
            },
        )
    except Exception as exc:  # pragma: no cover - exercised only with real model runtime
        return QwenGenerationResult(
            False,
            "",
            [f"qwen_runtime_error:{type(exc).__name__}"],
            {"stage": "qwen_runtime", "error": str(exc)[:300]},
        )


def _load_model(config: QwenRuntimeConfig):
    global _TOKENIZER, _MODEL, _LOADED_PATH
    if _TOKENIZER is not None and _MODEL is not None and _LOADED_PATH == config.model_path:
        return _TOKENIZER, _MODEL
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        device_map=config.device_map,
        torch_dtype=config.torch_dtype,
        **kwargs,
    )
    model.eval()
    _TOKENIZER = tokenizer
    _MODEL = model
    _LOADED_PATH = config.model_path
    return tokenizer, model
