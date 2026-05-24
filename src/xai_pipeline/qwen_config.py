"""Local Qwen runtime configuration and readiness checks.

The core contract is intentionally conservative: a local model can assist with
planning JSON, but it must never be required for deterministic solving or tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class QwenModelReadiness:
    ready: bool
    model_path: str
    issues: list[str]
    files: dict

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "model_path": self.model_path,
            "issues": list(self.issues),
            "files": dict(self.files),
        }


@dataclass(frozen=True)
class QwenRuntimeConfig:
    model_path: str
    enabled: bool
    local_files_only: bool
    device_map: str
    torch_dtype: str
    max_new_tokens: int
    temperature: float
    top_p: float
    timeout_seconds: float
    structured_backend: str
    structured_endpoint: str | None
    require_constrained_decoding: bool
    trust_remote_code: bool
    readiness: QwenModelReadiness

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "enabled": self.enabled,
            "local_files_only": self.local_files_only,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout_seconds": self.timeout_seconds,
            "structured_backend": self.structured_backend,
            "structured_endpoint": self.structured_endpoint,
            "require_constrained_decoding": self.require_constrained_decoding,
            "trust_remote_code": self.trust_remote_code,
            "readiness": self.readiness.to_dict(),
        }


def resolve_qwen_runtime_config(model_path: str | Path | None = None) -> QwenRuntimeConfig:
    """Resolve local Qwen settings from arguments and environment variables."""

    path = Path(model_path or os.environ.get("XAI_QWEN_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser()
    readiness = check_qwen_model_readiness(path)
    return QwenRuntimeConfig(
        model_path=str(path),
        enabled=os.environ.get("XAI_ENABLE_LOCAL_QWEN", "0").strip().lower() in {"1", "true", "yes", "on"},
        local_files_only=os.environ.get("XAI_QWEN_LOCAL_FILES_ONLY", "1").strip().lower() not in {"0", "false", "no", "off"},
        device_map=os.environ.get("XAI_QWEN_DEVICE_MAP", "auto"),
        torch_dtype=os.environ.get("XAI_QWEN_TORCH_DTYPE", "auto"),
        max_new_tokens=_env_int("XAI_QWEN_MAX_NEW_TOKENS", 768),
        temperature=_env_float("XAI_QWEN_TEMPERATURE", 0.0),
        top_p=_env_float("XAI_QWEN_TOP_P", 1.0),
        timeout_seconds=_env_float("XAI_QWEN_TIMEOUT_SECONDS", 30.0),
        structured_backend=os.environ.get("XAI_QWEN_STRUCTURED_BACKEND", "local_transformers_json_guarded"),
        structured_endpoint=os.environ.get("XAI_QWEN_STRUCTURED_ENDPOINT"),
        require_constrained_decoding=os.environ.get("XAI_REQUIRE_CONSTRAINED_QWEN", "0").strip().lower() in {"1", "true", "yes", "on"},
        trust_remote_code=os.environ.get("XAI_QWEN_TRUST_REMOTE_CODE", "0").strip().lower() in {"1", "true", "yes", "on"},
        readiness=readiness,
    )


def check_qwen_model_readiness(model_path: str | Path | None = None) -> QwenModelReadiness:
    """Check that the local Hugging Face model folder is complete enough."""

    path = Path(model_path or DEFAULT_MODEL_DIR).expanduser()
    required = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]
    files: dict[str, bool | int] = {}
    issues: list[str] = []
    if not path.exists():
        return QwenModelReadiness(False, str(path), ["model_path_missing"], {})
    if not path.is_dir():
        return QwenModelReadiness(False, str(path), ["model_path_not_directory"], {})
    for name in required:
        exists = (path / name).is_file()
        files[name] = exists
        if not exists:
            issues.append(f"missing_file:{name}")
    shard_count = len(list(path.glob("model-*.safetensors")))
    files["safetensor_shards"] = shard_count
    if shard_count == 0:
        issues.append("missing_safetensor_shards")
    return QwenModelReadiness(not issues, str(path), issues, files)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default
