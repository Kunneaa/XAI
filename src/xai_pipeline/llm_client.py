import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LLMConfig:
    enabled: bool
    model_id: str
    local_model_path: Optional[str] = None
    max_new_tokens: int = 320
    temperature: float = 0.0


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        enabled=os.getenv("XAI_USE_LLM", "1") == "1",
        model_id=os.getenv("XAI_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
        local_model_path=os.getenv("XAI_MODEL_PATH"),
    )


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(n in text for n in needles)


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or load_llm_config()
        self.ready = False
        self.error: Optional[str] = None
        self.model = None
        self.tokenizer = None

        if not self.config.enabled:
            self.error = "LLM disabled by XAI_USE_LLM=0"
            return

        try:
            model_ref = self.config.model_id
            if self.config.local_model_path:
                p = Path(self.config.local_model_path).expanduser().resolve()
                if not p.exists():
                    self.error = f"Local model path not found: {p}"
                    return
                model_ref = str(p)
            elif not self._is_model_allowed(self.config.model_id):
                self.error = (
                    f"Model `{self.config.model_id}` is blocked by policy. "
                    "Use an open-source model with <=8B params."
                )
                return

            self.model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                torch_dtype="auto",
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_ref)
            self.ready = True
            self.error = None
        except Exception as e:
            self.ready = False
            self.model = None
            self.tokenizer = None
            self.error = str(e)

    @staticmethod
    def _is_model_allowed(model_id: str) -> bool:
        mid = model_id.lower()
        if _contains_any(mid, ["gpt", "claude", "gemini"]):
            return False

        # Explicit allowlist override for controlled deployment.
        allowlist = os.getenv("XAI_MODEL_ALLOWLIST", "").strip()
        if allowlist:
            allowed_ids = [x.strip().lower() for x in allowlist.split(",") if x.strip()]
            return mid in allowed_ids

        if _contains_any(mid, ["70b", "72b", "13b", "14b", "27b", "32b", "34b"]):
            return False
        if _contains_any(mid, ["8b", "7b", "6.7b", "3b", "2b", "1.5b", "1b", "0.5b"]):
            return True
        return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        if not self.ready or self.model is None or self.tokenizer is None:
            return None

        try:
            temp = self.config.temperature if temperature is None else temperature
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            }
            if temp and temp > 0:
                gen_kwargs.update({"do_sample": True, "temperature": temp})

            generated_ids = self.model.generate(**model_inputs, **gen_kwargs)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
        except Exception:
            return None

    def generate_json(self, prompt: str, required_keys: list[str]) -> Optional[Dict[str, Any]]:
        raw = self.generate(prompt)
        if not raw:
            return None

        def _valid(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if all(k in obj for k in required_keys):
                return obj
            return None

        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                valid = _valid(obj)
                if valid is not None:
                    return valid
        except Exception:
            pass

        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(raw[start : end + 1])
                if isinstance(obj, dict):
                    valid = _valid(obj)
                    if valid is not None:
                        return valid
        except Exception:
            pass

        return None
