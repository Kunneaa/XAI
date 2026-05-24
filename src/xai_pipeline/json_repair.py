"""Conservative JSON parsing and repair helpers for planner output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JsonRepairResult:
    ok: bool
    value: Any
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {"ok": self.ok, "value": self.value, "issues": list(self.issues), "trace": dict(self.trace)}


def parse_or_repair_json(text: str) -> JsonRepairResult:
    """Parse JSON, then try one bounded extraction repair pass.

    The repair pass only extracts the outermost object and removes trailing
    commas. It never fabricates fields.
    """

    raw = str(text or "")
    try:
        return JsonRepairResult(True, json.loads(raw), [], {"stage": "json_repair", "mode": "strict"})
    except json.JSONDecodeError as first_error:
        candidate = _extract_object(raw)
        if candidate is None:
            return JsonRepairResult(False, None, [f"json_decode_error:{first_error.msg}"], {"stage": "json_repair", "mode": "failed"})
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return JsonRepairResult(
                True,
                json.loads(candidate),
                ["strict_parse_failed_repaired_by_object_extraction"],
                {"stage": "json_repair", "mode": "single_repair"},
            )
        except json.JSONDecodeError as second_error:
            return JsonRepairResult(
                False,
                None,
                [f"json_decode_error:{first_error.msg}", f"repair_decode_error:{second_error.msg}"],
                {"stage": "json_repair", "mode": "failed"},
            )


def _extract_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    return text[start : end + 1]
