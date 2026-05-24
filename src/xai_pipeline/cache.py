"""Small verified-response cache.

The cache is intentionally in-memory and conservative: callers may only store
responses after deterministic verification has accepted the answer.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Optional

from .normalizer import canonicalize_question


_VERIFIED_RESPONSE_CACHE: Dict[str, Dict[str, Any]] = {}


def get_verified_response(raw_question: str) -> Optional[Dict[str, Any]]:
    cached = _VERIFIED_RESPONSE_CACHE.get(_key(raw_question))
    return copy.deepcopy(cached) if cached is not None else None


def put_verified_response(raw_question: str, response: Dict[str, Any]) -> None:
    verifier = response.get("verifier") or {}
    if verifier.get("ok") is not True:
        return
    answer_checker = response.get("answer_checker") or {}
    if answer_checker.get("ok") is not True:
        return
    _VERIFIED_RESPONSE_CACHE[_key(raw_question)] = copy.deepcopy(response)


def clear_verified_response_cache() -> None:
    _VERIFIED_RESPONSE_CACHE.clear()


def _key(raw_question: str) -> str:
    canonical = canonicalize_question(str(raw_question or ""))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
