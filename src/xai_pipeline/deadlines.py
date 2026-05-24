"""Deadline bookkeeping for controlled pipeline exits."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Deadline:
    started_monotonic: float
    timeout_seconds: float

    def expired(self) -> bool:
        return self.elapsed_seconds() >= self.timeout_seconds

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_monotonic

    def remaining_seconds(self) -> float:
        return max(0.0, self.timeout_seconds - self.elapsed_seconds())

    def to_dict(self) -> dict:
        return {
            "timeout_seconds": self.timeout_seconds,
            "elapsed_seconds": round(self.elapsed_seconds(), 6),
            "remaining_seconds": round(self.remaining_seconds(), 6),
            "expired": self.expired(),
        }


def start_deadline(timeout_seconds: float = 55.0) -> Deadline:
    return Deadline(time.monotonic(), timeout_seconds)
