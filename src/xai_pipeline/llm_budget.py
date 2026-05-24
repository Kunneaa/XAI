"""LLM call budget state machine from the core contract."""

from __future__ import annotations

from dataclasses import dataclass, field


SECOND_CALL_FEATURES = {"implicit_classifier", "split_planning", "repair", "guarded_polish"}


@dataclass
class LlmBudgetState:
    max_calls: int = 2
    calls_used: int = 0
    second_call_feature: str | None = None
    events: list[dict] = field(default_factory=list)

    def can_call(self, feature: str) -> bool:
        if self.calls_used >= self.max_calls:
            return False
        if self.calls_used == 1 and feature in SECOND_CALL_FEATURES:
            return self.second_call_feature in {None, feature}
        if self.second_call_feature and feature in SECOND_CALL_FEATURES and feature != self.second_call_feature:
            return False
        return True

    def record_call(self, feature: str) -> bool:
        if not self.can_call(feature):
            self.events.append({"feature": feature, "accepted": False, "calls_used": self.calls_used})
            return False
        self.calls_used += 1
        if feature in SECOND_CALL_FEATURES:
            self.second_call_feature = feature
        self.events.append({"feature": feature, "accepted": True, "calls_used": self.calls_used})
        return True

    def can_polish(self) -> bool:
        return self.can_call("guarded_polish") and self.second_call_feature in {None, "guarded_polish"}

    def to_dict(self) -> dict:
        return {
            "max_calls": self.max_calls,
            "calls_used": self.calls_used,
            "second_call_feature": self.second_call_feature,
            "events": list(self.events),
        }


def budget_from_trace(trace: dict | None) -> LlmBudgetState:
    """Rehydrate a budget snapshot so downstream optional LLM steps share it."""

    if not isinstance(trace, dict):
        return LlmBudgetState()
    state = LlmBudgetState(
        max_calls=int(trace.get("max_calls", 2)),
        calls_used=int(trace.get("calls_used", 0)),
        second_call_feature=trace.get("second_call_feature"),
    )
    events = trace.get("events")
    if isinstance(events, list):
        state.events = list(events)
    return state
