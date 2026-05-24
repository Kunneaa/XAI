"""Shared schemas for deterministic pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


Span = Tuple[int, int]


@dataclass(frozen=True)
class Quantity:
    """A raw quantity found in the question text."""

    raw_text: str
    value: float
    unit: str
    raw_unit: str
    symbol: Optional[str] = None
    dimension: Optional[str] = None
    span: Optional[Span] = None
    context: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "value": self.value,
            "unit": self.unit,
            "raw_unit": self.raw_unit,
            "symbol": self.symbol,
            "dimension": self.dimension,
            "span": self.span,
            "context": self.context,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SymbolicQuantity:
    """A symbolic variable or symbolic quantity mention found in the question."""

    raw_text: str
    symbol: str
    unit: Optional[str] = None
    raw_unit: Optional[str] = None
    dimension: Optional[str] = None
    span: Optional[Span] = None
    context: str = ""
    confidence: float = 0.75

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "symbol": self.symbol,
            "unit": self.unit,
            "raw_unit": self.raw_unit,
            "dimension": self.dimension,
            "span": self.span,
            "context": self.context,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SymbolicRelation:
    """A deterministic symbolic relation extracted from the question text."""

    raw_text: str
    lhs: str
    rhs: str
    span: Optional[Span] = None
    context: str = ""
    confidence: float = 0.78

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "span": self.span,
            "context": self.context,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class NumericConstant:
    """A numeric assignment without an explicit physical unit."""

    raw_text: str
    symbol: str
    value: float
    dimension: Optional[str] = None
    span: Optional[Span] = None
    context: str = ""
    confidence: float = 0.82

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "symbol": self.symbol,
            "value": self.value,
            "dimension": self.dimension,
            "span": self.span,
            "context": self.context,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class NormalizedQuestion:
    """Output of the normalizer stage.

    The raw question is preserved for audit and cache policy. The canonical
    text is deterministic and should not depend on LLM rewrites.
    """

    raw_question: str
    canonical_question: str
    quantities: List[Quantity]
    parse_confidence: float
    symbolic_quantities: List[SymbolicQuantity] = field(default_factory=list)
    symbolic_relations: List[SymbolicRelation] = field(default_factory=list)
    numeric_constants: List[NumericConstant] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    target_hints: List[str] = field(default_factory=list)
    answer_type_hint: str = "unknown"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_question": self.raw_question,
            "canonical_question": self.canonical_question,
            "quantities": [q.to_dict() for q in self.quantities],
            "symbolic_quantities": [q.to_dict() for q in self.symbolic_quantities],
            "symbolic_relations": [r.to_dict() for r in self.symbolic_relations],
            "numeric_constants": [c.to_dict() for c in self.numeric_constants],
            "concepts": list(self.concepts),
            "target_hints": list(self.target_hints),
            "answer_type_hint": self.answer_type_hint,
            "parse_confidence": self.parse_confidence,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ImplicitFact:
    """A deterministic implicit assumption or constant added by the KB."""

    rule_id: str
    adds: Dict[str, str]
    premise: str
    trigger_text: str
    span: Optional[Span]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "adds": dict(self.adds),
            "premise": self.premise,
            "trigger_text": self.trigger_text,
            "span": self.span,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ImplicitKBResult:
    """Normalizer output enriched by deterministic implicit facts."""

    normalized: NormalizedQuestion
    implicit_facts: List[ImplicitFact]
    premises: List[str]
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized": self.normalized.to_dict(),
            "implicit_facts": [fact.to_dict() for fact in self.implicit_facts],
            "premises": list(self.premises),
            "trace": dict(self.trace),
        }
