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
    entity_id: Optional[str] = None
    state_id: Optional[str] = None
    role: Optional[str] = None

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
            "entity_id": self.entity_id,
            "state_id": self.state_id,
            "role": self.role,
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
class Entity:
    """A grounded physical object mention."""

    entity_id: str
    label: str
    entity_type: str
    symbol: Optional[str] = None
    span: Optional[Span] = None
    context: str = ""
    confidence: float = 0.72

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "entity_type": self.entity_type,
            "symbol": self.symbol,
            "span": self.span,
            "context": self.context,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class State:
    """A temporal or physical state used to scope facts and quantities."""

    state_id: str
    label: str
    trigger_text: str
    span: Optional[Span] = None
    confidence: float = 0.68

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "label": self.label,
            "trigger_text": self.trigger_text,
            "span": self.span,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class Event:
    """A state transition or change event mentioned by the question."""

    event_id: str
    event_type: str
    trigger_text: str
    span: Optional[Span] = None
    confidence: float = 0.68

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "trigger_text": self.trigger_text,
            "span": self.span,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TopologyGraph:
    """A conservative canonical topology sketch for circuit statements."""

    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    canonical_form: str = "no_circuit_topology"
    is_complex: bool = False
    ambiguity: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [dict(node) for node in self.nodes],
            "edges": [dict(edge) for edge in self.edges],
            "canonical_form": self.canonical_form,
            "is_complex": self.is_complex,
            "ambiguity": list(self.ambiguity),
        }


@dataclass(frozen=True)
class Relation:
    """A semantic relation over entities, topology, or geometry."""

    relation_type: str
    subject: str
    object: Optional[str] = None
    qualifier: Optional[str] = None
    span: Optional[Span] = None
    evidence: str = ""
    confidence: float = 0.72

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "subject": self.subject,
            "object": self.object,
            "qualifier": self.qualifier,
            "span": self.span,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class Constraint:
    """A deterministic semantic constraint inferred from text."""

    constraint_id: str
    kind: str
    expression: str
    source: str
    span: Optional[Span] = None
    confidence: float = 0.72

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind,
            "expression": self.expression,
            "source": self.source,
            "span": self.span,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class Goal:
    """A target quantity requested by the question."""

    goal_id: str
    text: str
    dimension: Optional[str] = None
    symbol: Optional[str] = None
    span: Optional[Span] = None
    confidence: float = 0.74

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "text": self.text,
            "dimension": self.dimension,
            "symbol": self.symbol,
            "span": self.span,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class DerivedFact:
    """A code-owned fact inferred by deterministic forward chaining."""

    fact_id: str
    kind: str
    expression: str
    supports: List[str] = field(default_factory=list)
    confidence: float = 0.82

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "kind": self.kind,
            "expression": self.expression,
            "supports": list(self.supports),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class NormalizedQuestion:
    """Output of the deterministic semantic parser stage.

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
    entities: List[Entity] = field(default_factory=list)
    states: List[State] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    topology_graph: TopologyGraph = field(default_factory=TopologyGraph)
    relations: List[Relation] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    goals: List[Goal] = field(default_factory=list)
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
            "entities": [entity.to_dict() for entity in self.entities],
            "states": [state.to_dict() for state in self.states],
            "events": [event.to_dict() for event in self.events],
            "topology_graph": self.topology_graph.to_dict(),
            "relations": [relation.to_dict() for relation in self.relations],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "goals": [goal.to_dict() for goal in self.goals],
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
class LogicEngineResult:
    """Semantic parser output enriched by deterministic implicit facts."""

    normalized: NormalizedQuestion
    implicit_facts: List[ImplicitFact]
    premises: List[str]
    trace: Dict[str, Any]
    derived_facts: List[DerivedFact] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized": self.normalized.to_dict(),
            "implicit_facts": [fact.to_dict() for fact in self.implicit_facts],
            "derived_facts": [fact.to_dict() for fact in self.derived_facts],
            "premises": list(self.premises),
            "trace": dict(self.trace),
        }
