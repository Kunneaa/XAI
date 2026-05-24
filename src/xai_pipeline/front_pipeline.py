"""Deterministic front pipeline: normalize -> implicit KB."""

from __future__ import annotations

from typing import Any, Dict

from .implicit_kb import apply_implicit_kb
from .normalizer import normalize_question


def process_question_front(question: str) -> Dict[str, Any]:
    """Run the deterministic stages before routing/planning.

    This function is intentionally LLM-free. It preserves raw input, emits
    audit-friendly traces, and only applies implicit facts from code-owned
    allowlists.
    """

    normalized = normalize_question(question)
    enriched = apply_implicit_kb(normalized)
    return {
        "raw_question": normalized.raw_question,
        "canonical_question": normalized.canonical_question,
        "quantities": [quantity.to_dict() for quantity in normalized.quantities],
        "symbolic_quantities": [quantity.to_dict() for quantity in normalized.symbolic_quantities],
        "symbolic_relations": [relation.to_dict() for relation in normalized.symbolic_relations],
        "numeric_constants": [constant.to_dict() for constant in normalized.numeric_constants],
        "concepts": list(normalized.concepts),
        "target_hints": list(normalized.target_hints),
        "answer_type_hint": normalized.answer_type_hint,
        "parse_confidence": normalized.parse_confidence,
        "warnings": list(normalized.warnings),
        "implicit_facts": [fact.to_dict() for fact in enriched.implicit_facts],
        "premises": list(enriched.premises),
        "trace": {
            "stages": ["normalize", "implicit_kb"],
            "normalize": {
                "quantity_count": len(normalized.quantities),
                "symbolic_quantity_count": len(normalized.symbolic_quantities),
                "symbolic_relation_count": len(normalized.symbolic_relations),
                "numeric_constant_count": len(normalized.numeric_constants),
                "concept_count": len(normalized.concepts),
                "target_hint_count": len(normalized.target_hints),
                "answer_type_hint": normalized.answer_type_hint,
                "warnings": list(normalized.warnings),
                "llm_used": False,
            },
            "implicit_kb": enriched.trace,
        },
    }
