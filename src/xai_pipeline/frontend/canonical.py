"""Canonical structural normalization for deterministic reasoning.

This module turns surface labels such as ``MNQ``, ``qM``, ``R_load`` or
``C2`` into small canonical structures that downstream engines can reason
over without depending on the exact names used in a dataset row.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


CANONICAL_TRIANGLE_POINTS = ("A", "B", "C")
CANONICAL_SQUARE_POINTS = ("A", "B", "C", "D")


def build_canonical_structures(front_payload: dict) -> dict:
    """Build deterministic canonical maps from an already extracted front IR."""

    return {
        "stage": "canonical_structure_builder",
        "geometry": {
            "triangles": _canonical_triangles(front_payload),
            "squares": _canonical_squares(front_payload),
        },
        "component_groups": _canonical_component_groups(front_payload),
    }


def first_triangle_context(front_payload: dict) -> dict | None:
    """Return the first canonical triangle context, if one is recoverable."""

    structures = front_payload.get("canonical_structures") or build_canonical_structures(front_payload)
    triangles = (structures.get("geometry") or {}).get("triangles") or []
    return triangles[0] if triangles else None


def canonical_point(point: str | None, triangle_context: dict | None = None) -> str | None:
    if not point:
        return None
    label = str(point).upper()
    if triangle_context:
        mapped = (triangle_context.get("canonical_by_original") or {}).get(label)
        if mapped:
            return mapped
    if label in CANONICAL_TRIANGLE_POINTS:
        return label
    return None


def canonical_side_key(symbol: str, triangle_context: dict | None = None) -> str | None:
    if not re.fullmatch(r"[A-Za-z]{2}", str(symbol or "")):
        return None
    left = canonical_point(symbol[0], triangle_context)
    right = canonical_point(symbol[1], triangle_context)
    if not left or not right or left == right:
        return None
    return "".join(sorted((left.lower(), right.lower())))


def right_angle_point(text: str) -> str | None:
    patterns = (
        r"right[- ]?angled\s+at\s+(?:point\s+)?([a-z])\b",
        r"right[- ]?angle(?:d)?\s+vertex\s+([a-z])\b",
        r"right angle\s+at\s+(?:point\s+)?([a-z])\b",
        r"angle\s+([a-z])\s*(?:is|=)\s*90",
    )
    for pattern in patterns:
        match = re.search(pattern, str(text or "").lower())
        if match:
            return match.group(1).upper()
    return None


def _canonical_triangles(front_payload: dict) -> list[dict]:
    text = str(front_payload.get("canonical_question") or "")
    labels = _triangle_labels_from_text(text) or _infer_triangle_labels_from_quantities(front_payload)
    if len(labels) != 3:
        return []
    right_angle = right_angle_point(text) or _right_isosceles_equal_leg_vertex(text, labels)
    labels = _normalize_triangle_label_order(labels, right_angle)
    canonical_by_original = {original: canonical for original, canonical in zip(labels, CANONICAL_TRIANGLE_POINTS)}
    original_by_canonical = {canonical: original for original, canonical in canonical_by_original.items()}
    context = {
        "structure_id": "triangle:1",
        "kind": "triangle",
        "labels": labels,
        "canonical_points": list(CANONICAL_TRIANGLE_POINTS),
        "canonical_by_original": canonical_by_original,
        "original_by_canonical": original_by_canonical,
        "right_angle_at": right_angle,
        "canonical_right_angle_at": canonical_point(right_angle, {"canonical_by_original": canonical_by_original}),
        "side_symbols": {},
        "charge_symbols": {},
    }
    for quantity in front_payload.get("quantities", []):
        symbol = str(quantity.get("symbol") or "")
        if quantity.get("dimension") == "length":
            key = canonical_side_key(symbol, context)
            if key:
                context["side_symbols"][symbol] = key
        if quantity.get("dimension") == "charge":
            match = re.fullmatch(r"q[_-]?([A-Za-z])", symbol, flags=re.IGNORECASE)
            if match:
                point = canonical_point(match.group(1), context)
                if point:
                    context["charge_symbols"][symbol] = point
    return [context]


def _normalize_triangle_label_order(labels: list[str], right_angle: str | None) -> list[str]:
    """Use a structural local frame instead of preserving arbitrary surface order.

    For right-triangle constructors, the right-angle vertex is the natural local
    origin. Mapping it to canonical ``A`` prevents downstream engines from
    accidentally depending on a dataset's triangle name order, such as only
    solving when the right-angle label is the first letter in ``ABC``.
    """

    if not right_angle or right_angle not in labels:
        return labels
    return [right_angle] + [label for label in labels if label != right_angle]


def _triangle_labels_from_text(text: str) -> list[str]:
    compact = re.search(r"\btriangle\s+([A-Za-z]{3})\b", text)
    if compact:
        labels = list(compact.group(1).upper())
        return labels if len(set(labels)) == 3 else []
    listed = re.search(
        r"\b(?:triangle\s+(?:with\s+|has\s+)?(?:vertices|points)|vertices\s+of\s+(?:a\s+)?triangle)\s+"
        r"([A-Za-z])\s*,?\s+([A-Za-z])\s*,?\s*(?:and\s+)?([A-Za-z])\b",
        text,
        flags=re.IGNORECASE,
    )
    if not listed:
        return []
    labels = [raw.upper() for raw in listed.groups()]
    return labels if len(set(labels)) == 3 else []


def _right_isosceles_equal_leg_vertex(text: str, labels: list[str]) -> str | None:
    """Infer the right-angle vertex from equal legs in a right-isosceles triangle.

    The shared endpoint of the two equal sides is the right-angle vertex. This
    is structural geometry recovery, not a dependency on labels such as AB/AC.
    """

    lowered = str(text or "").lower()
    if not re.search(r"\bright isosceles triangle\b|\bisosceles right triangle\b|\bright[- ]?angled isosceles triangle\b", lowered):
        return None
    label_set = set(labels)
    for match in re.finditer(r"\b([A-Za-z]{2})\s*=\s*([A-Za-z]{2})\b", text):
        first, second = match.group(1).upper(), match.group(2).upper()
        if not set(first) <= label_set or not set(second) <= label_set:
            continue
        shared = set(first) & set(second)
        if len(shared) == 1:
            return next(iter(shared))
    return None


def _infer_triangle_labels_from_quantities(front_payload: dict) -> list[str]:
    labels: list[str] = []
    text = str(front_payload.get("canonical_question") or "")
    point_pair = re.search(r"\bpoints?\s+([A-Za-z])\s+and\s+([A-Za-z])\b", text, flags=re.IGNORECASE)
    if point_pair:
        for raw in point_pair.groups():
            label = raw.upper()
            if label not in labels:
                labels.append(label)
    for match in re.finditer(r"\bpoint\s+([A-Za-z])\b", text, flags=re.IGNORECASE):
        label = match.group(1).upper()
        if label not in labels:
            labels.append(label)
        if len(labels) == 3:
            break
    angle_point = right_angle_point(str(front_payload.get("canonical_question") or ""))
    if angle_point:
        if angle_point not in labels:
            labels.append(angle_point)
    for quantity in front_payload.get("quantities", []):
        symbol = str(quantity.get("symbol") or "")
        if quantity.get("dimension") == "length" and re.fullmatch(r"[A-Za-z]{2}", symbol):
            for char in symbol.upper():
                if char not in labels:
                    labels.append(char)
        if quantity.get("dimension") == "charge":
            match = re.fullmatch(r"q[_-]?([A-Za-z])", symbol, flags=re.IGNORECASE)
            if match and match.group(1).upper() not in labels:
                labels.append(match.group(1).upper())
    return labels[:3]


def _canonical_squares(front_payload: dict) -> list[dict]:
    text = str(front_payload.get("canonical_question") or "")
    labels = _square_labels_from_text(text)
    if len(labels) != 4:
        return []
    canonical_by_original = {original: canonical for original, canonical in zip(labels, CANONICAL_SQUARE_POINTS)}
    original_by_canonical = {canonical: original for original, canonical in canonical_by_original.items()}
    context = {
        "structure_id": "square:1",
        "kind": "square",
        "labels": labels,
        "canonical_points": list(CANONICAL_SQUARE_POINTS),
        "canonical_by_original": canonical_by_original,
        "original_by_canonical": original_by_canonical,
        "side_symbols": {},
        "charge_symbols": {},
    }
    for quantity in front_payload.get("quantities", []):
        symbol = str(quantity.get("symbol") or "")
        if quantity.get("dimension") == "charge":
            match = re.fullmatch(r"q[_-]?([A-Za-z])", symbol, flags=re.IGNORECASE)
            if match:
                point = canonical_by_original.get(match.group(1).upper())
                if point:
                    context["charge_symbols"][symbol] = point
    return [context]


def _square_labels_from_text(text: str) -> list[str]:
    compact = re.search(r"\bsquare\s+([A-Za-z]{4})\b", text)
    if compact:
        labels = list(compact.group(1).upper())
        return labels if len(set(labels)) == 4 else []
    listed = re.search(
        r"\b(?:square\s+(?:with\s+|has\s+)?(?:vertices|corners|points)|vertices\s+of\s+(?:a\s+)?square)\s+"
        r"([A-Za-z])\s*,?\s+([A-Za-z])\s*,?\s+([A-Za-z])\s*,?\s*(?:and\s+)?([A-Za-z])\b",
        text,
        flags=re.IGNORECASE,
    )
    if not listed:
        return []
    labels = [raw.upper() for raw in listed.groups()]
    return labels if len(set(labels)) == 4 else []


def _canonical_component_groups(front_payload: dict) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for quantity in front_payload.get("quantities", []):
        dimension = quantity.get("dimension")
        if dimension not in {
            "capacitance",
            "charge",
            "current",
            "inductance",
            "length",
            "power",
            "resistance",
            "voltage",
        }:
            continue
        grouped[str(dimension)].append(quantity)
    out: dict[str, list[dict[str, Any]]] = {}
    for dimension, quantities in grouped.items():
        ordered = sorted(quantities, key=lambda item: item.get("span") or (10**9, 10**9))
        out[dimension] = [
            {
                "canonical_id": f"{dimension}:{index + 1}",
                "symbol": quantity.get("symbol"),
                "raw_text": quantity.get("raw_text"),
                "entity_id": quantity.get("entity_id"),
                "state_id": quantity.get("state_id"),
                "span": quantity.get("span"),
            }
            for index, quantity in enumerate(ordered)
        ]
    return out
