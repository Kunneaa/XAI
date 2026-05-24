"""Deterministic geometry template registry and vector engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from .registries import GEOMETRY_TEMPLATE_IDS


@dataclass(frozen=True)
class GeometryMatch:
    template_id: str
    confidence: float
    evidence: List[str]

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class Vector2:
    x: float
    y: float

    def add(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def sub(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def scale(self, factor: float) -> "Vector2":
        return Vector2(self.x * factor, self.y * factor)

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def unit(self) -> "Vector2":
        magnitude = self.norm()
        if magnitude <= 0:
            raise ValueError("zero_vector_has_no_direction")
        return self.scale(1.0 / magnitude)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class GeometryExecutionResult:
    ok: bool
    value: float | None
    unit: str | None
    components: dict | None
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "value": self.value,
            "unit": self.unit,
            "components": dict(self.components or {}),
            "issues": list(self.issues),
            "trace": dict(self.trace),
        }


K_COULOMB = 9e9


def match_geometry_templates(front_payload: dict) -> List[GeometryMatch]:
    """Return known geometry templates with textual evidence.

    This does not create coordinates by itself. It only records that a template
    is recoverable enough for deterministic geometry code.
    """

    text = str(front_payload.get("canonical_question") or "").lower()
    matches: List[GeometryMatch] = []
    if "midpoint" in text:
        matches.append(_match("point_on_midpoint", 0.86, ["midpoint"]))
    if "perpendicular bisector" in text:
        matches.append(_match("point_on_perpendicular_bisector", 0.86, ["perpendicular bisector"]))
    if "equilateral triangle" in text:
        matches.append(_match("equilateral_triangle_vertex", 0.84, ["equilateral triangle"]))
    if "isosceles right triangle" in text or "right isosceles triangle" in text:
        matches.append(_match("right_isosceles_triangle_vertex", 0.82, ["right isosceles triangle"]))
    if _has_three_side_triangle_evidence(text):
        matches.append(_match("triangle_sides", 0.84, ["three side lengths identify the geometry"]))
    if "square" in text and ("vertex" in text or "vertices" in text):
        matches.append(_match("square_vertex_field", 0.78, ["square", "vertices"]))
    if "straight line" in text or "collinear" in text:
        matches.append(_match("two_charges_collinear", 0.78, ["collinear/straight line"]))
    return matches


def build_template_coordinates(template_id: str, parameters: dict) -> dict:
    """Build local coordinates only for known templates.

    Qwen may select a template id, but this function owns all coordinates and
    parameter validation. Missing or impossible parameters raise ``ValueError``.
    """

    _ensure_template(template_id)
    if template_id == "point_on_midpoint":
        separation = _positive(parameters, "separation")
        return {"A": Vector2(-separation / 2.0, 0.0), "B": Vector2(separation / 2.0, 0.0), "P": Vector2(0.0, 0.0)}
    if template_id == "point_on_perpendicular_bisector":
        separation = _positive(parameters, "separation")
        height = _positive(parameters, "height")
        return {"A": Vector2(-separation / 2.0, 0.0), "B": Vector2(separation / 2.0, 0.0), "P": Vector2(0.0, height)}
    if template_id == "equilateral_triangle_vertex":
        side = _positive(parameters, "side")
        return {"A": Vector2(0.0, math.sqrt(3.0) * side / 2.0), "B": Vector2(-side / 2.0, 0.0), "C": Vector2(side / 2.0, 0.0)}
    if template_id == "right_isosceles_triangle_vertex":
        leg = _positive(parameters, "leg")
        return {"A": Vector2(0.0, 0.0), "B": Vector2(leg, 0.0), "C": Vector2(0.0, leg)}
    if template_id == "triangle_vertex":
        side_a = _positive(parameters, "side_a")
        side_b = _positive(parameters, "side_b")
        angle = math.radians(float(parameters.get("included_angle_deg")))
        if not math.isfinite(angle) or angle <= 0 or angle >= math.pi:
            raise ValueError("invalid_included_angle")
        return {"A": Vector2(0.0, 0.0), "B": Vector2(side_a, 0.0), "C": Vector2(side_b * math.cos(angle), side_b * math.sin(angle))}
    if template_id == "triangle_sides":
        return _triangle_coordinates_from_sides(_positive(parameters, "ab"), _positive(parameters, "ac"), _positive(parameters, "bc"))
    if template_id == "square_vertex_field":
        side = _positive(parameters, "side")
        return {"A": Vector2(0.0, side), "B": Vector2(side, side), "C": Vector2(side, 0.0), "D": Vector2(0.0, 0.0)}
    if template_id == "rectangle_vertex_field":
        width = _positive(parameters, "width")
        height = _positive(parameters, "height")
        return {"A": Vector2(0.0, height), "B": Vector2(width, height), "C": Vector2(width, 0.0), "D": Vector2(0.0, 0.0)}
    if template_id == "two_charges_collinear":
        separation = _positive(parameters, "separation")
        point = float(parameters.get("point_x", 0.0))
        return {"A": Vector2(0.0, 0.0), "B": Vector2(separation, 0.0), "P": Vector2(point, 0.0)}
    if template_id == "external_point_on_line":
        separation = _positive(parameters, "separation")
        distance_from_a = float(parameters.get("distance_from_a"))
        if not math.isfinite(distance_from_a):
            raise ValueError("invalid_distance_from_a")
        return {"A": Vector2(0.0, 0.0), "B": Vector2(separation, 0.0), "P": Vector2(distance_from_a, 0.0)}
    raise ValueError(f"geometry_template_not_implemented:{template_id}")


def execute_electric_field_superposition(template_id: str, parameters: dict, charges: list[dict], target_point: str = "P") -> GeometryExecutionResult:
    """Compute electric field vector by deterministic superposition."""

    try:
        coordinates = build_template_coordinates(template_id, parameters)
        target = coordinates[target_point]
        total = Vector2(0.0, 0.0)
        contributions = []
        for source in charges:
            point_id = source["point"]
            charge = float(source["charge_c"])
            source_point = coordinates[point_id]
            displacement = target.sub(source_point)
            radius = displacement.norm()
            if radius <= 0:
                raise ValueError("zero_distance_field_singularity")
            vector = displacement.unit().scale(K_COULOMB * charge / (radius * radius))
            total = total.add(vector)
            contributions.append({"point": point_id, "charge_c": charge, "vector": vector.to_dict(), "radius_m": radius})
        magnitude = total.norm()
        return GeometryExecutionResult(
            True,
            magnitude,
            "V/m",
            {"x": total.x, "y": total.y, "magnitude": magnitude},
            [],
            {"stage": "geometry_engine", "template_id": template_id, "coordinates": _coords_to_dict(coordinates), "contributions": contributions},
        )
    except Exception as exc:
        return GeometryExecutionResult(False, None, None, None, [f"geometry_error:{type(exc).__name__}:{exc}"], {"stage": "geometry_engine", "template_id": template_id})


def execute_coulomb_force_superposition(template_id: str, parameters: dict, sources: list[dict], target_charge: dict) -> GeometryExecutionResult:
    """Compute force vector on a target charge from source charges."""

    try:
        coordinates = build_template_coordinates(template_id, parameters)
        target_point = coordinates[target_charge["point"]]
        q_target = float(target_charge["charge_c"])
        total = Vector2(0.0, 0.0)
        contributions = []
        for source in sources:
            point_id = source["point"]
            q_source = float(source["charge_c"])
            source_point = coordinates[point_id]
            from_source_to_target = target_point.sub(source_point)
            radius = from_source_to_target.norm()
            if radius <= 0:
                raise ValueError("zero_distance_force_singularity")
            direction = from_source_to_target.unit()
            # Positive product repels along source->target; negative attracts.
            vector = direction.scale(K_COULOMB * q_source * q_target / (radius * radius))
            total = total.add(vector)
            contributions.append({"point": point_id, "charge_c": q_source, "vector": vector.to_dict(), "radius_m": radius})
        magnitude = total.norm()
        return GeometryExecutionResult(
            True,
            magnitude,
            "N",
            {"x": total.x, "y": total.y, "magnitude": magnitude},
            [],
            {"stage": "geometry_engine", "template_id": template_id, "coordinates": _coords_to_dict(coordinates), "contributions": contributions},
        )
    except Exception as exc:
        return GeometryExecutionResult(False, None, None, None, [f"geometry_error:{type(exc).__name__}:{exc}"], {"stage": "geometry_engine", "template_id": template_id})


def execute_coulomb_force_triangle_sides(
    *,
    ab: float,
    ac: float,
    bc: float,
    q_a: float,
    q_b: float,
    q_c: float,
    target_point: str = "C",
) -> GeometryExecutionResult:
    """Compute Coulomb force at one triangle vertex from all three side lengths.

    Coordinates are owned by deterministic code:
    A=(0,0), B=(AB,0), and C is reconstructed from AC/BC. No LLM-created
    diagram or coordinates are accepted.
    """

    try:
        coordinates = _triangle_coordinates_from_sides(ab, ac, bc)
        charges = {"A": q_a, "B": q_b, "C": q_c}
        if target_point not in charges:
            raise ValueError("unknown_triangle_target")
        sources = [
            {"point": point_id, "charge_c": charge}
            for point_id, charge in charges.items()
            if point_id != target_point
        ]
        return _force_from_coordinates(
            coordinates,
            sources,
            {"point": target_point, "charge_c": charges[target_point]},
            "triangle_sides",
        )
    except Exception as exc:
        return GeometryExecutionResult(False, None, None, None, [f"geometry_error:{type(exc).__name__}:{exc}"], {"stage": "geometry_engine", "template_id": "triangle_sides"})


def execute_electric_field_triangle_sides(
    *,
    ab: float,
    ac: float,
    bc: float,
    q_a: float,
    q_b: float,
    target_point: str = "C",
) -> GeometryExecutionResult:
    """Compute electric field at a triangle vertex from side lengths."""

    try:
        coordinates = _triangle_coordinates_from_sides(ab, ac, bc)
        if target_point != "C":
            raise ValueError("only_target_c_supported")
        return _field_from_coordinates(
            coordinates,
            [{"point": "A", "charge_c": q_a}, {"point": "B", "charge_c": q_b}],
            target_point,
            "triangle_sides",
        )
    except Exception as exc:
        return GeometryExecutionResult(False, None, None, None, [f"geometry_error:{type(exc).__name__}:{exc}"], {"stage": "geometry_engine", "template_id": "triangle_sides"})


def geometry_recoverability(front_payload: dict) -> dict:
    matches = match_geometry_templates(front_payload)
    return {
        "stage": "geometry_template_matcher",
        "recoverable": bool(matches),
        "matches": [match.to_dict() for match in matches],
        "known_template_ids": sorted(GEOMETRY_TEMPLATE_IDS),
    }


def _match(template_id: str, confidence: float, evidence: List[str]) -> GeometryMatch:
    _ensure_template(template_id)
    return GeometryMatch(template_id, confidence, evidence)


def _has_three_side_triangle_evidence(text: str) -> bool:
    lowered = text.lower()
    has_ab = ("ab =" in lowered or "a and b" in lowered or "points a and b" in lowered) and any(cue in lowered for cue in ["separated", "apart", "distance"])
    has_ac = "ac =" in lowered or "ca =" in lowered or "ac=" in lowered or "ca=" in lowered
    has_bc = "bc =" in lowered or "cb =" in lowered or "bc=" in lowered or "cb=" in lowered
    generic_two_charge_point = (
        any(cue in lowered for cue in ["separated by", "apart"])
        and any(cue in lowered for cue in ["from q1", "from charge q1", "from a"])
        and any(cue in lowered for cue in ["from q2", "from charge q2", "from b"])
    )
    return (has_ab and has_ac and has_bc) or generic_two_charge_point


def _ensure_template(template_id: str) -> None:
    if template_id not in GEOMETRY_TEMPLATE_IDS:
        raise ValueError(f"Unknown geometry template: {template_id}")


def _positive(parameters: dict, key: str) -> float:
    value = float(parameters.get(key))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid_positive_parameter:{key}")
    return value


def _coords_to_dict(coordinates: dict[str, Vector2]) -> dict:
    return {key: value.to_dict() for key, value in coordinates.items()}


def _triangle_coordinates_from_sides(ab: float, ac: float, bc: float) -> dict[str, Vector2]:
    ab = float(ab)
    ac = float(ac)
    bc = float(bc)
    if min(ab, ac, bc) <= 0:
        raise ValueError("triangle_side_not_positive")
    tolerance = 1e-12 * max(ab, ac, bc, 1.0)
    if ac + bc < ab - tolerance or ab + ac < bc - tolerance or ab + bc < ac - tolerance:
        raise ValueError("triangle_inequality_failed")
    if abs((ac + bc) - ab) <= tolerance:
        return {"A": Vector2(0.0, 0.0), "B": Vector2(ab, 0.0), "C": Vector2(ac, 0.0)}
    if abs((ab + bc) - ac) <= tolerance:
        return {"A": Vector2(0.0, 0.0), "B": Vector2(ab, 0.0), "C": Vector2(ac, 0.0)}
    if abs((ab + ac) - bc) <= tolerance:
        return {"A": Vector2(0.0, 0.0), "B": Vector2(ab, 0.0), "C": Vector2(-ac, 0.0)}
    x_c = ((ac * ac) + (ab * ab) - (bc * bc)) / (2.0 * ab)
    y_sq = (ac * ac) - (x_c * x_c)
    if y_sq < -1e-12:
        raise ValueError("triangle_geometry_not_recoverable")
    y_c = math.sqrt(max(0.0, y_sq))
    return {"A": Vector2(0.0, 0.0), "B": Vector2(ab, 0.0), "C": Vector2(x_c, y_c)}


def _force_from_coordinates(coordinates: dict[str, Vector2], sources: list[dict], target_charge: dict, template_id: str) -> GeometryExecutionResult:
    target_point = coordinates[target_charge["point"]]
    q_target = float(target_charge["charge_c"])
    total = Vector2(0.0, 0.0)
    contributions = []
    for source in sources:
        point_id = source["point"]
        q_source = float(source["charge_c"])
        source_point = coordinates[point_id]
        from_source_to_target = target_point.sub(source_point)
        radius = from_source_to_target.norm()
        if radius <= 0:
            raise ValueError("zero_distance_force_singularity")
        vector = from_source_to_target.unit().scale(K_COULOMB * q_source * q_target / (radius * radius))
        total = total.add(vector)
        contributions.append({"point": point_id, "charge_c": q_source, "vector": vector.to_dict(), "radius_m": radius})
    magnitude = total.norm()
    return GeometryExecutionResult(
        True,
        magnitude,
        "N",
        {"x": total.x, "y": total.y, "magnitude": magnitude},
        [],
        {"stage": "geometry_engine", "template_id": template_id, "coordinates": _coords_to_dict(coordinates), "contributions": contributions},
    )


def _field_from_coordinates(coordinates: dict[str, Vector2], sources: list[dict], target_point_id: str, template_id: str) -> GeometryExecutionResult:
    target = coordinates[target_point_id]
    total = Vector2(0.0, 0.0)
    contributions = []
    for source in sources:
        point_id = source["point"]
        charge = float(source["charge_c"])
        source_point = coordinates[point_id]
        displacement = target.sub(source_point)
        radius = displacement.norm()
        if radius <= 0:
            raise ValueError("zero_distance_field_singularity")
        vector = displacement.unit().scale(K_COULOMB * charge / (radius * radius))
        total = total.add(vector)
        contributions.append({"point": point_id, "charge_c": charge, "vector": vector.to_dict(), "radius_m": radius})
    magnitude = total.norm()
    return GeometryExecutionResult(
        True,
        magnitude,
        "V/m",
        {"x": total.x, "y": total.y, "magnitude": magnitude},
        [],
        {"stage": "geometry_engine", "template_id": template_id, "coordinates": _coords_to_dict(coordinates), "contributions": contributions},
    )
