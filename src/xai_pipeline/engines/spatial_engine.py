"""Deterministic geometry template registry and vector engine."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List

from .equation_engine import SolverResult
from ..frontend.canonical import canonical_point, canonical_side_key, first_triangle_context, right_angle_point
from ..knowledge.registries import FORMULA_REGISTRY, GEOMETRY_TEMPLATE_IDS
from ..knowledge.units import unit_info


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
    if "equidistant" in text and ("away from" in text or "line segment" in text):
        matches.append(_match("point_on_perpendicular_bisector", 0.82, ["equidistant point offset from source segment"]))
    if re.search(r"\b(?:equilateral|regular)\s+triangle\b", text):
        matches.append(_match("equilateral_triangle_vertex", 0.84, ["equilateral/regular triangle"]))
    if _has_right_isosceles_text(text):
        matches.append(_match("right_isosceles_triangle_vertex", 0.82, ["right isosceles triangle"]))
    if _has_right_triangle_evidence(text):
        matches.append(_match("triangle_sides", 0.82, ["right triangle side completion"]))
    if _has_three_side_triangle_evidence(text):
        matches.append(_match("triangle_sides", 0.84, ["three side lengths identify the geometry"]))
    if "square" in text and ("vertex" in text or "vertices" in text or "side length" in text):
        matches.append(_match("square_vertex_field", 0.78, ["square", "vertices"]))
    if "straight line" in text or "collinear" in text:
        matches.append(_match("two_charges_collinear", 0.78, ["collinear/straight line"]))
    if "line segment" in text or re.search(r"\bline\s+connecting\b", text):
        matches.append(_match("two_charges_collinear", 0.76, ["line segment/line connecting sources"]))
    if "direction" in text and _direction_probe_geometry_cue(text) and "distances to" in text and "separated" in text:
        matches.append(_match("two_charges_collinear", 0.76, ["direction query with source distances and separation"]))
    return matches


def build_template_coordinates(template_id: str, parameters: dict) -> dict:
    """Build local coordinates only for known templates.

    The fine-tuned LLM may select a template id, but this function owns all
    coordinates and parameter validation. Missing or impossible parameters raise
    ``ValueError``.
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
    q_a: float | None = None,
    q_b: float | None = None,
    q_c: float | None = None,
    target_point: str = "C",
) -> GeometryExecutionResult:
    """Compute electric field at a triangle vertex from side lengths."""

    try:
        coordinates = _triangle_coordinates_from_sides(ab, ac, bc)
        if target_point not in coordinates:
            raise ValueError("unknown_triangle_target")
        charges = {"A": q_a, "B": q_b, "C": q_c}
        sources = [
            {"point": point_id, "charge_c": charge}
            for point_id, charge in charges.items()
            if point_id != target_point and charge is not None
        ]
        if not sources:
            raise ValueError("missing_source_charges")
        return _field_from_coordinates(
            coordinates,
            sources,
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


def solve_spatial_from_front(front_payload: dict, route_result) -> SolverResult:
    """Dispatch recoverable geometry facts into deterministic vector execution."""

    if getattr(route_result, "task_type", "") not in {"electric_field_point", "coulomb_force"}:
        return _spatial_unsolved("route_not_spatial", getattr(route_result, "task_type", "unknown"))
    matches = match_geometry_templates(front_payload)
    target_dimension = "force" if getattr(route_result, "task_type", "") == "coulomb_force" else "electric_field"

    triangle_context = _triangle_context(front_payload)
    triangle_lengths, geometry_audit = _complete_right_triangle_lengths(
        front_payload,
        _triangle_lengths(front_payload, triangle_context),
        triangle_context,
    )
    charge_by_point = _charges_by_point(front_payload, triangle_context)
    symmetry_zero = _solve_symmetry_zero_from_front(front_payload, route_result, target_dimension)
    if symmetry_zero is not None:
        return symmetry_zero
    zero_line = _solve_zero_field_line_from_front(front_payload, route_result)
    if zero_line is not None:
        return zero_line
    if target_dimension == "electric_field":
        centroid_unknown = _solve_equilateral_centroid_cancel_unknown_charge(front_payload, route_result)
        if centroid_unknown is not None:
            return centroid_unknown
        symbolic_perp = _solve_symbolic_perpendicular_bisector_equal_charge_field(front_payload, route_result)
        if symbolic_perp is not None:
            return symbolic_perp
        rectangle_balance = _solve_rectangle_field_balance_unknown_charge(front_payload, route_result)
        if rectangle_balance is not None:
            return rectangle_balance
        collinear_multi = _solve_collinear_multi_charge_field_from_front(front_payload, route_result)
        if collinear_multi is not None:
            return collinear_multi
        zero_sum = _solve_two_charge_zero_field_with_sum(front_payload, route_result)
        if zero_sum is not None:
            return zero_sum
        midpoint_inverse = _solve_point_charge_midpoint_field_from_front(front_payload, route_result)
        if midpoint_inverse is not None:
            return midpoint_inverse
    if target_dimension == "force":
        direction_result = _solve_collinear_two_charge_direction_from_front(front_payload, route_result)
        if direction_result is not None:
            return direction_result
        equal_line = _solve_three_charge_equally_spaced_line_from_front(front_payload, route_result)
        if equal_line is not None:
            return equal_line
    if target_dimension == "force" and any(match.template_id == "point_on_perpendicular_bisector" for match in matches):
        perpendicular_force = _solve_perpendicular_bisector_force_from_front(front_payload, route_result)
        if perpendicular_force is not None:
            return perpendicular_force
    if target_dimension == "electric_field" and any(match.template_id == "point_on_perpendicular_bisector" for match in matches):
        perpendicular_field = _solve_perpendicular_bisector_field_from_front(front_payload, route_result)
        if perpendicular_field is not None:
            return perpendicular_field
    if any(match.template_id == "two_charges_collinear" for match in matches):
        collinear = _solve_collinear_two_source_vector_from_front(front_payload, route_result, target_dimension)
        if collinear is not None:
            return collinear
    collinear = _solve_collinear_two_source_vector_from_front(front_payload, route_result, target_dimension)
    if collinear is not None:
        return collinear
    if not any(match.template_id == "point_on_perpendicular_bisector" for match in matches):
        distance_triangle = _solve_two_source_target_distance_triangle_from_front(front_payload, route_result, target_dimension)
        if distance_triangle is not None:
            return distance_triangle
    if target_dimension == "force" and any(match.template_id == "point_on_perpendicular_bisector" for match in matches):
        perpendicular_force = _solve_perpendicular_bisector_force_from_front(front_payload, route_result)
        if perpendicular_force is not None:
            return perpendicular_force
    if target_dimension == "electric_field" and any(match.template_id == "point_on_perpendicular_bisector" for match in matches):
        perpendicular_field = _solve_perpendicular_bisector_field_from_front(front_payload, route_result)
        if perpendicular_field is not None:
            return perpendicular_field
    if target_dimension == "force" and any(match.template_id == "point_on_midpoint" for match in matches):
        midpoint_force = _solve_midpoint_force_from_front(front_payload, route_result)
        if midpoint_force is not None:
            return midpoint_force
    if target_dimension == "force" and any(match.template_id == "right_isosceles_triangle_vertex" for match in matches):
        right_iso_force = _solve_right_isosceles_force_from_front(front_payload, route_result)
        if right_iso_force is not None:
            return right_iso_force
        symbolic_right_iso_force = _solve_symbolic_right_isosceles_force(front_payload, route_result)
        if symbolic_right_iso_force is not None:
            return symbolic_right_iso_force
    if target_dimension == "electric_field":
        if any(match.template_id == "right_isosceles_triangle_vertex" for match in matches):
            right_iso_field = _solve_right_isosceles_field_from_front(front_payload, route_result)
            if right_iso_field is not None:
                return right_iso_field
        symbolic_result = _solve_symbolic_right_isosceles_altitude_field(front_payload, route_result, triangle_context)
        if symbolic_result is not None:
            return symbolic_result
        altitude_field = _solve_triangle_altitude_foot_field_from_front(front_payload, route_result, triangle_context)
        if altitude_field is not None:
            return altitude_field
    if {"ab", "ac", "bc"} <= set(triangle_lengths) and len(charge_by_point) >= 2:
        if target_dimension == "force" and len(charge_by_point) >= 3:
            target_point = _target_charge_point(front_payload, charge_by_point, triangle_context)
            if target_point is not None:
                result = execute_coulomb_force_triangle_sides(
                    ab=triangle_lengths["ab"],
                    ac=triangle_lengths["ac"],
                    bc=triangle_lengths["bc"],
                    q_a=charge_by_point.get("A", 0.0),
                    q_b=charge_by_point.get("B", 0.0),
                    q_c=charge_by_point.get("C", 0.0),
                    target_point=target_point,
                )
                return _solver_from_geometry(
                    result,
                    "coulomb_force_triangle_sides",
                    min(0.76, float(route_result.confidence)),
                    _spatial_source_facts(front_payload),
                    {**geometry_audit, "target_point_policy": "target_charge_from_goal_text"},
                    medium_scale=_medium_field_scale(front_payload),
                )
        if target_dimension == "electric_field" and len(charge_by_point) >= 2:
            target_point = _target_field_point(front_payload, charge_by_point, triangle_context)
            if target_point is None:
                target_point = _uncharged_triangle_vertex(charge_by_point)
            if target_point is None:
                target_point = _target_charge_point(front_payload, charge_by_point, triangle_context)
            if target_point is None:
                return _spatial_unsolved(
                    "triangle_field_target_not_grounded",
                    getattr(route_result, "task_type", "unknown"),
                    {"geometry_audit": geometry_audit, "charge_points": sorted(charge_by_point)},
                )
            result = execute_electric_field_triangle_sides(
                ab=triangle_lengths["ab"],
                ac=triangle_lengths["ac"],
                bc=triangle_lengths["bc"],
                q_a=charge_by_point.get("A"),
                q_b=charge_by_point.get("B"),
                q_c=charge_by_point.get("C"),
                target_point=target_point,
            )
            return _solver_from_geometry(
                result,
                "electric_field_two_charge_triangle_sides",
                min(0.76, float(route_result.confidence)),
                _spatial_source_facts(front_payload),
                {**geometry_audit, "target_point": target_point, "target_point_policy": "field_target_from_goal_or_uncharged_vertex"},
                medium_scale=_medium_field_scale(front_payload),
            )

    if target_dimension == "force" and any(match.template_id == "equilateral_triangle_vertex" for match in matches):
        center_force = _solve_equilateral_center_force_from_front(front_payload, route_result)
        if center_force is not None:
            return center_force
        equilateral = _solve_equilateral_force_from_front(front_payload, route_result)
        if equilateral is not None:
            return equilateral

    if target_dimension == "electric_field" and any(match.template_id == "square_vertex_field" for match in matches):
        square_cancel = _solve_square_cancel_charge_from_front(front_payload, route_result)
        if square_cancel is not None:
            return square_cancel
        square_field = _solve_square_three_vertex_field_from_front(front_payload, route_result)
        if square_field is not None:
            return square_field

    if target_dimension == "electric_field" and any(match.template_id == "point_on_midpoint" for match in matches):
        separation = _length_by_symbol(front_payload, "AB") or _first_length(front_payload)
        charges = _ordered_charges(front_payload)
        if separation and len(charges) >= 2:
            result = execute_electric_field_superposition(
                "point_on_midpoint",
                {"separation": separation},
                [{"point": "A", "charge_c": charges[0]}, {"point": "B", "charge_c": charges[1]}],
            )
            return _solver_from_geometry(
                result,
                "electric_field_two_charge_superposition",
                min(0.74, float(route_result.confidence)),
                _spatial_source_facts(front_payload),
                medium_scale=_medium_field_scale(front_payload),
            )

    if target_dimension == "electric_field" and any(match.template_id == "equilateral_triangle_vertex" for match in matches):
        side = _first_length(front_payload)
        triangle_context = _triangle_context(front_payload)
        charge_by_point = _charges_by_point(front_payload, triangle_context)
        target_point = _target_field_point(front_payload, charge_by_point, triangle_context)
        if target_point is None:
            target_point = _uncharged_triangle_vertex(charge_by_point, triangle_context)
        if side and target_point is not None and len(charge_by_point) >= 2:
            sources = [
                {"point": point_id, "charge_c": charge}
                for point_id, charge in charge_by_point.items()
                if point_id != target_point
            ]
            result = execute_electric_field_superposition(
                "equilateral_triangle_vertex",
                {"side": side},
                sources,
                target_point=target_point,
            )
            return _solver_from_geometry(
                result,
                "electric_field_equilateral_vertex",
                min(0.72, float(route_result.confidence)),
                _spatial_source_facts(front_payload),
                {"target_point": target_point, "target_point_policy": "field_target_from_goal_or_uncharged_vertex"},
                medium_scale=_medium_field_scale(front_payload),
            )
        charges = _ordered_charges(front_payload)
        if side and len(charges) >= 2 and _unlabeled_target_vertex_context(front_payload):
            result = execute_electric_field_superposition(
                "equilateral_triangle_vertex",
                {"side": side},
                [{"point": "B", "charge_c": charges[0]}, {"point": "C", "charge_c": charges[1]}],
                target_point="A",
            )
            return _solver_from_geometry(
                result,
                "electric_field_equilateral_vertex",
                min(0.72, float(route_result.confidence)),
                _spatial_source_facts(front_payload),
                {"target_point": "A", "target_point_policy": "unlabeled_remaining_vertex_template_frame"},
                medium_scale=_medium_field_scale(front_payload),
            )

    return _spatial_unsolved(
        "spatial_geometry_not_executable_from_front",
        getattr(route_result, "task_type", "unknown"),
        {"matches": [match.to_dict() for match in matches], "geometry_audit": geometry_audit},
    )


def _solve_symbolic_right_isosceles_altitude_field(front_payload: dict, route_result, triangle_context: dict | None) -> SolverResult | None:
    case = _right_isosceles_altitude_hypotenuse_case(front_payload, triangle_context)
    if case is None:
        return None
    formula_id = "electric_field_symbolic_superposition"
    spec = FORMULA_REGISTRY[formula_id]
    coefficient = case["result_coefficient"]
    side_symbol = case["side_symbol"]
    charge_symbol = case["charge_symbol"]
    if coefficient == "2*sqrt(2)":
        expression = f"E_H = 2√2 k {charge_symbol}/{side_symbol}^2"
    else:
        expression = f"E_H = {coefficient} k {charge_symbol}/{side_symbol}^2"
    answer = f"{expression}, directed {case['direction_text']}"
    return SolverResult(
        solved=True,
        answer=answer,
        value=answer,
        unit=spec.target_unit,
        formula_id=formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise, "Point-charge electric fields add by vector superposition."],
        trace={
            "stage": "symbolic_spatial_vector_engine",
            "formula_id": formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "compiled_geometry_case": case["geometry_case"],
            "field_law": "E_i(P) = k*q_i*(P-r_i)/|P-r_i|^3",
            "aggregation": "symbolic_vector_sum",
            "symbolic_geometry": case["geometry"],
            "symbolic_components": case["components"],
            "symbolic_vector_terms": case["vector_terms"],
            "source": case["source_facts"],
            "constants": {"k": "1/(4*pi*epsilon0)"},
            "geometry_audit": case["audit"],
            "binding_audit": {
                "policy": "normalize_geometry_then_apply_point_charge_superposition",
                "template_id": "right_isosceles_triangle_vertex",
            },
        },
        confidence=min(0.72, float(route_result.confidence)),
    )


def _solve_triangle_altitude_foot_field_from_front(front_payload: dict, route_result, triangle_context: dict | None) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "altitude" not in text or "hypotenuse" not in text:
        return None
    if not _has_right_triangle_evidence(text):
        return None
    lengths, geometry_audit = _complete_right_triangle_lengths(
        front_payload,
        _triangle_lengths(front_payload, triangle_context),
        triangle_context,
    )
    if not {"ab", "ac", "bc"} <= set(lengths):
        return None
    right = _canonical_point(_right_angle_point(text), triangle_context) or "A"
    if right != "A":
        return None
    charges = _ordered_charges(front_payload)
    if len(charges) == 1 and re.search(r"\b(?:identical|equal|same)\s+(?:point\s+)?charges\b", text):
        charges = [charges[0], charges[0], charges[0]]
    if len(charges) < 3:
        charge_by_point = _charges_by_point(front_payload, triangle_context)
        if len(charge_by_point) >= 3:
            charges = [charge_by_point["A"], charge_by_point["B"], charge_by_point["C"]]
    if len(charges) < 3:
        return None
    coordinates = _triangle_coordinates_from_sides(lengths["ab"], lengths["ac"], lengths["bc"])
    a = coordinates["A"]
    b = coordinates["B"]
    c = coordinates["C"]
    bc = c.sub(b)
    denom = bc.x * bc.x + bc.y * bc.y
    if denom <= 0:
        return None
    t = ((a.x - b.x) * bc.x + (a.y - b.y) * bc.y) / denom
    coordinates["H"] = Vector2(b.x + t * bc.x, b.y + t * bc.y)
    result = _field_from_coordinates(
        coordinates,
        [
            {"point": "A", "charge_c": charges[0]},
            {"point": "B", "charge_c": charges[1]},
            {"point": "C", "charge_c": charges[2]},
        ],
        "H",
        "right_triangle_altitude_foot",
    )
    return _solver_from_geometry(
        result,
        "electric_field_symbolic_superposition",
        min(0.72, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        {
            **geometry_audit,
            "stage": "right_triangle_altitude_foot_field_completion",
            "status": "constructed_projection_to_hypotenuse",
            "target_point": "H",
            "source_points": ["A", "B", "C"],
        },
        medium_scale=_medium_field_scale(front_payload),
    )


def _solve_symbolic_perpendicular_bisector_equal_charge_field(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "")
    lowered = text.lower()
    if "perpendicular bisector" not in lowered or "electric field" not in lowered:
        return None
    charge_relation = re.search(
        r"\b(?P<source_1>[qQ][A-Za-z0-9_′']*)\s*=\s*(?P<source_2>[qQ][A-Za-z0-9_′']*)\s*=\s*"
        r"(?P<base>[qQ][A-Za-z0-9_′']*)\b",
        text,
    )
    if not charge_relation:
        return None
    side_relation = re.search(r"\b(?P<side>[A-Za-z]{2})\s*=\s*2\s*\*?\s*(?P<half>[A-Za-z][A-Za-z0-9_]*)\b", text)
    if not side_relation:
        return None
    charge_symbol = charge_relation.group("base")
    side_symbol = side_relation.group("half")
    source_side = side_relation.group("side").upper()
    target_point = _symbolic_target_point_label(text) or "P"
    height_symbol = _symbolic_distance_symbol(text) or "h"
    spec = FORMULA_REGISTRY["electric_field_symbolic_superposition"]
    if "maximum" in lowered:
        answer = f"h = {side_symbol}/√2; E_max = 4k{charge_symbol}/(3√3 {side_symbol}^2)"
        policy = "maximize_equal_charge_perpendicular_bisector_field"
        expression = "E(h)=2*k*q*h/(a^2+h^2)^(3/2); dE/dh=0"
    else:
        answer = f"E_{target_point} = 2k|{charge_symbol}|{height_symbol}/({side_symbol}^2 + {height_symbol}^2)^(3/2)"
        policy = "equal_charge_perpendicular_bisector_field_expression"
        expression = "E(h)=2*k*q*h/(a^2+h^2)^(3/2)"
    return SolverResult(
        True,
        answer,
        answer,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise, "By symmetry, horizontal components cancel and perpendicular components add."],
        {
            "stage": "symbolic_spatial_vector_engine",
            "formula_id": spec.formula_id,
            "expression": expression,
            "target_dimension": spec.target_dimension,
            "symbolic_geometry": {
                "template_id": "point_on_perpendicular_bisector",
                "source_points": {source_side[0]: f"(-{side_symbol},0)", source_side[1]: f"({side_symbol},0)"},
                "target_point": {target_point: f"(0,{height_symbol})"},
                "source_separation": f"2{side_symbol}",
            },
            "source": _symbolic_source_facts(front_payload),
            "constants": {"k": "Coulomb constant"},
            "geometry_audit": {"policy": policy, "status": "proved_by_symmetry_and_vector_components"},
            "binding_audit": {"policy": "symbolic_perpendicular_bisector_equal_source_charges"},
        },
        min(0.72, float(route_result.confidence)),
    )


def _solve_rectangle_field_balance_unknown_charge(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "")
    lowered = text.lower()
    if "rectangle" not in lowered or not re.search(r"\be\s*2\s*=\s*(?:vector\s*)?e\s*13\b", lowered):
        return None
    target_text = _target_text(front_payload)
    if "q1" not in target_text and "q3" not in target_text:
        return None
    height = _length_by_symbol(front_payload, "AD")
    width = _length_by_symbol(front_payload, "AB")
    q2_quantity = next(
        (
            quantity
            for quantity in front_payload.get("quantities") or []
            if quantity.get("dimension") == "charge" and str(quantity.get("symbol") or "").lower() == "q2"
        ),
        None,
    )
    if height is None or width is None or q2_quantity is None:
        return None
    diagonal = math.hypot(width, height)
    if diagonal <= 0:
        return None
    q2 = _si_value(q2_quantity)
    if "q1" in target_text:
        value = q2 * (height ** 3) / (diagonal ** 3)
        unknown = "q1"
        component = "vertical"
    else:
        value = q2 * (width ** 3) / (diagonal ** 3)
        unknown = "q3"
        component = "horizontal"
    spec = FORMULA_REGISTRY["electric_field_balance_unknown_charge"]
    return SolverResult(
        True,
        f"{value:.6g} {spec.target_unit}",
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "spatial_vector_balance_engine",
            "formula_id": spec.formula_id,
            "expression": "E2(D)=E1(D)+E3(D) component-wise in rectangle ABCD",
            "target_dimension": spec.target_dimension,
            "inputs": {"AD": height, "AB": width, "q2": _quantity_trace(q2_quantity)},
            "constants": {"diagonal": diagonal},
            "geometry_audit": {
                "template_id": "rectangle_vertex_field",
                "target_field_point": "D",
                "source_points": {"q1": "A", "q2": "B", "q3": "C"},
                "component_solved": component,
            },
            "binding_audit": {"policy": "rectangle_vector_balance_E2_equals_E13"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.72, float(route_result.confidence)),
    )


def _solve_two_charge_zero_field_with_sum(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "")
    lowered = text.lower()
    if "electric field" not in lowered or not re.search(r"\be\s*=\s*0\b|field[^.]{0,30}\bzero\b", lowered):
        return None
    sum_value = _charge_sum_value(text)
    if sum_value is None:
        return None
    distances = _two_source_distance_facts(front_payload)
    if distances is None:
        return None
    r1, r2 = distances["d1"], distances["d2"]
    if r1 <= 0 or r2 <= 0:
        return None
    # Signed collinear zero field: q1/r1^2 + q2/r2^2 = 0 and q1 + q2 = S.
    ratio = -(r2 * r2) / (r1 * r1)
    denominator = 1.0 + ratio
    if abs(denominator) <= 1e-15:
        return None
    q1 = sum_value / denominator
    q2 = sum_value - q1
    target_text = _target_text(front_payload)
    value = q2 if "q2" in target_text else q1
    unknown = "q2" if "q2" in target_text else "q1"
    spec = FORMULA_REGISTRY["electric_field_balance_unknown_charge"]
    return SolverResult(
        True,
        f"{value:.6g} {spec.target_unit}",
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "spatial_vector_balance_engine",
            "formula_id": spec.formula_id,
            "expression": "q1/r1^2 + q2/r2^2 = 0; q1 + q2 = S",
            "target_dimension": spec.target_dimension,
            "inputs": {"charge_sum_c": sum_value, "r1_m": r1, "r2_m": r2},
            "constants": {},
            "geometry_audit": {"template_id": "two_charges_collinear", "distance_binding": distances["audit"]},
            "binding_audit": {"policy": "two_unknown_charges_from_zero_field_and_sum", "unknown": unknown},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.72, float(route_result.confidence)),
    )


def _solve_equilateral_centroid_cancel_unknown_charge(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    target_text = _target_text(front_payload)
    if "equilateral triangle" not in text or "centroid" not in text:
        return None
    if "electric field" not in text and "field strength" not in text:
        return None
    if not re.search(r"\b(?:zero|vanish|cancel|is\s+0)\b|e\s*=\s*0", text):
        return None
    if not _target_asks_unknown_charge(target_text, text):
        return None
    charges = [
        quantity
        for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "charge" and unit_info(quantity.get("unit") or "") is not None
    ]
    if len(charges) < 2:
        return None
    q1 = _si_value(charges[0])
    q2 = _si_value(charges[1])
    if not math.isclose(q1, q2, rel_tol=1e-9, abs_tol=max(1e-18, abs(q1) * 1e-12, abs(q2) * 1e-12)):
        return None
    spec = FORMULA_REGISTRY["electric_field_centroid_equilateral_unknown_charge"]
    value = q1
    return SolverResult(
        True,
        f"{value:.6g} {spec.target_unit}",
        value,
        spec.target_unit,
        spec.formula_id,
        spec.principle_id,
        [spec.premise],
        {
            "stage": "symmetry_balance_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {"q1": _quantity_trace(charges[0]), "q2": _quantity_trace(charges[1])},
            "constants": {},
            "geometry_audit": {
                "template_id": "equilateral_triangle_vertex",
                "target_point": "centroid",
                "condition": "net electric field is zero",
                "unknown_charge": "third_vertex_charge",
            },
            "binding_audit": {"policy": "equilateral_centroid_zero_field_requires_equal_vertex_charges"},
            "attempted_formula_ids": [spec.formula_id],
        },
        min(0.74, float(route_result.confidence)),
    )


def _charge_sum_value(text: str) -> float | None:
    match = re.search(
        r"\bq1\s*\+\s*q2\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?:x|×|\*)\s*10\s*\^?\s*(?P<exp>[-+]?\d+)\s*(?P<unit>μc|uc|nc|pc|c)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group("value")) * (10 ** int(match.group("exp")))
    unit = match.group("unit").replace("μ", "u").lower()
    scale = {"c": 1.0, "uc": 1e-6, "nc": 1e-9, "pc": 1e-12}.get(unit)
    return value * scale if scale is not None else None


def _solve_collinear_multi_charge_field_from_front(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "collinear" not in text and "same line" not in text:
        return None
    charges = _ordered_charges(front_payload)
    if len(charges) < 3:
        return None
    spacing = _equal_line_spacing(front_payload)
    if spacing is None:
        return None
    target_text = _target_text(front_payload)
    target_point = "M" if re.search(r"\bpoint\s+m\b|\bat\s+m\b", target_text) else "N" if re.search(r"\bpoint\s+n\b|\bat\s+n\b", target_text) else None
    if target_point is None:
        return None
    coordinates = {
        "M": Vector2(-spacing, 0.0),
        "A": Vector2(0.0, 0.0),
        "B": Vector2(spacing, 0.0),
        "C": Vector2(2.0 * spacing, 0.0),
        "N": Vector2(3.0 * spacing, 0.0),
    }
    sources = [
        {"point": point, "charge_c": charge}
        for point, charge in zip(["A", "B", "C"], charges[:3])
    ]
    result = _field_from_coordinates(coordinates, sources, target_point, "collinear_equal_spacing_multi_charge")
    return _solver_from_geometry(
        result,
        "electric_field_two_charge_superposition",
        min(0.74, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        {
            "stage": "collinear_multi_charge_field_completion",
            "status": "constructed_equal_spacing_line_from_chain_equalities",
            "spacing_m": spacing,
            "target_point": target_point,
            "source_points": ["A", "B", "C"],
        },
        medium_scale=_medium_field_scale(front_payload),
    )


def _equal_line_spacing(front_payload: dict) -> float | None:
    text = str(front_payload.get("canonical_question") or "")
    if not re.search(r"\bMA\s*=\s*AB\s*=\s*BC\s*=\s*CN\s*=", text, flags=re.IGNORECASE):
        return None
    lengths = [
        quantity
        for quantity in front_payload.get("quantities") or []
        if quantity.get("dimension") == "length" and unit_info(quantity.get("unit") or "") is not None
    ]
    if not lengths:
        return None
    return _si_value(lengths[0])


def _solve_symmetry_zero_from_front(front_payload: dict, route_result, target_dimension: str) -> SolverResult | None:
    case = _symmetry_zero_case(front_payload, target_dimension)
    if case is None:
        return None
    spec = FORMULA_REGISTRY[case["formula_id"]]
    unit = spec.target_unit
    answer = f"0 {unit}"
    return SolverResult(
        solved=True,
        answer=answer,
        value=0.0,
        unit=unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "symmetry_reduction_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "geometry_engine": {
                "template_id": case["template_id"],
                "symmetry_group": case["symmetry_group"],
                "components": {"x": 0.0, "y": 0.0, "magnitude": 0.0},
                "value": 0.0,
            },
            "components": [],
            "source": _spatial_source_facts(front_payload),
            "constants": {"k": K_COULOMB},
            "geometry_audit": case["audit"],
            "binding_audit": {
                "policy": "symbolic_symmetry_reduction",
                "template_id": case["template_id"],
            },
        },
        confidence=min(0.72, float(route_result.confidence)),
    )


def _symmetry_zero_case(front_payload: dict, target_dimension: str) -> dict | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["center", "centre", "centroid", "intersection point of the diagonals", "diagonals"]):
        if "midpoint" not in text:
            return None
        if not re.search(r"\bequal\s+magnitude\b|\bsame\s+sign\b|\bidentical\b|\bequal\s+charges\b", text):
            return None
        if target_dimension == "force":
            formula_id = "symmetric_zero_force"
        elif target_dimension == "electric_field":
            formula_id = "electric_field_symmetric_zero"
        else:
            return None
        return {
            "formula_id": formula_id,
            "template_id": "point_on_midpoint",
            "symmetry_group": "C2_line_midpoint",
            "audit": {
                "stage": "symbolic_symmetry_reduction",
                "status": "zero_by_equal_source_midpoint_symmetry",
                "symmetry_group": "C2_line_midpoint",
                "target_point": "midpoint",
                "shape": "line_segment",
                "charge_condition": "equal same-sign source charges",
            },
        }
    equal_charges = bool(
        re.search(r"\b(?:identical|equal|same)\s+(?:positive\s+|negative\s+)?charges\b", text)
        or re.search(r"\b(?:all|three|four)\s+(?:identical|equal)\b", text)
        or re.search(r"\bsame\s+magnitude\b", text)
    )
    if not equal_charges:
        return None
    if "square" in text:
        template_id = "square_vertex_field"
        symmetry_group = "D4_square_center"
    elif "equilateral triangle" in text or "regular triangle" in text:
        template_id = "equilateral_triangle_vertex"
        symmetry_group = "C3_equilateral_centroid"
    else:
        return None
    if target_dimension == "force":
        if not _direction_probe_geometry_cue(text) and not re.search(r"\bcharge\b[^.?;]{0,50}\b(?:center|centre|centroid)\b", text):
            return None
        formula_id = "symmetric_zero_force"
    elif target_dimension == "electric_field":
        formula_id = "electric_field_symmetric_zero"
    else:
        return None
    return {
        "formula_id": formula_id,
        "template_id": template_id,
        "symmetry_group": symmetry_group,
        "audit": {
            "stage": "symbolic_symmetry_reduction",
            "status": "zero_by_equal_source_symmetry",
            "symmetry_group": symmetry_group,
            "target_point": "center_or_centroid",
            "shape": "square" if template_id == "square_vertex_field" else "equilateral_triangle",
            "charge_condition": "equal source charges",
        },
    }


def _right_isosceles_altitude_hypotenuse_case(front_payload: dict, triangle_context: dict | None) -> dict | None:
    text = str(front_payload.get("canonical_question") or "")
    lowered = text.lower()
    if not _has_right_isosceles_text(lowered):
        return None
    if "hypotenuse" not in lowered or "altitude" not in lowered:
        return None
    if not any((goal.get("dimension") == "electric_field") for goal in front_payload.get("goals") or []):
        return None
    if triangle_context is None:
        return None

    right_original = right_angle_point(text) or (triangle_context.get("right_angle_at") if triangle_context else None)
    right_point = _canonical_point(right_original, triangle_context)
    if right_point != "A":
        return None

    side_match = re.search(r"\b([A-Z]{2})\s*=\s*([A-Z]{2})\s*=\s*([A-Za-z][A-Za-z0-9_]*)\b", text)
    if not side_match:
        return None
    side_keys = {
        _canonical_side_key(side_match.group(1), triangle_context),
        _canonical_side_key(side_match.group(2), triangle_context),
    }
    if side_keys != {"ab", "ac"}:
        return None
    side_symbol = side_match.group(3)

    charge_symbol, factors = _symbolic_charge_factors(front_payload, triangle_context)
    if not charge_symbol or set(factors) != {"A", "B", "C"}:
        return None

    fa, fb, fc = factors["A"], factors["B"], factors["C"]
    ex_factor = fa - fb + fc
    ey_factor = fa + fb - fc
    if ex_factor == 0 and ey_factor == 0:
        return None
    coefficient = _field_magnitude_coefficient(ex_factor, ey_factor)
    direction_text = _symbolic_direction_text(ex_factor, ey_factor, triangle_context)
    original_by_canonical = triangle_context.get("original_by_canonical") or {}
    return {
        "side_symbol": side_symbol,
        "charge_symbol": charge_symbol,
        "result_coefficient": coefficient,
        "direction_text": direction_text,
        "geometry_case": "right_isosceles_altitude_to_hypotenuse",
        "geometry": {
            "template_id": "right_isosceles_triangle_vertex",
            "right_angle_at": original_by_canonical.get("A", "A"),
            "equal_legs": [side_match.group(1), side_match.group(2)],
            "leg_symbol": side_symbol,
            "hypotenuse": f"{original_by_canonical.get('B', 'B')}{original_by_canonical.get('C', 'C')}",
            "target_point": "H",
            "target_definition": "foot of altitude from the right-angle vertex to the hypotenuse",
            "owned_coordinates": {
                original_by_canonical.get("A", "A"): "(0, 0)",
                original_by_canonical.get("B", "B"): f"({side_symbol}, 0)",
                original_by_canonical.get("C", "C"): f"(0, {side_symbol})",
                "H": f"({side_symbol}/2, {side_symbol}/2)",
            },
        },
        "components": {
            f"E_from_{original_by_canonical.get('A', 'A')}": f"(√2*k*{fa}{charge_symbol}/{side_symbol}^2, √2*k*{fa}{charge_symbol}/{side_symbol}^2)",
            f"E_from_{original_by_canonical.get('B', 'B')}": f"(-√2*k*{fb}{charge_symbol}/{side_symbol}^2, √2*k*{fb}{charge_symbol}/{side_symbol}^2)",
            f"E_from_{original_by_canonical.get('C', 'C')}": f"(√2*k*{fc}{charge_symbol}/{side_symbol}^2, -√2*k*{fc}{charge_symbol}/{side_symbol}^2)",
            "E_total": f"({ex_factor}√2*k*{charge_symbol}/{side_symbol}^2, {ey_factor}√2*k*{charge_symbol}/{side_symbol}^2)",
        },
        "vector_terms": [
            {
                "source_point": original_by_canonical.get("A", "A"),
                "target_point": "H",
                "charge_factor": fa,
                "law": "k*q_i*(P-r_i)/|P-r_i|^3",
                "component": f"(√2*k*{fa}{charge_symbol}/{side_symbol}^2, √2*k*{fa}{charge_symbol}/{side_symbol}^2)",
            },
            {
                "source_point": original_by_canonical.get("B", "B"),
                "target_point": "H",
                "charge_factor": fb,
                "law": "k*q_i*(P-r_i)/|P-r_i|^3",
                "component": f"(-√2*k*{fb}{charge_symbol}/{side_symbol}^2, √2*k*{fb}{charge_symbol}/{side_symbol}^2)",
            },
            {
                "source_point": original_by_canonical.get("C", "C"),
                "target_point": "H",
                "charge_factor": fc,
                "law": "k*q_i*(P-r_i)/|P-r_i|^3",
                "component": f"(√2*k*{fc}{charge_symbol}/{side_symbol}^2, -√2*k*{fc}{charge_symbol}/{side_symbol}^2)",
            },
        ],
        "source_facts": _symbolic_source_facts(front_payload),
        "audit": {
            "stage": "symbolic_right_isosceles_altitude_completion",
            "status": "proved_by_symbolic_vector_superposition",
            "formula_family": "point_charge_field_superposition",
            "charge_factors": factors,
            "result_vector_factor": {"x": ex_factor, "y": ey_factor},
            "point_label_mapping": dict(triangle_context.get("canonical_by_original") or {}),
        },
    }


def _symbolic_target_point_label(text: str) -> str | None:
    patterns = (
        r"\bpoint\s+([A-Z])\b[^.]{0,80}\bperpendicular bisector\b",
        r"\bperpendicular bisector\b[^.]{0,80}\bpoint\s+([A-Z])\b",
        r"\belectric field\s+at\s+(?:point\s+)?([A-Z])\b",
        r"\bat\s+(?:point\s+)?([A-Z])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def _symbolic_distance_symbol(text: str) -> str | None:
    patterns = (
        r"\bdistance\s+([a-zA-Z][A-Za-z0-9_]*)\b",
        r"\bheight\s+([a-zA-Z][A-Za-z0-9_]*)\b",
        r"\b(?:is|equals|=)\s+([a-zA-Z][A-Za-z0-9_]*)\s+(?:from|above|away)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _symbolic_charge_factors(front_payload: dict, triangle_context: dict | None) -> tuple[str | None, dict[str, int]]:
    text = str(front_payload.get("canonical_question") or "")
    factors: dict[str, int] = {}
    charge_symbol: str | None = None
    equal_match = re.search(r"\bq([A-Z])\s*=\s*q([A-Z])\s*=\s*([qQ][A-Za-z0-9_′']*)\b", text)
    if equal_match:
        charge_symbol = equal_match.group(3)
        for raw_point in (equal_match.group(1), equal_match.group(2)):
            point = _canonical_point(raw_point, triangle_context)
            if point:
                factors[point] = 1
    if charge_symbol:
        for match in re.finditer(rf"\bq([A-Z])\s*=\s*(?:(\d+)\s*\*?\s*)?{re.escape(charge_symbol)}\b", text):
            point = _canonical_point(match.group(1), triangle_context)
            if point:
                factors[point] = int(match.group(2) or "1")
    compact = re.sub(r"\s+", "", text)
    if charge_symbol:
        for match in re.finditer(rf"q([A-Z])=(\d+){re.escape(charge_symbol)}\b", compact):
            point = _canonical_point(match.group(1), triangle_context)
            if point:
                factors[point] = int(match.group(2))
    return charge_symbol, factors


def _field_magnitude_coefficient(ex_factor: int, ey_factor: int) -> str:
    squared = ex_factor * ex_factor + ey_factor * ey_factor
    if squared == 1:
        return "sqrt(2)"
    if squared == 2:
        return "2"
    if squared == 4:
        return "2*sqrt(2)"
    return f"sqrt({2 * squared})"


def _symbolic_direction_text(ex_factor: int, ey_factor: int, triangle_context: dict | None) -> str:
    original_by_canonical = (triangle_context or {}).get("original_by_canonical") or {}
    a_label = original_by_canonical.get("A", "A")
    b_label = original_by_canonical.get("B", "B")
    c_label = original_by_canonical.get("C", "C")
    if ey_factor == 0 and ex_factor > 0:
        return f"parallel to {a_label}{b_label}, from {a_label} toward {b_label}"
    if ey_factor == 0 and ex_factor < 0:
        return f"parallel to {a_label}{b_label}, from {b_label} toward {a_label}"
    if ex_factor == 0 and ey_factor > 0:
        return f"parallel to {a_label}{c_label}, from {a_label} toward {c_label}"
    if ex_factor == 0 and ey_factor < 0:
        return f"parallel to {a_label}{c_label}, from {c_label} toward {a_label}"
    return f"with component ratio ({ex_factor}, {ey_factor}) in the ({a_label}{b_label}, {a_label}{c_label}) axes"


def _symbolic_source_facts(front_payload: dict) -> list[dict]:
    facts = []
    for relation in front_payload.get("symbolic_relations") or []:
        facts.append(
            {
                "raw_text": relation.get("raw_text"),
                "symbol": relation.get("lhs"),
                "dimension": "symbolic_relation",
                "unit": None,
                "si_value": None,
                "source": relation.get("context"),
            }
        )
    for quantity in front_payload.get("symbolic_quantities") or []:
        if quantity.get("dimension") in {"charge", "length"}:
            facts.append(
                {
                    "raw_text": quantity.get("raw_text"),
                    "symbol": quantity.get("symbol"),
                    "dimension": quantity.get("dimension"),
                    "unit": None,
                    "si_value": None,
                    "source": quantity.get("context"),
                }
            )
    return facts


def _solver_from_geometry(
    result: GeometryExecutionResult,
    formula_id: str,
    confidence: float,
    source_facts: list[dict],
    geometry_audit: dict | None = None,
    medium_scale: float = 1.0,
) -> SolverResult:
    spec = FORMULA_REGISTRY[formula_id]
    if not result.ok:
        return _spatial_unsolved("geometry_execution_failed", spec.task_type, {"geometry": result.to_dict()})
    value = float(result.value)
    components = dict(result.components or {})
    geometry_trace = dict(result.trace)
    if medium_scale != 1.0:
        value *= medium_scale
        for key in ("x", "y", "magnitude"):
            if key in components and isinstance(components[key], (int, float)):
                components[key] = float(components[key]) * medium_scale
        scaled_contributions = []
        for contribution in geometry_trace.get("contributions") or []:
            if not isinstance(contribution, dict):
                scaled_contributions.append(contribution)
                continue
            item = dict(contribution)
            vector = item.get("vector")
            if isinstance(vector, dict):
                item["vector"] = {
                    axis: float(axis_value) * medium_scale
                    for axis, axis_value in vector.items()
                    if isinstance(axis_value, (int, float))
                }
            scaled_contributions.append(item)
        geometry_trace["contributions"] = scaled_contributions
        geometry_trace["medium_scaling"] = {
            "applied": True,
            "scale": medium_scale,
            "reason": "uniform_dielectric_reduces_coulomb_field_or_force",
        }
    return SolverResult(
        solved=True,
        answer=f"{value:.6g} {result.unit}",
        value=value,
        unit=result.unit,
        formula_id=formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "spatial_vector_engine",
            "formula_id": formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "geometry_engine": {**geometry_trace, "components": components, "value": value},
            "components": [],
            "source": source_facts,
            "constants": {"k": K_COULOMB, "medium_scale": medium_scale},
            "geometry_audit": dict(geometry_audit or {}),
            "binding_audit": {"policy": "geometry_template_coordinates", "template_id": result.trace.get("template_id")},
        },
        confidence=confidence,
    )


def _triangle_lengths(front_payload: dict, triangle_context: dict | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for quantity in front_payload.get("quantities", []):
        symbol = str(quantity.get("symbol") or "")
        if quantity.get("dimension") != "length":
            continue
        key = _canonical_side_key(symbol, triangle_context)
        if key:
            out[key] = _si_value(quantity)
            continue
        inferred_key = _infer_unlabeled_side_key(front_payload, quantity, triangle_context)
        if inferred_key and inferred_key not in out:
            out[inferred_key] = _si_value(quantity)
    return out


def _complete_right_triangle_lengths(
    front_payload: dict,
    triangle_lengths: dict[str, float],
    triangle_context: dict | None = None,
) -> tuple[dict[str, float], dict]:
    """Derive one missing side from a textually anchored right angle.

    This keeps geometry recovery structural: the parser may identify side
    lengths and the right-angle vertex, but coordinates and the Pythagorean
    completion are owned by deterministic code.
    """

    lengths = dict(triangle_lengths)
    raw_vertex = _right_angle_point(str(front_payload.get("canonical_question") or "").lower())
    vertex = _canonical_point(raw_vertex, triangle_context)
    if not raw_vertex or not vertex:
        return lengths, {}

    hypotenuse_by_vertex = {"A": "bc", "B": "ac", "C": "ab"}
    legs_by_vertex = {"A": ("ab", "ac"), "B": ("ab", "bc"), "C": ("ac", "bc")}
    hypotenuse = hypotenuse_by_vertex[vertex]
    legs = legs_by_vertex[vertex]
    audit = {
        "stage": "right_triangle_completion",
        "right_angle_at": raw_vertex,
        "canonical_right_angle_at": vertex,
        "hypotenuse": hypotenuse,
        "given_sides": sorted(triangle_lengths),
        "derived_sides": {},
        "status": "no_completion_needed",
    }
    if triangle_context:
        audit["point_label_mapping"] = dict(triangle_context.get("canonical_by_original") or {})
        audit["original_by_canonical"] = dict(triangle_context.get("original_by_canonical") or {})
    tolerance = 1e-12 * max([1.0, *[abs(value) for value in lengths.values()]])

    if hypotenuse in lengths:
        known_legs = [leg for leg in legs if leg in lengths]
        if len(known_legs) == 1:
            known = known_legs[0]
            missing = next(leg for leg in legs if leg != known)
            squared = lengths[hypotenuse] * lengths[hypotenuse] - lengths[known] * lengths[known]
            if squared <= tolerance:
                audit["status"] = "invalid_hypotenuse_or_leg_lengths"
                return lengths, audit
            lengths[missing] = math.sqrt(max(0.0, squared))
            audit["status"] = "derived_missing_leg"
            audit["derived_sides"] = {missing: lengths[missing]}
            return lengths, audit
        if len(known_legs) == 2:
            expected = math.hypot(lengths[legs[0]], lengths[legs[1]])
            if abs(expected - lengths[hypotenuse]) > max(tolerance, 1e-9 * max(expected, lengths[hypotenuse], 1.0)):
                audit["status"] = "inconsistent_right_triangle_lengths"
            return lengths, audit

    if all(leg in lengths for leg in legs):
        lengths[hypotenuse] = math.hypot(lengths[legs[0]], lengths[legs[1]])
        audit["status"] = "derived_hypotenuse"
        audit["derived_sides"] = {hypotenuse: lengths[hypotenuse]}
    return lengths, audit


def _distance_facts_from_triangle_lengths(triangle_lengths: dict[str, float], triangle_audit: dict) -> dict | None:
    """Bind a two-source/one-target triangle through canonical side keys.

    The local frame for this solver is source A, source B, target C. When the
    frontend has already canonicalized arbitrary surface labels such as M/N/Q
    into A/B/C, this avoids brittle "third length is separation" assumptions.
    """

    if not {"ab", "ac", "bc"} <= set(triangle_lengths):
        return None
    return {
        "separation": triangle_lengths["ab"],
        "d1": triangle_lengths["ac"],
        "d2": triangle_lengths["bc"],
        "audit": {
            "source_1_distance": "canonical_side:ac",
            "source_2_distance": "canonical_side:bc",
            "source_separation": "canonical_side:ab",
            "binding_policy": "canonical_triangle_side_keys_source_a_source_b_target_c",
            "triangle_completion": dict(triangle_audit or {}),
        },
    }


def _medium_field_scale(front_payload: dict) -> float:
    """Return the multiplicative field/force scale for a uniform medium."""

    epsilon_r = _relative_permittivity(front_payload)
    if epsilon_r is None or epsilon_r <= 0:
        return 1.0
    return 1.0 / epsilon_r


def _relative_permittivity(front_payload: dict) -> float | None:
    for constant in front_payload.get("numeric_constants") or []:
        if constant.get("dimension") != "permittivity":
            continue
        try:
            value = float(constant.get("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    text = str(front_payload.get("canonical_question") or "")
    match = re.search(
        r"\b(?:dielectric\s+constant|relative\s+permittivity|epsilon_r|εr|ε)\b[^.?,;:=]{0,30}(?:=|is|of|:)?\s*"
        r"(?P<value>\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        value = float(match.group("value"))
        if math.isfinite(value) and value > 0:
            return value
    return None


def _spatial_source_facts(front_payload: dict) -> list[dict]:
    facts = []
    for quantity in front_payload.get("quantities", []):
        if quantity.get("dimension") not in {"charge", "length"}:
            continue
        facts.append(
            {
                "raw_text": quantity.get("raw_text"),
                "symbol": quantity.get("symbol"),
                "dimension": quantity.get("dimension"),
                "unit": quantity.get("unit"),
                "si_value": _si_value(quantity),
                "entity_id": quantity.get("entity_id"),
                "state_id": quantity.get("state_id"),
            }
        )
    return facts


def _quantity_trace(quantity: dict) -> dict:
    payload = {
        "raw_text": quantity.get("raw_text"),
        "symbol": quantity.get("symbol"),
        "dimension": quantity.get("dimension"),
        "unit": quantity.get("unit"),
        "entity_id": quantity.get("entity_id"),
        "state_id": quantity.get("state_id"),
    }
    try:
        payload["si_value"] = _si_value(quantity)
    except Exception:
        payload["si_value"] = quantity.get("si_value")
    return payload


def _charges_by_point(front_payload: dict, triangle_context: dict | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    assigned_symbols: set[str] = set()
    for quantity in front_payload.get("quantities", []):
        if quantity.get("dimension") != "charge":
            continue
        symbol = str(quantity.get("symbol") or "")
        match = re.search(r"q[_-]?([A-Za-z])$", symbol, flags=re.IGNORECASE)
        if match:
            point = _canonical_point(match.group(1), triangle_context)
            if point:
                out[point] = _si_value(quantity)
                assigned_symbols.add(symbol.lower())
    symbol_points = _charge_symbol_point_map(front_payload, triangle_context)
    for quantity in front_payload.get("quantities", []):
        if quantity.get("dimension") != "charge":
            continue
        symbol = str(quantity.get("symbol") or "")
        point = symbol_points.get(symbol.lower())
        if point and point in {"A", "B", "C"}:
            out[point] = _si_value(quantity)
            assigned_symbols.add(symbol.lower())
    if _respectively_ordered_point_context(front_payload):
        points = _triangle_points(triangle_context)
        for quantity in front_payload.get("quantities", []):
            if quantity.get("dimension") != "charge":
                continue
            symbol = str(quantity.get("symbol") or "")
            match = re.fullmatch(r"q(\d+)", symbol, flags=re.IGNORECASE)
            if match:
                index = int(match.group(1)) - 1
                if 0 <= index < len(points):
                    out.setdefault(points[index], _si_value(quantity))
                    assigned_symbols.add(symbol.lower())
    missing_points = [point for point in _triangle_points(triangle_context) if point not in out]
    unassigned_charges = [
        quantity
        for quantity in front_payload.get("quantities", [])
        if quantity.get("dimension") == "charge"
        and str(quantity.get("symbol") or "").lower() not in assigned_symbols
        and unit_info(quantity.get("unit") or "") is not None
    ]
    if len(missing_points) == 1 and len(unassigned_charges) == 1 and len(out) >= 2:
        out[missing_points[0]] = _si_value(unassigned_charges[0])
    if not out:
        for point, value in zip(["A", "B", "C"], _ordered_charges(front_payload)):
            out[point] = value
    return out


def _respectively_ordered_point_context(front_payload: dict) -> bool:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "respectively" in text and any(cue in text for cue in ["points", "vertices", " at a and b", " at a, b"]):
        return True
    return bool(re.search(r"\b(?:vertices|points)\s+[a-z]\s*,\s*[a-z]\s*,\s*and\s+[a-z]\b", text))


def _target_charge_point(front_payload: dict, charge_by_point: dict[str, float], triangle_context: dict | None = None) -> str | None:
    target_text = _target_text(front_payload)
    full_text = str(front_payload.get("canonical_question") or "").lower()
    symbol_points = _charge_symbol_point_map(front_payload, triangle_context)
    for symbol, point in symbol_points.items():
        if point in charge_by_point and re.search(rf"\b{re.escape(symbol)}\b", target_text, flags=re.IGNORECASE):
            return point
    numbered = re.search(r"\bq(\d+)\b", target_text, flags=re.IGNORECASE)
    if numbered:
        points = _triangle_points(triangle_context)
        index = int(numbered.group(1)) - 1
        if 0 <= index < len(points) and points[index] in charge_by_point:
            return points[index]
    for point in charge_by_point:
        labels = [point]
        original = (triangle_context or {}).get("original_by_canonical", {}).get(point)
        if original and original not in labels:
            labels.append(original)
        for label in labels:
            lower = label.lower()
            if re.search(rf"\bq[_-]?{lower}\b", target_text):
                return point
            if re.search(rf"\bcharge\s+(?:at\s+)?(?:point\s+)?{lower}\b", target_text):
                return point
            if re.search(rf"\b(?:force|acting)\b[^.?;]*\bcharge\s+at\s+(?:point\s+)?{lower}\b", full_text):
                return point
    return None


def _target_field_point(front_payload: dict, charge_by_point: dict[str, float], triangle_context: dict | None = None) -> str | None:
    target_text = _target_text(front_payload)
    full_text = str(front_payload.get("canonical_question") or "")
    points = _triangle_points(triangle_context)
    for point in points:
        original = (triangle_context or {}).get("original_by_canonical", {}).get(point, point)
        labels = {point.lower(), str(original).lower()}
        for label in labels:
            if re.search(rf"\b(?:field|intensity|potential)\s+(?:at|on)\s+(?:point\s+)?{re.escape(label)}\b", target_text):
                return point
            if re.search(rf"\bat\s+(?:point\s+)?{re.escape(label)}\b", target_text):
                return point
            if re.search(rf"\b(?:field|intensity|potential)\b[^.?;]*\bat\s+(?:point\s+)?{re.escape(label)}\b", full_text, flags=re.IGNORECASE):
                return point
    return _target_charge_point(front_payload, charge_by_point, triangle_context)


def _uncharged_triangle_vertex(charge_by_point: dict[str, float], triangle_context: dict | None = None) -> str | None:
    missing = [point for point in _triangle_points(triangle_context) if point not in charge_by_point]
    return missing[0] if len(missing) == 1 else None


def _triangle_points(triangle_context: dict | None = None) -> list[str]:
    if triangle_context and triangle_context.get("canonical_points"):
        return list(triangle_context["canonical_points"])
    return ["A", "B", "C"]


def _target_text(front_payload: dict) -> str:
    goal_texts = []
    goal_texts.extend(str(item) for item in front_payload.get("target_hints") or [])
    goal_texts.extend(str(goal.get("text") or goal.get("raw_text") or goal.get("name") or "") for goal in front_payload.get("goals") or [])
    return " ".join(goal_texts).lower()


def _charge_symbol_point_map(front_payload: dict, triangle_context: dict | None = None) -> dict[str, str]:
    text = str(front_payload.get("canonical_question") or "")
    mapping: dict[str, str] = {}

    for match in re.finditer(
        r"\b(?P<sym>q[0-9A-Za-z]*)\s*=\s*[^.]{0,90}?\b(?:(?:is\s+)?(?:placed\s+)?at|located\s+at)\s+(?:point\s+)?(?P<point>[A-Z])\b",
        text,
        flags=re.IGNORECASE,
    ):
        if not match.group("point").isupper():
            continue
        point = _canonical_point(match.group("point"), triangle_context) or match.group("point").upper()
        mapping[match.group("sym").lower()] = point

    for match in re.finditer(
        r"\b(?P<s1>q[0-9A-Za-z]*)\s*=\s*[^.]{0,120}?\band\s+(?P<s2>q[0-9A-Za-z]*)\s*=\s*[^.]{0,120}?"
        r"\b(?:are\s+)?placed(?:\s+in\s+\w+)?\s+at\s+(?:two\s+)?(?:points?\s+)?(?P<p1>[A-Z])\s+and\s+(?P<p2>[A-Z])\b",
        text,
        flags=re.IGNORECASE,
    ):
        if not (match.group("p1").isupper() and match.group("p2").isupper()):
            continue
        p1 = _canonical_point(match.group("p1"), triangle_context) or match.group("p1").upper()
        p2 = _canonical_point(match.group("p2"), triangle_context) or match.group("p2").upper()
        mapping.setdefault(match.group("s1").lower(), p1)
        mapping.setdefault(match.group("s2").lower(), p2)

    for match in re.finditer(
        r"\b(?P<s1>q[0-9A-Za-z]*)\s+and\s+(?P<s2>q[0-9A-Za-z]*)\b[^.]{0,100}?"
        r"\bat\s+(?:two\s+)?points?\s+(?P<p1>[A-Z])\s+and\s+(?P<p2>[A-Z])\b",
        text,
        flags=re.IGNORECASE,
    ):
        if not (match.group("p1").isupper() and match.group("p2").isupper()):
            continue
        p1 = _canonical_point(match.group("p1"), triangle_context) or match.group("p1").upper()
        p2 = _canonical_point(match.group("p2"), triangle_context) or match.group("p2").upper()
        mapping.setdefault(match.group("s1").lower(), p1)
        mapping.setdefault(match.group("s2").lower(), p2)

    return mapping


def _infer_unlabeled_side_key(front_payload: dict, quantity: dict, triangle_context: dict | None = None) -> str | None:
    text = str(front_payload.get("canonical_question") or "")
    span = quantity.get("span")
    if not span:
        return None
    start, end = span
    before = text[max(0, start - 100) : start]
    after = text[end : min(len(text), end + 40)]
    window = f"{before}{quantity.get('raw_text') or ''}{after}"
    if not re.search(r"\b(apart|separated|distance|away)\b", window, flags=re.IGNORECASE):
        return None
    candidates = []
    for pattern in (
        r"\bpoints?\s+([A-Z])\s+and\s+([A-Z])\b",
        r"\b([A-Z])\s+and\s+([A-Z])\b",
        r"\bsegment\s+([A-Z]{2})\b",
        r"\bline\s+segment\s+([A-Z]{2})\b",
    ):
        for match in re.finditer(pattern, window, flags=re.IGNORECASE):
            if len(match.groups()) == 1:
                symbol = match.group(1)
                key = _canonical_side_key(symbol, triangle_context)
            else:
                key = _canonical_side_key(f"{match.group(1)}{match.group(2)}", triangle_context)
            if key:
                candidates.append((match.start(), key))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _ordered_charges(front_payload: dict) -> list[float]:
    charges = [
        quantity
        for quantity in front_payload.get("quantities", [])
        if quantity.get("dimension") == "charge" and unit_info(quantity.get("unit") or "") is not None
    ]
    charges = sorted(charges, key=lambda quantity: quantity.get("span") or (10**9, 10**9))
    return [_si_value(quantity) for quantity in charges]


def _length_by_symbol(front_payload: dict, symbol: str) -> float | None:
    for quantity in front_payload.get("quantities", []):
        if str(quantity.get("symbol") or "").lower() == symbol.lower() and quantity.get("dimension") == "length":
            return _si_value(quantity)
    return None


def _first_length(front_payload: dict) -> float | None:
    for quantity in sorted(front_payload.get("quantities", []), key=lambda item: item.get("span") or (10**9, 10**9)):
        if quantity.get("dimension") == "length" and unit_info(quantity.get("unit") or "") is not None:
            return _si_value(quantity)
    return None


def _si_value(quantity: dict) -> float:
    info = unit_info(quantity.get("unit") or "")
    if info is None:
        raise ValueError(f"unknown_unit:{quantity.get('unit')}")
    return float(quantity["value"]) * info.si_factor


def _spatial_unsolved(reason: str, task_type: str, extra: dict | None = None) -> SolverResult:
    trace = {"stage": "spatial_vector_engine", "reason": reason, "task_type": task_type}
    if extra:
        trace.update(extra)
    return SolverResult(False, "", None, None, None, None, [], trace, 0.0)


def _solve_equilateral_force_from_front(front_payload: dict, route_result) -> SolverResult | None:
    """Handle equilateral force superposition when side length is given once.

    The geometry is fully determined by one side length. If the frontend
    identifies two identical source charges and a target charge at the remaining
    vertex, the constructor expands the repeated source symbol structurally and
    delegates the vector sum to deterministic Coulomb superposition.
    """

    side = _first_length(front_payload)
    if side is None:
        return None
    text = str(front_payload.get("canonical_question") or "").lower()
    charges = _ordered_charges(front_payload)
    geometry_audit = {
        "stage": "equilateral_force_completion",
        "template_id": "equilateral_triangle_vertex",
        "side_m": side,
        "status": "not_recoverable",
    }

    triangle_context = _triangle_context(front_payload)
    charge_by_point = _charges_by_point(front_payload, triangle_context)
    if len(charge_by_point) >= 3:
        target_point = _target_charge_point(front_payload, charge_by_point, triangle_context)
        if target_point is None:
            return None
        sources = [
            {"point": point_id, "charge_c": charge}
            for point_id, charge in charge_by_point.items()
            if point_id != target_point
        ]
        geometry_audit.update(
            {
                "status": "labeled_equilateral_vertices",
                "target_point": target_point,
                "source_points": [source["point"] for source in sources],
            }
        )
        result = execute_coulomb_force_superposition(
            "equilateral_triangle_vertex",
            {"side": side},
            sources,
            {"point": target_point, "charge_c": charge_by_point[target_point]},
        )
        return _solver_from_geometry(
            result,
            "coulomb_force_triangle_sides",
            min(0.76, float(route_result.confidence)),
            _spatial_source_facts(front_payload),
            geometry_audit,
            medium_scale=_medium_field_scale(front_payload),
        )

    if len(charges) >= 2 and _two_identical_source_charges_cue(text):
        source_charge = charges[0]
        target_charge = charges[1]
        geometry_audit.update(
            {
                "status": "two_identical_sources_at_other_vertices",
                "target_point": "A",
                "source_points": ["B", "C"],
                "source_charge_c": source_charge,
                "target_charge_c": target_charge,
            }
        )
        result = execute_coulomb_force_superposition(
            "equilateral_triangle_vertex",
            {"side": side},
            [{"point": "B", "charge_c": source_charge}, {"point": "C", "charge_c": source_charge}],
            {"point": "A", "charge_c": target_charge},
        )
        return _solver_from_geometry(
            result,
            "coulomb_force_triangle_sides",
            min(0.76, float(route_result.confidence)),
            _spatial_source_facts(front_payload),
            geometry_audit,
            medium_scale=_medium_field_scale(front_payload),
        )

    return None


def _solve_equilateral_center_force_from_front(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not re.search(r"\b(?:equilateral|regular)\s+triangle\b", text):
        return None
    if not any(cue in text for cue in ["center", "centre", "centroid"]):
        return None
    side = _first_length(front_payload)
    if side is None:
        return None
    charges = _ordered_charges(front_payload)
    if len(charges) < 4:
        return None
    target_quantities = [
        quantity
        for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "charge" and unit_info(quantity.get("unit") or "") is not None
    ]
    target_index = _target_charge_quantity_index(front_payload, target_quantities)
    if target_index is None:
        target_index = len(charges) - 1
    if target_index < 0 or target_index >= len(charges):
        return None
    source_values = [value for index, value in enumerate(charges) if index != target_index][:3]
    if len(source_values) < 3:
        return None
    coordinates = build_template_coordinates("equilateral_triangle_vertex", {"side": side})
    coordinates["O"] = Vector2(
        sum(point.x for point in coordinates.values()) / 3.0,
        sum(point.y for point in coordinates.values()) / 3.0,
    )
    sources = [
        {"point": point, "charge_c": charge}
        for point, charge in zip(["A", "B", "C"], source_values)
    ]
    result = _force_from_coordinates(coordinates, sources, {"point": "O", "charge_c": charges[target_index]}, "equilateral_triangle_centroid")
    return _solver_from_geometry(
        result,
        "coulomb_force_triangle_sides",
        min(0.74, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        {
            "stage": "equilateral_center_force_completion",
            "status": "constructed_centroid_from_equilateral_side",
            "side_m": side,
            "target_point": "O",
            "target_charge_index": target_index,
            "source_points": ["A", "B", "C"],
        },
        medium_scale=_medium_field_scale(front_payload),
    )


def _solve_square_three_vertex_field_from_front(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "square" not in text or not _unoccupied_square_vertex_cue(text):
        return None
    side = _first_length(front_payload)
    if side is None:
        return None
    charges = _ordered_charges(front_payload)
    if len(charges) == 1 and re.search(r"\bthree\s+(?:equal|identical|same)\b", text):
        charges = [charges[0], charges[0], charges[0]]
    if len(charges) < 3:
        return None
    result = execute_electric_field_superposition(
        "square_vertex_field",
        {"side": side},
        [
            {"point": "A", "charge_c": charges[0]},
            {"point": "B", "charge_c": charges[1]},
            {"point": "C", "charge_c": charges[2]},
        ],
        target_point="D",
    )
    return _solver_from_geometry(
        result,
        "electric_field_square_three_equal_vertex",
        min(0.74, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        {
            "stage": "square_unoccupied_vertex_field_completion",
            "status": "constructed_square_from_side_and_three_source_vertices",
            "side_m": side,
            "source_points": ["A", "B", "C"],
            "target_point": "D",
        },
        medium_scale=_medium_field_scale(front_payload),
    )


def _solve_square_cancel_charge_from_front(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "")
    lowered = text.lower()
    if "square" not in lowered or "zero" not in lowered or "electric field" not in lowered:
        return None
    if "what charge" not in lowered and "determine the charge" not in lowered:
        return None

    if "center" in lowered or "centre" in lowered:
        numeric = _square_center_cancel_numeric(front_payload)
        if numeric is not None:
            value, unknown_point = numeric
            spec = FORMULA_REGISTRY["electric_field_square_center_cancel_charge"]
            return SolverResult(
                solved=True,
                answer=f"{value:.6g} {spec.target_unit}",
                value=value,
                unit=spec.target_unit,
                formula_id=spec.formula_id,
                principle_id=spec.principle_id,
                premises=[spec.premise],
                trace={
                    "stage": "square_symmetry_unknown_charge_engine",
                    "formula_id": spec.formula_id,
                    "expression": "sum_i q_i*(O-r_i)=0",
                    "target_dimension": "charge",
                    "geometry_audit": {
                        "stage": "square_center_cancel_completion",
                        "status": "solved_unknown_vertex_charge_by_vector_balance",
                        "unknown_point": unknown_point,
                    },
                    "source": _spatial_source_facts(front_payload),
                    "constants": {"k": K_COULOMB},
                    "binding_audit": {"policy": "square_center_equal_distance_vector_balance"},
                },
                confidence=min(0.72, float(route_result.confidence)),
            )

    symbolic_vertex_case = _symbolic_square_vertex_zero_field_case(front_payload)
    if symbolic_vertex_case is not None:
        spec = FORMULA_REGISTRY["electric_field_square_center_cancel_charge"]
        answer = f"{symbolic_vertex_case['unknown_symbol']} = -2√2 {symbolic_vertex_case['known_symbol']}"
        return SolverResult(
            solved=True,
            answer=answer,
            value=answer,
            unit=spec.target_unit,
            formula_id=spec.formula_id,
            principle_id=spec.principle_id,
            premises=[
                "At a square vertex, the two adjacent equal charges contribute perpendicular components of magnitude kq/a^2.",
                "The opposite-corner unknown charge contributes components along the diagonal; setting both components to zero gives q_unknown = -2√2 q.",
            ],
            trace={
                "stage": "symbolic_square_zero_field_engine",
                "formula_id": spec.formula_id,
                "expression": "q_opposite = -2*sqrt(2)*q_adjacent for zero field at a square vertex",
                "target_dimension": "charge",
                "geometry_audit": {
                    "stage": "square_vertex_zero_field_completion",
                    "status": "symbolic_vector_balance",
                    **symbolic_vertex_case["geometry"],
                },
                "source": _symbolic_source_facts(front_payload),
                "constants": {"k": "Coulomb constant"},
                "binding_audit": {"policy": "square_vertex_component_balance"},
            },
            confidence=min(0.7, float(route_result.confidence)),
        )
    return None


def _target_asks_unknown_charge(target_text: str, full_text: str) -> bool:
    combined = f"{target_text} {full_text}".lower()
    return bool(
        re.search(r"\b(?:what|find|determine|calculate)\b[^.?;]{0,80}\bcharge\b", combined)
        or re.search(r"\bunknown\s+(?:vertex\s+|corner\s+|point\s+)?charge\b", combined)
        or re.search(r"\bcharge\s+(?:at|on)\s+(?:the\s+)?(?:third|remaining|unknown|missing)\s+(?:vertex|corner|point)\b", combined)
        or re.search(r"\bq[A-Za-z0-9_′']*\b[^.?;]{0,40}\b(?:must|should|needed|required)\b", combined)
    )


def _symbolic_square_vertex_zero_field_case(front_payload: dict) -> dict | None:
    text = str(front_payload.get("canonical_question") or "")
    target_point = _square_target_field_point(front_payload)
    if target_point is None:
        return None
    adjacent_points = _square_adjacent_points(target_point)
    opposite_point = _square_opposite_point(target_point)
    if opposite_point is None:
        return None
    charge_symbols = _symbolic_square_charge_symbols(text)
    adjacent_symbols = [charge_symbols.get(point) for point in adjacent_points]
    if not adjacent_symbols[0] or adjacent_symbols[0] != adjacent_symbols[1]:
        return None
    unknown_symbol = charge_symbols.get(opposite_point) or f"q{opposite_point}"
    return {
        "known_symbol": adjacent_symbols[0],
        "unknown_symbol": unknown_symbol,
        "geometry": {
            "template_id": "square_vertex_field",
            "target_field_point": target_point,
            "adjacent_equal_charge_points": adjacent_points,
            "unknown_charge_point": opposite_point,
        },
    }


def _square_target_field_point(front_payload: dict) -> str | None:
    text = f"{_target_text(front_payload)} {front_payload.get('canonical_question') or ''}"
    field_patterns = [
        r"\b(?:electric\s+field|field|intensity)\b[^.?;]{0,80}\b(?:at|on)\s+(?:point\s+)?([A-D])\b",
        r"\b(?:at|on)\s+(?:point\s+)?([A-D])\b[^.?;]{0,80}\b(?:electric\s+field|field|intensity)\b",
    ]
    for pattern in field_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    target_text = _target_text(front_payload)
    if re.search(r"\bcharge\b[^.?;]{0,60}\b(?:placed|put|located|set)\s+(?:at|on)\b", target_text, flags=re.IGNORECASE):
        return None
    for point in ("A", "B", "C", "D"):
        if re.search(rf"\bat\s+(?:point\s+)?{point.lower()}\b", target_text, flags=re.IGNORECASE):
            return point
    return None


def _square_adjacent_points(point: str) -> list[str]:
    return {
        "A": ["B", "D"],
        "B": ["A", "C"],
        "C": ["B", "D"],
        "D": ["A", "C"],
    }.get(point, [])


def _square_opposite_point(point: str) -> str | None:
    return {"A": "C", "B": "D", "C": "A", "D": "B"}.get(point)


def _symbolic_square_charge_symbols(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    symbol = r"([A-Za-z][A-Za-z0-9_′']*)"
    chain = re.search(
        rf"\bq[_-]?([A-D1-4])\s*=\s*q[_-]?([A-D1-4])\s*=\s*{symbol}\b",
        text,
        flags=re.IGNORECASE,
    )
    if chain:
        base = chain.group(3)
        for label in (chain.group(1), chain.group(2)):
            point = _square_point_from_charge_label(label)
            if point:
                out[point] = base
    for match in re.finditer(r"\bq[_-]?([A-D])\s*=\s*([qQ][A-Za-z0-9_′']*)\b", text, flags=re.IGNORECASE):
        if re.match(r"\s*=", text[match.end() :]):
            continue
        out.setdefault(match.group(1).upper(), match.group(2))
    for match in re.finditer(r"\bq([1-4])\s*=\s*([qQ][A-Za-z0-9_′']*)\b", text, flags=re.IGNORECASE):
        if re.match(r"\s*=", text[match.end() :]):
            continue
        out.setdefault(["A", "B", "C", "D"][int(match.group(1)) - 1], match.group(2))
    return out


def _square_point_from_charge_label(label: str) -> str | None:
    upper = label.upper()
    if upper in {"A", "B", "C", "D"}:
        return upper
    if label.isdigit() and 1 <= int(label) <= 4:
        return ["A", "B", "C", "D"][int(label) - 1]
    return None


def _square_center_cancel_numeric(front_payload: dict) -> tuple[float, str] | None:
    charges_by_point = _square_charges_by_point(front_payload)
    unknown = _square_unknown_charge_point(front_payload, charges_by_point)
    if unknown is None:
        return None
    coordinates = build_template_coordinates("square_vertex_field", {"side": 1.0})
    center = Vector2(0.5, 0.5)
    known_vector = Vector2(0.0, 0.0)
    for point, charge in charges_by_point.items():
        if point == unknown:
            continue
        displacement = center.sub(coordinates[point])
        known_vector = known_vector.add(displacement.scale(charge))
    unknown_vector = center.sub(coordinates[unknown])
    candidates = []
    if abs(unknown_vector.x) > 1e-15:
        candidates.append(-known_vector.x / unknown_vector.x)
    if abs(unknown_vector.y) > 1e-15:
        candidates.append(-known_vector.y / unknown_vector.y)
    if not candidates:
        return None
    if max(candidates) - min(candidates) > 1e-9 * max(1.0, max(abs(value) for value in candidates)):
        return None
    return sum(candidates) / len(candidates), unknown


def _square_charges_by_point(front_payload: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    text = str(front_payload.get("canonical_question") or "")
    point_map = _charge_symbol_point_map(front_payload)
    for quantity in front_payload.get("quantities") or []:
        if quantity.get("dimension") != "charge" or unit_info(quantity.get("unit") or "") is None:
            continue
        symbol = str(quantity.get("symbol") or "")
        point = point_map.get(symbol.lower())
        if point in {"A", "B", "C", "D"}:
            out[point] = _si_value(quantity)
            continue
        match = re.fullmatch(r"q(\d+)", symbol, flags=re.IGNORECASE)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < 4:
                out[["A", "B", "C", "D"][index]] = _si_value(quantity)
                continue
    if len(out) < 3:
        ordered = _ordered_charges(front_payload)
        if len(ordered) >= 3 and re.search(r"\bvertices?\s+a\s*,\s*b\s*,?\s+and\s+c\b", text, flags=re.IGNORECASE):
            for point, charge in zip(["A", "B", "C"], ordered[:3]):
                out.setdefault(point, charge)
    return out


def _square_unknown_charge_point(front_payload: dict, charges_by_point: dict[str, float]) -> str | None:
    target_text = _target_text(front_payload)
    for point in ["A", "B", "C", "D"]:
        if re.search(rf"\b(?:at|placed\s+at)\s+(?:point\s+)?{point.lower()}\b", target_text):
            return point
    for symbol, point in _charge_symbol_point_map(front_payload).items():
        if symbol in target_text and point in {"A", "B", "C", "D"}:
            return point
    missing = [point for point in ["A", "B", "C", "D"] if point not in charges_by_point]
    return missing[0] if len(missing) == 1 else None


def _solve_perpendicular_bisector_force_from_front(front_payload: dict, route_result) -> SolverResult | None:
    separation, height = _perpendicular_bisector_parameters(front_payload)
    roles = _line_template_force_roles(front_payload)
    if separation is None or height is None or roles is None:
        return None
    result = execute_coulomb_force_superposition(
        "point_on_perpendicular_bisector",
        {"separation": separation, "height": height},
        roles["sources"],
        roles["target"],
    )
    return _solver_from_geometry(
        result,
        "coulomb_force_triangle_sides",
        min(0.74, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        {
            "stage": "perpendicular_bisector_force_completion",
            "status": "constructed_from_segment_and_height",
            "separation_m": separation,
            "height_m": height,
            **roles["audit"],
        },
        medium_scale=_medium_field_scale(front_payload),
    )


def _solve_perpendicular_bisector_field_from_front(front_payload: dict, route_result) -> SolverResult | None:
    separation, height = _perpendicular_bisector_parameters(front_payload)
    sources = _line_template_field_sources(front_payload)
    if separation is None or height is None or sources is None:
        return None
    result = execute_electric_field_superposition(
        "point_on_perpendicular_bisector",
        {"separation": separation, "height": height},
        sources,
    )
    return _solver_from_geometry(
        result,
        "electric_field_two_charge_superposition",
        min(0.74, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        {
            "stage": "perpendicular_bisector_field_completion",
            "status": "constructed_from_segment_and_height",
            "separation_m": separation,
            "height_m": height,
            "source_points": [source["point"] for source in sources],
            "binding_policy": "two_source_field_roles_for_line_template",
        },
        medium_scale=_medium_field_scale(front_payload),
    )


def _solve_midpoint_force_from_front(front_payload: dict, route_result) -> SolverResult | None:
    separation = _first_length(front_payload)
    roles = _line_template_force_roles(front_payload)
    if separation is None or roles is None:
        return None
    result = execute_coulomb_force_superposition(
        "point_on_midpoint",
        {"separation": separation},
        roles["sources"],
        roles["target"],
    )
    return _solver_from_geometry(
        result,
        "coulomb_force_triangle_sides",
        min(0.74, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        {
            "stage": "midpoint_force_completion",
            "status": "constructed_from_midpoint",
            "separation_m": separation,
            **roles["audit"],
        },
        medium_scale=_medium_field_scale(front_payload),
    )


def _solve_right_isosceles_force_from_front(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not _has_right_isosceles_text(text):
        return None
    side = _first_length(front_payload)
    if side is None:
        return None
    charges = _ordered_charges(front_payload)
    if len(charges) == 1 and re.search(r"\bthree\s+(?:identical|equal|same)\s+charges\b", text):
        charges = [charges[0], charges[0], charges[0]]
    if len(charges) < 3:
        return None
    charge_quantities = [
        quantity
        for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "charge" and unit_info(quantity.get("unit") or "") is not None
    ]
    target_index = _target_charge_quantity_index(front_payload, charge_quantities)
    if target_index is None:
        target_index = 0 if "right angle vertex" in text or "right-angle vertex" in text else 2
    target_index = max(0, min(target_index, len(charges) - 1))
    target_charge = charges[target_index]
    source_values = [value for index, value in enumerate(charges) if index != target_index][:2]
    if len(source_values) < 2:
        return None
    result = execute_coulomb_force_superposition(
        "right_isosceles_triangle_vertex",
        {"leg": side},
        [{"point": "B", "charge_c": source_values[0]}, {"point": "C", "charge_c": source_values[1]}],
        {"point": "A", "charge_c": target_charge},
    )
    return _solver_from_geometry(
        result,
        "coulomb_right_isosceles_identical_vertex" if len(set(round(v, 24) for v in charges[:3])) == 1 else "coulomb_force_triangle_sides",
        min(0.74, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        {
            "stage": "right_isosceles_force_completion",
            "status": "constructed_from_equal_legs_and_right_angle_target",
            "leg_m": side,
            "target_template_point": "A",
            "source_template_points": ["B", "C"],
            "target_charge_index": target_index,
            "binding_policy": "target_charge_at_right_angle_when_no_explicit_vertex_map",
        },
        medium_scale=_medium_field_scale(front_payload),
    )


def _solve_right_isosceles_field_from_front(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not _has_right_isosceles_text(text):
        return None
    if not any(cue in text for cue in ["right angle vertex", "right-angle vertex", "vertex of the right angle"]):
        return None
    side = _first_length(front_payload)
    if side is None:
        return None
    charges = _ordered_charges(front_payload)
    if len(charges) == 1 and re.search(r"\bthree\s+(?:identical|equal|same)\s+charges\b", text):
        charges = [charges[0], charges[0], charges[0]]
    if len(charges) < 3:
        return None
    target_index = 0
    source_values = [value for index, value in enumerate(charges) if index != target_index][:2]
    if len(source_values) < 2:
        return None
    result = execute_electric_field_superposition(
        "right_isosceles_triangle_vertex",
        {"leg": side},
        [{"point": "B", "charge_c": source_values[0]}, {"point": "C", "charge_c": source_values[1]}],
        target_point="A",
    )
    return _solver_from_geometry(
        result,
        "electric_field_two_charge_superposition",
        min(0.74, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        {
            "stage": "right_isosceles_field_completion",
            "status": "constructed_from_equal_legs_and_right_angle_field_target",
            "leg_m": side,
            "target_template_point": "A",
            "source_template_points": ["B", "C"],
            "binding_policy": "field_at_right_angle_excludes_source_charge_at_target_point",
        },
        medium_scale=_medium_field_scale(front_payload),
    )


def _solve_symbolic_right_isosceles_force(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "")
    lowered = text.lower()
    if not _has_right_isosceles_text(lowered):
        return None
    if not re.search(r"\bF0\b|\bF_0\b|\bF₀\b", text):
        return None
    if "remaining vertex" not in lowered and "right angle vertex" not in lowered and "right-angle vertex" not in lowered:
        return None
    formula_id = "coulomb_right_isosceles_identical_vertex"
    spec = FORMULA_REGISTRY[formula_id]
    answer = "√2 F0"
    return SolverResult(
        solved=True,
        answer=answer,
        value=answer,
        unit=spec.target_unit,
        formula_id=formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise, "Two equal perpendicular force components combine to √2 times one component."],
        trace={
            "stage": "symbolic_spatial_vector_engine",
            "formula_id": formula_id,
            "expression": "F_net = sqrt(F0^2 + F0^2)",
            "target_dimension": "force",
            "geometry_audit": {
                "stage": "right_isosceles_symbolic_force_completion",
                "status": "two_equal_perpendicular_components",
                "template_id": "right_isosceles_triangle_vertex",
            },
            "source": _symbolic_source_facts(front_payload),
            "constants": {"k": "Coulomb constant"},
            "binding_audit": {"policy": "given_single_component_force_F0"},
        },
        confidence=min(0.72, float(route_result.confidence)),
    )


def _line_template_field_sources(front_payload: dict) -> list[dict] | None:
    symbol_points = _charge_symbol_point_map(front_payload)
    mapped: dict[str, float] = {}
    for quantity in front_payload.get("quantities") or []:
        if quantity.get("dimension") != "charge" or unit_info(quantity.get("unit") or "") is None:
            continue
        symbol = str(quantity.get("symbol") or "").lower()
        point = symbol_points.get(symbol)
        if point and point.upper() in {"A", "B"}:
            mapped[point.upper()] = _si_value(quantity)
    if "A" in mapped and "B" in mapped:
        return [{"point": "A", "charge_c": mapped["A"]}, {"point": "B", "charge_c": mapped["B"]}]
    charges = _ordered_charges(front_payload)
    if len(charges) >= 2:
        return [{"point": "A", "charge_c": charges[0]}, {"point": "B", "charge_c": charges[1]}]
    return None


def _line_template_force_roles(front_payload: dict) -> dict | None:
    """Ground source and target charges for line/segment templates.

    Internal templates use local labels A/B for fixed source points and P for
    the midpoint or perpendicular-bisector point. This helper maps arbitrary
    surface labels such as M, N, q, q1, q2 into that local frame using only
    explicit placement evidence from the question.
    """

    symbol_points = _charge_symbol_point_map(front_payload)
    charges_by_surface_point: dict[str, float] = {}
    unmapped_charges: list[dict] = []
    for quantity in front_payload.get("quantities") or []:
        if quantity.get("dimension") != "charge" or unit_info(quantity.get("unit") or "") is None:
            continue
        symbol = str(quantity.get("symbol") or "").lower()
        point = symbol_points.get(symbol)
        if point:
            charges_by_surface_point[point.upper()] = _si_value(quantity)
        else:
            unmapped_charges.append(quantity)

    if "A" not in charges_by_surface_point or "B" not in charges_by_surface_point:
        ordered = _ordered_charges(front_payload)
        if len(ordered) >= 3 and "midpoint" in str(front_payload.get("canonical_question") or "").lower():
            return {
                "sources": [{"point": "A", "charge_c": ordered[0]}, {"point": "B", "charge_c": ordered[1]}],
                "target": {"point": "P", "charge_c": ordered[2]},
                "audit": {
                    "source_points": ["A", "B"],
                    "target_point": "P",
                    "surface_to_template_points": {"source_1": "A", "source_2": "B", "target": "P"},
                    "binding_policy": "ordered_two_sources_then_midpoint_target",
                },
            }
        return None
    target_candidates = [
        point
        for point in sorted(charges_by_surface_point)
        if point not in {"A", "B"}
    ]
    target_charge = None
    target_surface_point = None
    if len(target_candidates) == 1:
        target_surface_point = target_candidates[0]
        target_charge = charges_by_surface_point[target_surface_point]
    elif len(unmapped_charges) == 1:
        target_surface_point = "P"
        target_charge = _si_value(unmapped_charges[0])
    else:
        return None
    return {
        "sources": [
            {"point": "A", "charge_c": charges_by_surface_point["A"]},
            {"point": "B", "charge_c": charges_by_surface_point["B"]},
        ],
        "target": {"point": "P", "charge_c": target_charge},
        "audit": {
            "source_points": ["A", "B"],
            "target_point": "P",
            "surface_to_template_points": {
                "A": "A",
                "B": "B",
                target_surface_point: "P",
            },
            "binding_policy": "explicit_charge_point_roles_for_line_template",
        },
    }


def _solve_zero_field_line_from_front(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "zero" not in text or "electric field" not in text:
        return None
    if not any((goal.get("dimension") == "length") for goal in front_payload.get("goals") or []) and not re.search(
        r"\b(?:where|coordinate|distance|how far|from\s+[ab]|am|bm)\b",
        text,
    ):
        return None
    charges = _ordered_charges(front_payload)
    separation = _source_separation_length(front_payload)
    symbolic_ratio_result = _solve_symbolic_zero_field_line(front_payload, route_result, separation)
    if symbolic_ratio_result is not None:
        return symbolic_ratio_result
    if len(charges) < 2 or separation is None:
        return None
    q1, q2 = charges[0], charges[1]
    point_x = _zero_field_coordinate(q1, q2, separation)
    if point_x is None:
        return None
    target_text = _target_text(front_payload)
    if re.search(r"\bfrom\s+(?:point\s+)?b\b|\bto\s+b\b", target_text, flags=re.IGNORECASE):
        value = abs(point_x - separation)
        reference = "B"
    else:
        value = abs(point_x)
        reference = "A"
    spec = FORMULA_REGISTRY["electric_field_zero_line_two_charges"]
    return SolverResult(
        solved=True,
        answer=f"{value:.6g} {spec.target_unit}",
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "spatial_zero_field_line_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "geometry_engine": {
                "template_id": "two_charges_collinear",
                "coordinates": {"A": {"x": 0.0, "y": 0.0}, "B": {"x": separation, "y": 0.0}, "C": {"x": point_x, "y": 0.0}},
                "value": value,
                "components": {"distance_from_reference_m": value, "reference_point": reference},
            },
            "source": _spatial_source_facts(front_payload),
            "constants": {"k": K_COULOMB},
            "geometry_audit": {
                "stage": "zero_field_line_completion",
                "status": "proved_by_signed_1d_field_equilibrium",
                "source_separation_m": separation,
                "zero_coordinate_m": point_x,
                "reference_point": reference,
            },
            "binding_audit": {
                "q1": {"policy": "ordered_source_charge", "selected_index": 0},
                "q2": {"policy": "ordered_source_charge", "selected_index": 1},
                "separation": {"policy": "explicit_source_separation"},
            },
        },
        confidence=min(0.76, float(route_result.confidence)),
    )


def _solve_symbolic_zero_field_line(front_payload: dict, route_result, separation: float | None) -> SolverResult | None:
    if separation is None or separation <= 0:
        return None
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["same sign", "same-sign", "same polarity"]):
        return None
    ratio = _symbolic_charge_ratio_q1_over_q2(text)
    if ratio is None or ratio <= 0:
        return None
    sqrt_ratio = math.sqrt(ratio)
    distance_from_a = separation * sqrt_ratio / (sqrt_ratio + 1.0)
    target_text = _target_text(front_payload)
    if re.search(r"\bfrom\s+(?:point\s+)?b\b|\bto\s+b\b", target_text, flags=re.IGNORECASE):
        value = abs(separation - distance_from_a)
        reference = "B"
    else:
        value = distance_from_a
        reference = "A"
    spec = FORMULA_REGISTRY["electric_field_zero_line_two_charges"]
    return SolverResult(
        solved=True,
        answer=f"{value:.6g} {spec.target_unit}",
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "spatial_zero_field_line_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "source": _spatial_source_facts(front_payload),
            "constants": {"q1_over_q2": ratio},
            "geometry_audit": {
                "stage": "zero_field_line_symbolic_ratio_completion",
                "status": "proved_by_same_sign_signed_1d_field_equilibrium",
                "source_separation_m": separation,
                "reference_point": reference,
            },
            "binding_audit": {"policy": "same_sign_symbolic_charge_ratio"},
        },
        confidence=min(0.72, float(route_result.confidence)),
    )


def _symbolic_charge_ratio_q1_over_q2(text: str) -> float | None:
    compact = re.sub(r"\s+", "", text.lower())
    match = re.search(r"q1=(?P<factor>\d+(?:\.\d+)?)q2", compact)
    if match:
        return float(match.group("factor"))
    match = re.search(r"q2=(?P<factor>\d+(?:\.\d+)?)q1", compact)
    if match:
        factor = float(match.group("factor"))
        return 1.0 / factor if factor > 0 else None
    return None


def _solve_point_charge_midpoint_field_from_front(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "midpoint" not in text or "same electric field line" not in text:
        return None
    fields = [
        quantity
        for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "electric_field" and unit_info(quantity.get("unit") or "") is not None
    ]
    if len(fields) < 2:
        return None
    values = sorted([_si_value(field) for field in fields], reverse=True)
    if values[0] <= 0 or values[1] <= 0:
        return None
    near, far = values[0], values[1]
    ratio = math.sqrt(near / far)
    if ratio <= 0:
        return None
    value = near / (((1.0 + ratio) / 2.0) ** 2)
    spec = FORMULA_REGISTRY["point_charge_field_midpoint_from_two_fields"]
    return SolverResult(
        solved=True,
        answer=f"{value:.6g} {spec.target_unit}",
        value=value,
        unit=spec.target_unit,
        formula_id=spec.formula_id,
        principle_id=spec.principle_id,
        premises=[spec.premise],
        trace={
            "stage": "inverse_square_field_line_engine",
            "formula_id": spec.formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "inputs": {
                "E_near": {"si_value": near, "unit": "V/m"},
                "E_far": {"si_value": far, "unit": "V/m"},
            },
            "geometry_audit": {
                "stage": "field_line_midpoint_completion",
                "status": "derived_relative_distances_from_inverse_square_law",
                "distance_ratio_r_far_over_r_near": ratio,
            },
            "source": _spatial_source_facts(front_payload),
            "constants": {},
            "binding_audit": {"policy": "near_far_order_from_field_magnitude"},
        },
        confidence=min(0.72, float(route_result.confidence)),
    )


def _zero_field_coordinate(q1: float, q2: float, separation: float) -> float | None:
    if separation <= 0 or q1 == 0 or q2 == 0:
        return None
    a = math.sqrt(abs(q1))
    b = math.sqrt(abs(q2))
    if q1 * q2 > 0:
        return separation * a / (a + b)
    if abs(a - b) <= 1e-15 * max(a, b, 1.0):
        return None
    if a > b:
        return separation * a / (a - b)
    return -separation * a / (b - a)


def _source_separation_length(front_payload: dict) -> float | None:
    for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9)):
        if quantity.get("dimension") != "length" or unit_info(quantity.get("unit") or "") is None:
            continue
        local = f"{quantity.get('context') or ''} {quantity.get('raw_text') or ''}".lower()
        if any(cue in local for cue in ["separated", "apart", "between", "source separation"]):
            return _si_value(quantity)
    return _first_length(front_payload)


def _solve_collinear_two_source_vector_from_front(front_payload: dict, route_result, target_dimension: str) -> SolverResult | None:
    case = _collinear_two_source_vector_case(front_payload, target_dimension)
    if case is None:
        return None
    if target_dimension == "electric_field":
        result = _field_from_coordinates(case["coordinates"], case["sources"], "P", "two_charges_collinear")
        formula_id = "electric_field_two_charge_superposition"
    else:
        result = _force_from_coordinates(case["coordinates"], case["sources"], case["target"], "two_charges_collinear")
        formula_id = "coulomb_force_triangle_sides"
    return _solver_from_geometry(
        result,
        formula_id,
        min(0.74, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        case["audit"],
        medium_scale=_medium_field_scale(front_payload),
    )


def _solve_two_source_target_distance_triangle_from_front(front_payload: dict, route_result, target_dimension: str) -> SolverResult | None:
    triangle_context = _triangle_context(front_payload)
    charge_points = _charges_by_point(front_payload, triangle_context)
    if target_dimension == "electric_field":
        named_target = _target_field_point(front_payload, charge_points, triangle_context)
        if named_target is not None and named_target != "C":
            return None
    elif target_dimension == "force":
        named_target = _target_charge_point(front_payload, charge_points, triangle_context)
        if named_target is not None and named_target != "C":
            return None
    triangle_lengths, triangle_audit = _complete_right_triangle_lengths(
        front_payload,
        _triangle_lengths(front_payload, triangle_context),
        triangle_context,
    )
    distances = _distance_facts_from_triangle_lengths(triangle_lengths, triangle_audit)
    if distances is None:
        distances = _two_source_distance_facts(front_payload)
    if distances is None:
        return None
    if _collinear_point_coordinate(distances["d1"], distances["d2"], distances["separation"]) is not None:
        return None
    charges = _ordered_charges(front_payload)
    if target_dimension == "electric_field":
        if len(charges) < 2:
            return None
        result = execute_electric_field_triangle_sides(
            ab=distances["separation"],
            ac=distances["d1"],
            bc=distances["d2"],
            q_a=charges[0],
            q_b=charges[1],
            q_c=None,
            target_point="C",
        )
        return _solver_from_geometry(
            result,
            "electric_field_two_charge_triangle_sides",
            min(0.74, float(route_result.confidence)),
            _spatial_source_facts(front_payload),
            {
                "stage": "two_source_target_distance_triangle_completion",
                "status": "constructed_from_source_separation_and_target_distances",
                "distance_binding": distances["audit"],
                "target_point": "C",
            },
            medium_scale=_medium_field_scale(front_payload),
        )
    if len(charges) < 3:
        return None
    target_index = _target_charge_quantity_index(front_payload, [
        quantity for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "charge"
    ])
    if target_index is None:
        target_index = 2
    charge_values = charges[:]
    if target_index != 2 and 0 <= target_index < len(charge_values):
        target_value = charge_values[target_index]
        source_values = [value for index, value in enumerate(charge_values) if index != target_index]
        if len(source_values) < 2:
            return None
        q_a, q_b, q_c = source_values[0], source_values[1], target_value
    else:
        q_a, q_b, q_c = charge_values[0], charge_values[1], charge_values[2]
    result = execute_coulomb_force_triangle_sides(
        ab=distances["separation"],
        ac=distances["d1"],
        bc=distances["d2"],
        q_a=q_a,
        q_b=q_b,
        q_c=q_c,
        target_point="C",
    )
    return _solver_from_geometry(
        result,
        "coulomb_force_triangle_sides",
        min(0.76, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        {
            "stage": "two_source_target_distance_triangle_completion",
            "status": "constructed_from_source_separation_and_target_distances",
            "distance_binding": distances["audit"],
            "target_point": "C",
        },
        medium_scale=_medium_field_scale(front_payload),
    )


def _solve_three_charge_equally_spaced_line_from_front(front_payload: dict, route_result) -> SolverResult | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if not any(cue in text for cue in ["straight line", "collinear", "same line"]):
        return None
    if not re.search(r"\b(?:apart|equally spaced|separated)\b", text):
        return None
    lengths = [
        quantity
        for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "length" and unit_info(quantity.get("unit") or "") is not None
    ]
    charges = _ordered_charges(front_payload)
    if len(lengths) != 1 or len(charges) < 3:
        return None
    distance = _si_value(lengths[0])
    charge_quantities = [
        quantity
        for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "charge" and unit_info(quantity.get("unit") or "") is not None
    ]
    target_index = _target_charge_quantity_index(front_payload, charge_quantities)
    if target_index is None:
        return None
    points = ["A", "P", "B"]
    coordinates = {"A": Vector2(0.0, 0.0), "P": Vector2(distance, 0.0), "B": Vector2(2.0 * distance, 0.0)}
    target_point = points[target_index]
    sources = [
        {"point": points[index], "charge_c": charges[index]}
        for index in range(3)
        if index != target_index
    ]
    result = _force_from_coordinates(
        coordinates,
        sources,
        {"point": target_point, "charge_c": charges[target_index]},
        "three_charges_equally_spaced_line",
    )
    return _solver_from_geometry(
        result,
        "coulomb_force_triangle_sides",
        min(0.74, float(route_result.confidence)),
        _spatial_source_facts(front_payload),
        {
            "stage": "three_charge_line_completion",
            "status": "constructed_equal_spacing_from_single_apart_distance",
            "spacing_m": distance,
            "target_charge_index": target_index,
            "target_point": target_point,
        },
        medium_scale=_medium_field_scale(front_payload),
    )


def _collinear_two_source_vector_case(front_payload: dict, target_dimension: str) -> dict | None:
    geometry = _collinear_runtime_geometry(front_payload)
    if geometry is None:
        return None
    if target_dimension == "electric_field":
        charges = _ordered_charges(front_payload)
        if len(charges) < 2:
            return None
        return {
            "coordinates": geometry["coordinates"],
            "sources": [{"point": "A", "charge_c": charges[0]}, {"point": "B", "charge_c": charges[1]}],
            "target": None,
            "audit": {**geometry["audit"], "stage": "collinear_field_completion", "status": "two_source_field_superposition"},
        }
    roles = _collinear_force_roles(front_payload)
    if roles is None:
        return None
    return {
        "coordinates": geometry["coordinates"],
        "sources": [{"point": "A", "charge_c": roles["q1"]}, {"point": "B", "charge_c": roles["q2"]}],
        "target": {"point": "P", "charge_c": roles["target"]},
        "audit": {**geometry["audit"], **roles["audit"], "stage": "collinear_force_completion", "status": "two_source_force_superposition"},
    }


def _collinear_runtime_geometry(front_payload: dict) -> dict | None:
    distances = _two_source_distance_facts(front_payload)
    if distances is not None:
        geometry = _collinear_point_coordinate(distances["d1"], distances["d2"], distances["separation"])
        if geometry is None:
            return None
        return {
            "coordinates": {
                "A": Vector2(0.0, 0.0),
                "B": Vector2(distances["separation"], 0.0),
                "P": Vector2(geometry["point_x"], 0.0),
            },
            "audit": {
                "template_id": "two_charges_collinear",
                "position_case": geometry["position_case"],
                "distance_binding": distances["audit"],
            },
        }
    midpoint = _line_midpoint_from_source_separation(front_payload)
    if midpoint is not None:
        return midpoint
    one_distance = _line_segment_point_distance_case(front_payload)
    if one_distance is not None:
        return one_distance
    lengths = [
        quantity
        for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "length" and unit_info(quantity.get("unit") or "") is not None
    ]
    text = str(front_payload.get("canonical_question") or "").lower()
    if len(lengths) >= 2 and any(cue in text for cue in ["opposite sides", "passing through", "same straight line"]):
        d1 = _si_value(lengths[0])
        d2 = _si_value(lengths[1])
        return {
            "coordinates": {"A": Vector2(-d1, 0.0), "B": Vector2(d2, 0.0), "P": Vector2(0.0, 0.0)},
            "audit": {
                "template_id": "two_charges_collinear",
                "position_case": "target_between_sources",
                "source_1_distance": lengths[0].get("raw_text"),
                "source_2_distance": lengths[1].get("raw_text"),
                "binding_policy": "opposite_sides_distances_from_target",
            },
        }
    return None


def _line_midpoint_from_source_separation(front_payload: dict) -> dict | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "equidistant" not in text or not any(cue in text for cue in ["line connecting", "line segment", "straight line"]):
        return None
    if any(cue in text for cue in ["away from the line", "away from line", "perpendicular bisector"]):
        return None
    separation = _source_separation_length(front_payload)
    if separation is None or separation <= 0:
        return None
    return {
        "coordinates": {"A": Vector2(0.0, 0.0), "B": Vector2(separation, 0.0), "P": Vector2(separation / 2.0, 0.0)},
        "audit": {
            "template_id": "two_charges_collinear",
            "position_case": "between_sources",
            "source_separation_m": separation,
            "binding_policy": "equidistant_point_on_source_line",
        },
    }


def _line_segment_point_distance_case(front_payload: dict) -> dict | None:
    text = str(front_payload.get("canonical_question") or "").lower()
    if "perpendicular bisector" in text:
        return None
    if not any(cue in text for cue in ["line segment", "line connecting", "on the line"]):
        return None
    lengths = [
        quantity
        for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "length" and unit_info(quantity.get("unit") or "") is not None
    ]
    if len(lengths) < 2:
        return None
    separation_q = None
    point_q = None
    point_from = "A"
    for quantity in lengths:
        local = f"{quantity.get('context') or ''} {quantity.get('raw_text') or ''}".lower()
        if separation_q is None and any(cue in local for cue in ["long line segment", "separated", "apart", "distance ab", "ab ="]):
            separation_q = quantity
            continue
        if point_q is None and any(cue in local for cue in ["away from", "from q1", "from point a", "from a", "from qcalculate", "charge q1", "of a"]):
            point_q = quantity
            point_from = "A"
            continue
        if point_q is None and any(cue in local for cue in ["from q2", "from point b", "from b", "charge q2", "of b"]):
            point_q = quantity
            point_from = "B"
    if separation_q is None:
        separation_q = lengths[0]
    if point_q is None:
        point_candidates = [quantity for quantity in lengths if quantity is not separation_q]
        if not point_candidates:
            return None
        point_q = point_candidates[0]
    separation = _si_value(separation_q)
    distance = _si_value(point_q)
    if separation <= 0 or distance < 0:
        return None
    if distance > separation and "outside" not in text:
        return None
    if point_from == "A":
        if any(cue in text for cue in ["to the right of a", "right side of a", "right of charge q1", "right of q1"]):
            point_x = distance
        elif any(cue in text for cue in ["to the left of a", "left side of a", "left of charge q1", "left of q1"]):
            point_x = -distance
        elif distance > separation:
            point_x = distance
        else:
            point_x = distance
    else:
        if any(cue in text for cue in ["to the right of b", "right side of b", "right of charge q2", "right of q2"]):
            point_x = separation + distance
        elif any(cue in text for cue in ["to the left of b", "left side of b", "left of charge q2", "left of q2"]):
            point_x = separation - distance
        else:
            point_x = separation - distance
    return {
        "coordinates": {"A": Vector2(0.0, 0.0), "B": Vector2(separation, 0.0), "P": Vector2(point_x, 0.0)},
        "audit": {
            "template_id": "two_charges_collinear",
            "position_case": "between_sources",
            "source_separation": separation_q.get("raw_text"),
            "target_distance": point_q.get("raw_text"),
            "target_reference": point_from,
            "binding_policy": "line_segment_plus_one_source_distance",
        },
    }


def _collinear_force_roles(front_payload: dict) -> dict | None:
    charges = [
        quantity
        for quantity in sorted(front_payload.get("quantities") or [], key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "charge" and unit_info(quantity.get("unit") or "") is not None
    ]
    if len(charges) >= 3:
        target_index = _target_charge_quantity_index(front_payload, charges)
        if target_index is None:
            target_index = next(
                (index for index, quantity in enumerate(charges) if str(quantity.get("symbol") or "").lower() in {"q0", "qo", "q_test", "qprobe"}),
                None,
            )
        if target_index is None:
            target_index = 2 if _target_text(front_payload) else 0
        sources = [quantity for index, quantity in enumerate(charges) if index != target_index]
        if len(sources) < 2:
            return None
        return {
            "q1": _si_value(sources[0]),
            "q2": _si_value(sources[1]),
            "target": _si_value(charges[target_index]),
            "audit": {"charge_binding_policy": "explicit_three_charge_target_grounding", "target_charge_index": target_index},
        }
    text = str(front_payload.get("canonical_question") or "").lower()
    if len(charges) == 2 and _two_identical_source_charges_cue(text):
        target_index = _target_charge_quantity_index(front_payload, charges)
        if target_index is None:
            target_index = _target_like_charge_index(front_payload, charges)
        if target_index is None:
            target_index = 1
        source_index = 1 - target_index
        return {
            "q1": _si_value(charges[source_index]),
            "q2": _si_value(charges[source_index]),
            "target": _si_value(charges[target_index]),
            "audit": {
                "charge_binding_policy": "one_target_charge_plus_two_equal_source_charges",
                "source_charge_index": source_index,
                "target_charge_index": target_index,
                "evidence": "two identical/equal source charge cue",
            },
        }
    return None


def _target_charge_quantity_index(front_payload: dict, charges: list[dict]) -> int | None:
    target_text = _target_text(front_payload)
    for index, quantity in enumerate(charges):
        symbol = str(quantity.get("symbol") or "")
        if symbol and re.search(rf"\b{re.escape(symbol)}\b", target_text, flags=re.IGNORECASE):
            return index
    return None


def _target_like_charge_index(front_payload: dict, charges: list[dict]) -> int | None:
    """Infer a target charge role from local wording, not from fixed labels."""

    for index, quantity in enumerate(charges):
        symbol = str(quantity.get("symbol") or "").lower()
        raw = str(quantity.get("raw_text") or "").lower()
        context = str(quantity.get("context") or "").lower()
        combined = f"{symbol} {raw} {context}"
        if any(cue in combined for cue in ["q0", "q′", "q'", "q_test", "qprobe", "test charge", "probe charge"]):
            return index
    return None


def _solve_collinear_two_charge_direction_from_front(front_payload: dict, route_result) -> SolverResult | None:
    case = _collinear_two_charge_direction_case(front_payload)
    if case is None:
        return None
    formula_id = "coulomb_force_direction_superposition"
    spec = FORMULA_REGISTRY[formula_id]
    direction = case["direction"]
    answer = case["answer"]
    return SolverResult(
        solved=True,
        answer=answer,
        value=answer,
        unit=spec.target_unit,
        formula_id=formula_id,
        principle_id=spec.principle_id,
        premises=[
            spec.premise,
            "A bare test or probe charge uses the standard positive-test-charge convention, so force direction follows the net electric field direction.",
        ],
        trace={
            "stage": "spatial_vector_direction_engine",
            "formula_id": formula_id,
            "expression": spec.expression,
            "target_dimension": spec.target_dimension,
            "compiled_geometry_case": "two_source_collinear_test_point",
            "geometry": case["geometry"],
            "axis_components": case["components"],
            "direction": direction,
            "source": _spatial_source_facts(front_payload),
            "constants": {"k": K_COULOMB},
            "geometry_audit": case["audit"],
            "binding_audit": {
                "policy": "distances_to_sources_plus_source_separation",
                "template_id": "two_charges_collinear",
            },
        },
        confidence=min(0.7, float(route_result.confidence)),
    )


def _collinear_two_charge_direction_case(front_payload: dict) -> dict | None:
    text = str(front_payload.get("canonical_question") or "")
    lowered = text.lower()
    if "direction" not in lowered or not _direction_probe_geometry_cue(lowered):
        return None
    charges = [
        quantity
        for quantity in sorted(front_payload.get("quantities", []), key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "charge"
    ]
    if len(charges) < 2:
        return None
    distances = _two_source_distance_facts(front_payload)
    if distances is None:
        return None
    q1 = _si_value(charges[0])
    q2 = _si_value(charges[1])
    label1 = str(charges[0].get("symbol") or "q1")
    label2 = str(charges[1].get("symbol") or "q2")
    d1, d2, separation = distances["d1"], distances["d2"], distances["separation"]
    geometry = _collinear_point_coordinate(d1, d2, separation)
    if geometry is None:
        return None
    x_p = geometry["point_x"]
    x1 = 0.0
    x2 = separation
    e1 = q1 * (x_p - x1) / abs(x_p - x1) ** 3
    e2 = q2 * (x_p - x2) / abs(x_p - x2) ** 3
    total = e1 + e2
    if abs(total) <= 1e-12 * max(abs(e1), abs(e2), 1.0):
        return {
            "answer": "No net direction; the electric forces cancel.",
            "direction": "zero",
            "geometry": geometry,
            "components": {"from_" + label1: e1, "from_" + label2: e2, "net_axis_component": total},
            "audit": {
                "stage": "collinear_direction_completion",
                "status": "zero_resultant",
                "distance_binding": distances["audit"],
            },
        }
    axis_sign = 1 if total > 0 else -1
    direction_text = _collinear_direction_text(axis_sign, geometry["position_case"], label1, label2)
    answer = direction_text
    return {
        "answer": answer,
        "direction": "positive_axis" if axis_sign > 0 else "negative_axis",
        "geometry": {
            **geometry,
            "source_1": {"label": label1, "x_m": x1, "charge_c": q1},
            "source_2": {"label": label2, "x_m": x2, "charge_c": q2},
            "target": {"label": "test_charge", "x_m": x_p},
        },
        "components": {
            f"field_from_{label1}": e1 * K_COULOMB,
            f"field_from_{label2}": e2 * K_COULOMB,
            "net_field_axis_component": total * K_COULOMB,
            "positive_axis": f"from {label1} toward {label2}",
        },
        "audit": {
            "stage": "collinear_direction_completion",
            "status": "proved_by_signed_1d_superposition",
            "test_charge_sign_convention": "positive_test_charge",
            "distance_binding": distances["audit"],
            "position_case": geometry["position_case"],
        },
    }


def _direction_probe_geometry_cue(lowered_question: str) -> bool:
    return bool(
        re.search(r"\b(?:test|probe|trial|target)\s+charge\b", lowered_question)
        or re.search(r"\bpoint\s+whose\s+distances?\b", lowered_question)
        or re.search(r"\bdistances?\s+to\s+(?:the\s+)?(?:two|three|\d+)\s+charges?\b", lowered_question)
        or re.search(
            r"\b(?:force|field)\s+(?:acting\s+)?(?:on|at|toward|towards)\s+(?:the\s+)?(?:test|probe|trial|target)?\s*charge\b",
            lowered_question,
        )
    )


def _unlabeled_target_vertex_context(front_payload: dict) -> bool:
    text = str(front_payload.get("canonical_question") or "").lower()
    return bool(
        re.search(r"\b(?:remaining|other|third|last|unoccupied|empty)\s+(?:vertex|corner|point)\b", text)
        or re.search(r"\b(?:vertex|corner|point)\s+(?:without|with no)\s+(?:a\s+)?charge\b", text)
    )


def _two_source_distance_facts(front_payload: dict) -> dict | None:
    triangle_context = _triangle_context(front_payload)
    triangle_lengths, triangle_audit = _complete_right_triangle_lengths(
        front_payload,
        _triangle_lengths(front_payload, triangle_context),
        triangle_context,
    )
    canonical_distances = _distance_facts_from_triangle_lengths(triangle_lengths, triangle_audit)
    if canonical_distances is not None:
        return canonical_distances

    lengths = [
        quantity
        for quantity in sorted(front_payload.get("quantities", []), key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "length" and unit_info(quantity.get("unit") or "") is not None
    ]
    text = str(front_payload.get("canonical_question") or "").lower()
    if len(lengths) < 3:
        if "equidistant" in text and (
            any(cue in text for cue in ["line connecting", "located on the line", "on the line connecting"])
            or any(cue in text for cue in ["away from the line", "away from line", "away from the line segment"])
        ):
            return None
        if "equidistant" in text and len(lengths) >= 1:
            separation_q = None
            target_distance_q = None
            for quantity in lengths:
                window = f"{quantity.get('context') or ''} {quantity.get('raw_text') or ''}".lower()
                if separation_q is None and any(cue in window for cue in ["separated", "apart", "distance ab", "ab ="]):
                    separation_q = quantity
                    continue
                if target_distance_q is None and any(cue in window for cue in ["equidistant", "by a distance", "from a and b"]):
                    target_distance_q = quantity
            if separation_q is None:
                separation_q = lengths[0]
            if target_distance_q is None:
                target_distance_q = lengths[-1] if len(lengths) > 1 else separation_q
            separation = _si_value(separation_q)
            distance = _si_value(target_distance_q)
            return {
                "d1": distance,
                "d2": distance,
                "separation": separation,
                "audit": {
                    "source_1_distance": target_distance_q.get("raw_text"),
                    "source_2_distance": target_distance_q.get("raw_text"),
                    "source_separation": separation_q.get("raw_text"),
                    "binding_policy": "equidistant_target_from_two_sources",
                },
            }
        return None
    separation_index = None
    for index, quantity in enumerate(lengths):
        window = f"{quantity.get('context') or ''} {quantity.get('raw_text') or ''}".lower()
        if any(cue in window for cue in ["separated", "apart", "between the two charges", "fixed and separated"]):
            separation_index = index
    if separation_index is None:
        separation_index = 2
    separation = _si_value(lengths[separation_index])
    point_lengths = [quantity for index, quantity in enumerate(lengths) if index != separation_index]
    if len(point_lengths) < 2:
        return None
    d1 = _si_value(point_lengths[0])
    d2 = _si_value(point_lengths[1])
    return {
        "d1": d1,
        "d2": d2,
        "separation": separation,
        "audit": {
            "source_1_distance": point_lengths[0].get("raw_text"),
            "source_2_distance": point_lengths[1].get("raw_text"),
            "source_separation": lengths[separation_index].get("raw_text"),
            "binding_policy": "ordered_distances_to_q1_q2_with_separation_context",
        },
    }


def _collinear_point_coordinate(d1: float, d2: float, separation: float) -> dict | None:
    if min(d1, d2, separation) <= 0:
        return None
    tolerance = 1e-9 * max(d1, d2, separation, 1.0)
    if abs((d1 + d2) - separation) <= tolerance:
        return {
            "template_id": "two_charges_collinear",
            "position_case": "between_sources",
            "source_separation_m": separation,
            "distance_to_source_1_m": d1,
            "distance_to_source_2_m": d2,
            "point_x": d1,
        }
    if abs((d2 - d1) - separation) <= tolerance:
        return {
            "template_id": "two_charges_collinear",
            "position_case": "outside_near_source_1",
            "source_separation_m": separation,
            "distance_to_source_1_m": d1,
            "distance_to_source_2_m": d2,
            "point_x": -d1,
        }
    if abs((d1 - d2) - separation) <= tolerance:
        return {
            "template_id": "two_charges_collinear",
            "position_case": "outside_near_source_2",
            "source_separation_m": separation,
            "distance_to_source_1_m": d1,
            "distance_to_source_2_m": d2,
            "point_x": separation + d2,
        }
    return None


def _collinear_direction_text(axis_sign: int, position_case: str, label1: str, label2: str) -> str:
    if position_case == "between_sources":
        return f"Toward {label2}" if axis_sign > 0 else f"Toward {label1}"
    if axis_sign > 0:
        return f"Along the {label1}-{label2} line, from {label1} toward {label2}"
    return f"Along the {label1}-{label2} line, from {label2} toward {label1}"


def _perpendicular_bisector_parameters(front_payload: dict) -> tuple[float | None, float | None]:
    lengths = [
        quantity
        for quantity in sorted(front_payload.get("quantities", []), key=lambda item: item.get("span") or (10**9, 10**9))
        if quantity.get("dimension") == "length" and unit_info(quantity.get("unit") or "") is not None
    ]
    if not lengths:
        return None, None
    separation = None
    height = None
    for quantity in lengths:
        context = f"{quantity.get('context') or ''} {quantity.get('raw_text') or ''}".lower()
        if height is None and any(cue in context for cue in ["away from", "perpendicular bisector", "distance h", "from ab"]):
            height = _si_value(quantity)
            continue
        if separation is None and (
            any(cue in context for cue in ["separated", "apart", "distance between"])
            or re.search(r"\b(?:segment|line segment)\s+ab\b", context)
            or re.search(r"\bab\s*=", context)
        ):
            separation = _si_value(quantity)
    if separation is None and lengths:
        separation = _si_value(lengths[0])
    if height is None and len(lengths) >= 2:
        height = _si_value(lengths[1])
    return separation, height


def _two_identical_source_charges_cue(text: str) -> bool:
    return bool(
        re.search(r"\btwo\s+(?:identical|equal|same)\s+charges\b", text)
        or re.search(r"\btwo\s+charges\s+(?:with\s+)?(?:equal|identical|same)\s+(?:magnitude|charge)", text)
        or re.search(
            r"\btwo\s+[+-]?\s*(?:\d+(?:\.\d+)?|\.\d+)"
            r"(?:\s*(?:×|x|\*)\s*10\s*\^?\s*[+-]?\d+)?\s*(?:μc|µc|uc|nc|pc|mc|c)\s+"
            r"(?:point\s+|electric\s+)?charges?\b",
            text,
        )
    )


def _match(template_id: str, confidence: float, evidence: List[str]) -> GeometryMatch:
    _ensure_template(template_id)
    return GeometryMatch(template_id, confidence, evidence)


def _has_three_side_triangle_evidence(text: str) -> bool:
    lowered = text.lower()
    has_named_triangle = re.search(r"\btriangle\s+([a-z]{3})\b", lowered) and len(re.findall(r"\b[a-z]{2}\s*=", lowered)) >= 2
    side_assignments = {
        "".join(sorted(match.group(1).lower()))
        for match in re.finditer(r"\b([a-z]{2})\s*=", lowered)
        if len(set(match.group(1).lower())) == 2 and not match.group(1).lower().startswith("q")
    }
    has_three_named_sides = len(side_assignments) >= 3
    generic_two_charge_point = (
        bool(re.search(r"\b(?:whose\s+)?distances?\s+to\b", lowered))
        and bool(re.search(r"\bfrom\s+(?:charge\s+)?q\w+\b", lowered))
        and len(re.findall(r"\bfrom\s+(?:charge\s+)?q\w+\b", lowered)) >= 2
    )
    return bool(has_named_triangle) or has_three_named_sides or generic_two_charge_point


def _unoccupied_square_vertex_cue(text: str) -> bool:
    return bool(
        any(
            cue in text
            for cue in [
                "fourth vertex",
                "remaining vertex",
                "empty vertex",
                "unoccupied vertex",
                "missing vertex",
                "vertex without a charge",
                "vertex with no charge",
                "remaining corner",
                "empty corner",
                "unoccupied corner",
                "corner without a charge",
                "corner with no charge",
            ]
        )
        or re.search(r"\b(?:field|force)\s+at\s+the\s+(?:other|last)\s+(?:vertex|corner)\b", text)
        or re.search(r"\b(?:other|last)\s+(?:vertex|corner)\s+(?:of|in)\s+the\s+square\b", text)
    )


def _has_right_triangle_evidence(text: str) -> bool:
    return bool(
        re.search(r"\bright[- ]?angled triangle\b", text)
        or re.search(r"\bright triangle\b", text)
        or re.search(r"\bright[- ]?angled\s+at\s+(?:point\s+)?[a-z]\b", text)
        or re.search(r"\bright angle\s+at\s+(?:point\s+)?[a-z]\b", text)
    )


def _has_right_isosceles_text(text: str) -> bool:
    return bool(
        "right isosceles triangle" in text
        or "isosceles right triangle" in text
        or "isosceles right-angled triangle" in text
        or "right-angled isosceles triangle" in text
        or "isosceles right angled triangle" in text
        or "right angled isosceles triangle" in text
    )


def _right_angle_point(text: str) -> str | None:
    return right_angle_point(text)


def _triangle_context(front_payload: dict) -> dict | None:
    return first_triangle_context(front_payload)


def _infer_triangle_labels_from_quantities(front_payload: dict) -> list[str]:
    labels: list[str] = []
    right_vertex = _right_angle_point(str(front_payload.get("canonical_question") or "").lower())
    if right_vertex:
        labels.append(right_vertex)
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


def _canonical_point(point: str | None, triangle_context: dict | None = None) -> str | None:
    return canonical_point(point, triangle_context)


def _canonical_side_key(symbol: str, triangle_context: dict | None = None) -> str | None:
    return canonical_side_key(symbol, triangle_context)


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
