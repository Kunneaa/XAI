"""Answer-format contracts for structured physics solve plans.

The LLM proposes a plan, not an answer. These contracts tell the planner and
compiler what shape the deterministic answer must have for each question type.
They are deliberately small and code-owned so prompt wording cannot redefine
the response schema.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


ANSWER_FORMAT_KINDS = frozenset(
    {
        "numeric_scalar",
        "dimensionless_numeric",
        "symbolic_expression",
        "conceptual_text",
        "yes_no",
        "ordered_multi_output",
        "controlled_fallback",
    }
)


def build_output_format(front_payload: dict, answer_type: str, task_type: str, targets: list[dict]) -> dict:
    """Return the deterministic output-format contract for a plan.

    The contract is part of the plan boundary. Engines still own the final
    values, but the planner must declare whether it expects a numeric scalar,
    symbolic expression, conceptual text, yes/no label, or ordered multi-output.
    """

    ordered_targets = [str(target.get("id")) for target in targets if target.get("id")]
    preferred_unit = None if answer_type in {"conceptual", "yes_no", "unknown"} else _preferred_unit(targets)
    contract = _base_contract(answer_type, preferred_unit)
    contract.update(
        {
            "ordered_targets": ordered_targets,
            "preferred_unit": preferred_unit,
            "target_count": len(ordered_targets),
        }
    )
    if task_type in {"coulomb_force", "electric_field_point", "resultant_force"}:
        contract["vector_policy"] = "include_magnitude; include_direction_when_requested_or_derivable"
    if answer_type == "multi_output":
        contract["items"] = [
            {
                "target_id": str(target.get("id")),
                "quantity": target.get("quantity"),
                "unit": target.get("unit"),
            }
            for target in targets
        ]
    if front_payload.get("answer_type_hint") == "symbolic":
        contract["symbolic_policy"] = "preserve accepted symbols; do not coerce to numeric values"
    return contract


def prompt_answer_format_contract(front_payload: dict, route_result=None, targets: Iterable[dict] | None = None) -> dict:
    """Return a compact prompt-safe format contract for the current route."""

    answer_type = str(front_payload.get("answer_type_hint") or getattr(route_result, "answer_type", "unknown"))
    task_type = str(getattr(route_result, "task_type", None) or "unknown")
    target_list = list(targets or _prompt_targets(front_payload))
    return {
        "current": build_output_format(front_payload, answer_type, task_type, target_list),
        "allowed_format_kinds": sorted(ANSWER_FORMAT_KINDS),
        "by_answer_type": {
            "numeric": _base_contract("numeric", _preferred_unit(target_list)),
            "symbolic": _base_contract("symbolic", _preferred_unit(target_list)),
            "conceptual": _base_contract("conceptual", None),
            "yes_no": _base_contract("yes_no", None),
            "multi_output": _base_contract("multi_output", _preferred_unit(target_list)),
            "unknown": _base_contract("unknown", None),
        },
        "policy": [
            "The plan declares output shape only; deterministic engines produce the value.",
            "Do not place final values, arithmetic, or free-form reasoning in output_format.",
            "For vector quantities, plan vector operations and let the spatial engine compute magnitude and direction.",
        ],
    }


def output_format_issues(plan: dict) -> list[str]:
    """Validate the output-format section of a structured plan."""

    issues: list[str] = []
    output_format = plan.get("output_format")
    if not isinstance(output_format, dict):
        return ["output_format_not_object"]
    kind = output_format.get("format_kind")
    if kind not in ANSWER_FORMAT_KINDS:
        issues.append(f"unknown_output_format_kind:{kind}")
    if plan.get("status") != "ok":
        if kind != "controlled_fallback":
            issues.append(f"non_ok_plan_must_use_controlled_fallback:{kind}")
        ordered_targets = output_format.get("ordered_targets")
        if not isinstance(ordered_targets, list):
            issues.append("output_format_missing_ordered_targets")
        return issues
    answer_type = plan.get("answer_type")
    expected = _expected_kind_for_answer_type(str(answer_type), output_format)
    if expected and kind != expected:
        issues.append(f"answer_type_output_format_mismatch:{answer_type}:{kind}:{expected}")
    ordered_targets = output_format.get("ordered_targets")
    if not isinstance(ordered_targets, list):
        issues.append("output_format_missing_ordered_targets")
    if answer_type == "multi_output":
        if output_format.get("separator") != "; ":
            issues.append("multi_output_separator_must_be_semicolon_space")
        if not isinstance(output_format.get("items"), list) or not output_format.get("items"):
            issues.append("multi_output_missing_items")
    if answer_type == "yes_no":
        labels = output_format.get("allowed_labels")
        if labels != ["Yes", "No"]:
            issues.append("yes_no_allowed_labels_must_be_yes_no")
    return issues


def _base_contract(answer_type: str, preferred_unit: str | None) -> Dict[str, Any]:
    if answer_type == "numeric":
        if preferred_unit in {None, "-", "%", "times"}:
            return {
                "format_kind": "dimensionless_numeric" if preferred_unit in {"-", "%", "times"} else "numeric_scalar",
                "value_template": "<number>" if preferred_unit in {None, "-"} else f"<number> {preferred_unit}",
                "requires_unit": preferred_unit not in {None, "-"},
                "unit_policy": "use preferred_unit when present; otherwise use target base unit",
            }
        return {
            "format_kind": "numeric_scalar",
            "value_template": f"<number> {preferred_unit}",
            "requires_unit": True,
            "unit_policy": "convert verified SI value to preferred_unit",
        }
    if answer_type == "symbolic":
        return {
            "format_kind": "symbolic_expression",
            "value_template": "<symbolic expression>[, direction if vector]",
            "requires_unit": False,
            "unit_policy": "preserve dimensions symbolically; include unit only when requested",
        }
    if answer_type == "conceptual":
        return {
            "format_kind": "conceptual_text",
            "value_template": "<concise principle-grounded sentence>",
            "requires_unit": False,
            "unit_policy": "not applicable",
        }
    if answer_type == "yes_no":
        return {
            "format_kind": "yes_no",
            "value_template": "Yes|No",
            "allowed_labels": ["Yes", "No"],
            "requires_unit": False,
            "unit_policy": "not applicable",
        }
    if answer_type == "multi_output":
        return {
            "format_kind": "ordered_multi_output",
            "value_template": "<target_1>; <target_2>; ...",
            "separator": "; ",
            "requires_unit": True,
            "unit_policy": "each item uses its own verified target unit",
        }
    return {
        "format_kind": "controlled_fallback",
        "value_template": "Uncertain",
        "requires_unit": False,
        "unit_policy": "not applicable",
    }


def _expected_kind_for_answer_type(answer_type: str, output_format: dict) -> str | None:
    if answer_type == "numeric":
        kind = output_format.get("format_kind")
        if kind in {"numeric_scalar", "dimensionless_numeric"}:
            return str(kind)
        return "numeric_scalar"
    return {
        "symbolic": "symbolic_expression",
        "conceptual": "conceptual_text",
        "yes_no": "yes_no",
        "multi_output": "ordered_multi_output",
        "unknown": "controlled_fallback",
    }.get(answer_type)


def _prompt_targets(front_payload: dict) -> list[dict]:
    out = []
    for index, goal in enumerate(front_payload.get("goals") or [], start=1):
        out.append(
            {
                "id": goal.get("goal_id") or f"target:{index}",
                "quantity": goal.get("dimension") or "unknown",
                "unit": None,
            }
        )
    return out or [{"id": "target:1", "quantity": "unknown", "unit": None}]


def _preferred_unit(targets: Iterable[dict]) -> str | None:
    for target in targets:
        unit = target.get("unit") if isinstance(target, dict) else None
        if unit:
            return str(unit)
    return None
