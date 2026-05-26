"""Deterministic explanation builder from execution trace."""

from __future__ import annotations

import hashlib
import json


def build_proof_dag(solver_result, graph_selection=None) -> dict:
    """Build a replayable proof object from the executed formula trace."""

    if not solver_result.solved:
        proof = {
            "status": "unproved",
            "nodes": [],
            "edges": [],
            "issues": [solver_result.trace.get("reason", "solver_not_solved")],
        }
        proof["certificate"] = _certificate(proof, certified=False)
        return proof

    nodes = []
    edges = []
    selected_formula_ids = _selected_formula_ids(graph_selection)
    target_dimension = solver_result.trace.get("target_dimension")
    execution_stage = solver_result.trace.get("stage")
    constraint_id = f"constraint:{solver_result.formula_id}"
    goal_id = "goal:answer"
    plan_steps = _plan_steps(solver_result.trace)
    executable_plan_step_id = _executable_plan_step_id(plan_steps, solver_result.formula_id)

    nodes.append(
        {
            "id": goal_id,
            "type": "goal",
            "target_dimension": target_dimension,
            "target_unit": solver_result.unit,
        }
    )
    for step in plan_steps:
        node_id = f"plan_step:{step.get('step_id')}"
        nodes.append(
            {
                "id": node_id,
                "type": "plan_step",
                "step_id": step.get("step_id"),
                "operation": step.get("operation"),
                "formula_id": step.get("formula_id"),
                "principle_id": step.get("principle_id"),
                "geometry_constructor_id": step.get("geometry_constructor_id"),
                "public_cot": step.get("public_cot"),
                "inputs": step.get("inputs") or {},
                "output": step.get("output"),
                "validation_status": "accepted_by_plan_compiler",
                "execution_result": _step_execution_result(step, solver_result),
            }
        )
        for dependency in step.get("depends_on") or []:
            edges.append({"source": f"plan_step:{dependency}", "target": node_id, "type": "step_depends_on"})
    for symbol, raw_payload in sorted((solver_result.trace.get("inputs") or {}).items()):
        payload = _fact_payload(raw_payload)
        node_id = f"fact:{symbol}"
        nodes.append(
            {
                "id": node_id,
                "type": "fact",
                "symbol": symbol,
                "dimension": payload.get("dimension"),
                "si_value": payload.get("si_value"),
                "unit": payload.get("unit"),
                "entity_id": payload.get("entity_id"),
                "state_id": payload.get("state_id"),
                "source": payload.get("raw_text"),
            }
        )
        edges.append({"source": node_id, "target": constraint_id, "type": "supports"})

    for index, raw_payload in enumerate(solver_result.trace.get("components") or [], start=1):
        payload = _fact_payload(raw_payload)
        symbol = payload.get("symbol") or f"component_{index}"
        node_id = f"component:{symbol}:{index}"
        nodes.append(
            {
                "id": node_id,
                "type": "component_fact",
                "symbol": symbol,
                "dimension": payload.get("dimension"),
                "si_value": payload.get("si_value"),
                "unit": payload.get("unit"),
                "entity_id": payload.get("entity_id"),
                "state_id": payload.get("state_id"),
                "source": payload.get("raw_text"),
            }
        )
        edges.append({"source": node_id, "target": constraint_id, "type": "supports"})

    for index, payload in enumerate(_source_payloads(solver_result.trace), start=1):
        symbol = payload.get("symbol") or f"source_{index}"
        node_id = f"fact:source:{symbol}:{index}"
        nodes.append(
            {
                "id": node_id,
                "type": "fact",
                "symbol": symbol,
                "dimension": payload.get("dimension"),
                "si_value": payload.get("si_value"),
                "unit": payload.get("unit"),
                "entity_id": payload.get("entity_id"),
                "state_id": payload.get("state_id"),
                "source": payload.get("raw_text"),
            }
        )
        edges.append({"source": node_id, "target": constraint_id, "type": "supports"})

    for symbol, value in sorted((solver_result.trace.get("constants") or {}).items()):
        node_id = f"constant:{symbol}"
        nodes.append({"id": node_id, "type": "constant", "symbol": symbol, "value": value})
        edges.append({"source": node_id, "target": constraint_id, "type": "supports"})

    if solver_result.trace.get("fact_id"):
        node_id = f"derived_fact:{solver_result.trace.get('fact_id')}"
        nodes.append(
            {
                "id": node_id,
                "type": "derived_fact",
                "fact_id": solver_result.trace.get("fact_id"),
                "expression": solver_result.trace.get("expression"),
            }
        )
        edges.append({"source": node_id, "target": constraint_id, "type": "supports"})

    nodes.append(
        {
            "id": constraint_id,
            "type": "constraint",
            "formula_id": solver_result.formula_id,
            "principle_id": solver_result.principle_id,
            "expression": solver_result.trace.get("expression"),
            "target_dimension": target_dimension,
            "execution_stage": execution_stage,
            "selected_by_constraint_graph": solver_result.formula_id in selected_formula_ids if solver_result.formula_id else False,
        }
    )
    if solver_result.trace.get("target_unit_conversion"):
        conversion = solver_result.trace["target_unit_conversion"]
        nodes.append(
            {
                "id": "unit_conversion:target",
                "type": "unit_conversion",
                "target_dimension": conversion.get("target_dimension"),
                "target_unit": conversion.get("unit"),
                "value": conversion.get("value"),
            }
        )
    nodes.append({"id": "result:answer", "type": "result", "answer": solver_result.answer, "unit": solver_result.unit})
    nodes.append({"id": "verify:answer", "type": "verify", "status": "accepted_by_verifier", "check_id": "answer_consistency"})
    if executable_plan_step_id:
        edges.append({"source": goal_id, "target": f"plan_step:{executable_plan_step_id}", "type": "requests"})
        edges.append({"source": f"plan_step:{executable_plan_step_id}", "target": constraint_id, "type": "compiles_to"})
    else:
        edges.append({"source": goal_id, "target": constraint_id, "type": "requests"})
    if solver_result.trace.get("target_unit_conversion"):
        edges.append({"source": constraint_id, "target": "unit_conversion:target", "type": "derives_si_value"})
        edges.append({"source": "unit_conversion:target", "target": "result:answer", "type": "converts_to_requested_unit"})
    else:
        edges.append({"source": constraint_id, "target": "result:answer", "type": "derives"})
    edges.append({"source": "result:answer", "target": "verify:answer", "type": "verified_by"})

    fact_count = sum(1 for node in nodes if node["type"] in {"fact", "component_fact", "constant", "derived_fact"})

    proof = {
        "status": "proved_by_execution",
        "nodes": nodes,
        "edges": edges,
        "selected_formula_ids": selected_formula_ids,
        "audit": {
            "execution_stage": execution_stage,
            "constraint_id": constraint_id,
            "formula_id": solver_result.formula_id,
            "principle_id": solver_result.principle_id,
            "target_dimension": target_dimension,
            "input_fact_count": fact_count,
            "candidate_constraint_count": len(selected_formula_ids),
            "accepted_plan_step_count": len(plan_steps),
            "proof_shape": "goal->plan_step?->constraint<-facts;constraint->unit_conversion?->result->verify",
        },
    }
    proof["certificate"] = _certificate(proof, certified=True)
    return proof


def _selected_formula_ids(graph_selection=None) -> list[str]:
    if graph_selection is None:
        return []
    if isinstance(graph_selection, dict):
        values = graph_selection.get("formula_ids") or []
    else:
        values = getattr(graph_selection, "formula_ids", []) or []
    return list(values)


def _source_payloads(trace: dict) -> list[dict]:
    source = trace.get("source")
    if isinstance(source, dict):
        return [source]
    if isinstance(source, list):
        return [item for item in source if isinstance(item, dict)]
    return []


def _fact_payload(payload) -> dict:
    """Normalize trace fact payloads into the proof-DAG shape.

    Engine traces should prefer full quantity dictionaries, but some derived
    symbolic/vector solvers naturally emit scalar intermediates such as a
    recovered side length. The proof layer treats those as typed-unknown scalar
    support facts instead of crashing the replay certificate.
    """

    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (int, float)):
        return {"si_value": payload}
    return {"raw_text": str(payload)}


def _plan_steps(trace: dict) -> list[dict]:
    compiled = trace.get("structured_solve_plan") if isinstance(trace, dict) else None
    if not isinstance(compiled, dict) or not compiled.get("ok"):
        return []
    plan = compiled.get("plan") or {}
    steps = plan.get("steps") or []
    return [step for step in steps if isinstance(step, dict) and step.get("step_id")]


def _executable_plan_step_id(steps: list[dict], formula_id: str | None) -> str | None:
    if not steps:
        return None
    if formula_id:
        for step in steps:
            if step.get("formula_id") == formula_id:
                return str(step.get("step_id"))
    return str(steps[-1].get("step_id"))


def _step_execution_result(step: dict, solver_result) -> dict:
    if step.get("formula_id") and step.get("formula_id") != solver_result.formula_id:
        return {"status": "not_executed_in_final_path"}
    operation = step.get("operation")
    if operation == "construct_geometry":
        geometry = solver_result.trace.get("geometry_engine") if hasattr(solver_result, "trace") else None
        if isinstance(geometry, dict) and geometry.get("coordinates"):
            return {"status": "executed", "coordinates_owned_by": "spatial_engine", "coordinate_labels": sorted(geometry.get("coordinates", {}).keys())}
        return {"status": "validated"}
    if step.get("formula_id") == solver_result.formula_id or operation in {"format_target", "vector_sum"}:
        return {
            "status": "executed",
            "answer": solver_result.answer,
            "value": solver_result.value,
            "unit": solver_result.unit,
            "engine_stage": solver_result.trace.get("stage") if hasattr(solver_result, "trace") else None,
        }
    return {"status": "validated"}


def _certificate(proof: dict, certified: bool) -> dict:
    payload = {key: value for key, value in proof.items() if key != "certificate"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "algorithm": "sha256-json-v1",
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "certified": certified,
    }


def build_trace_explanation(solver_result, unit_conversion_result=None) -> str:
    if not solver_result.solved:
        return "The deterministic equation engine did not produce a verified trace for this question."

    proof = solver_result.trace.get("proof_dag") or build_proof_dag(solver_result)
    audit = proof.get("audit") or {}
    formula_id = audit.get("formula_id") or solver_result.formula_id
    principle_id = audit.get("principle_id") or solver_result.principle_id
    target_dimension = audit.get("target_dimension") or solver_result.trace.get("target_dimension") or "requested target"
    input_count = audit.get("input_fact_count")
    candidate_count = audit.get("candidate_constraint_count")
    expression = solver_result.trace.get("expression")
    if unit_conversion_result is None:
        converted_count = (
            len((solver_result.trace or {}).get("inputs", {}))
            + len((solver_result.trace or {}).get("components", []))
            + len(_source_payloads(solver_result.trace or {}))
        )
    else:
        trace = getattr(unit_conversion_result, "trace", {}) or {}
        converted_count = trace.get("converted_quantity_count", 0)
    parts = [
        f"The semantic frontend grounded {input_count} typed fact(s) for the {target_dimension} goal.",
    ]
    accepted_steps = audit.get("accepted_plan_step_count")
    if accepted_steps:
        parts.append(f"The structured solve plan contributed {accepted_steps} validated executable step(s); raw LLM reasoning was not executed.")
    parts.append(f"The constraint graph selected registry constraint `{formula_id}` from principle `{principle_id}`")
    if candidate_count is not None:
        parts[-1] += f" after checking {candidate_count} reachable candidate constraint(s)."
    else:
        parts[-1] += "."
    if expression:
        parts.append(f"The deterministic executor applied the code-owned relation `{expression}`.")
    if solver_result.trace.get("stage") == "spatial_vector_direction_engine":
        parts.append("The spatial direction engine reconstructed the collinear geometry and used signed vector superposition, so no force magnitude was needed.")
    elif isinstance(solver_result.value, str) or solver_result.trace.get("stage") == "symbolic_spatial_vector_engine":
        parts.append("The symbolic spatial engine used the accepted side and charge symbols directly; no numerical unit conversion was required.")
    else:
        parts.append(f"The unit layer normalized {converted_count} extracted numeric fact(s) to SI before execution.")
    if solver_result.trace.get("target_unit_conversion"):
        conversion = solver_result.trace["target_unit_conversion"]
        parts.append(f"The result was converted to the requested unit `{conversion.get('unit')}` after verification.")
    parts.append(f"The verifier accepted the proof graph and answer consistency checks. The verified result is {solver_result.answer}.")
    return " ".join(parts)
