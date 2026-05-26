# Physics XAI Core

NSP-Core is a neuro-symbolic Physics solver for electric circuits and
electrostatics. It covers resistance, voltage, current, power, capacitance,
electric field, Coulomb force, stored energy, LC/RLC circuits, inductance,
magnetic field, symbolic expressions, conceptual answers, yes/no checks,
vectors, and ordered multi-output answers.

The guiding rule is simple:

```text
frontend grounds facts
LLM chooses a structured solve path when enabled
compiler validates registry-backed steps
deterministic engines compute
verifier decides trust
proof DAG explains the result
```

The local fine-tuned LLM is a planner only. It may select a plan card or a
structured step DAG, but it never computes final numeric answers.

## Pipeline

```text
question
  -> core/api.py
  -> runtime/cache.py
  -> frontend/semantic_parser.py
  -> frontend/canonical.py
  -> engines/logic_engine.py for implicit facts
  -> knowledge/constraint_graph.py
  -> planning/local_llm.py when enabled
  -> planning/plan_compiler.py
  -> deterministic engine dispatch
       -> engines/multi_output.py
       -> engines/logic_engine.py
       -> engines/equation_engine.py
       -> engines/spatial_engine.py
       -> engines/algebraic_engine.py
  -> verification/verifier.py
  -> xai/trace.py
  -> xai/explanation.py
  -> verification/answer_check.py
  -> response
```

`core/pipeline.py` is the only orchestration layer. Runtime file logging is
disabled by default.

## Planning Modes

`llm_required` is the production XAI mode:

- deterministic frontend runs before the model;
- local LLM must produce an accepted plan card or structured plan;
- compiler validates every operation, registry ID, dependency, and output
  format;
- deterministic engines execute only the accepted path;
- if the LLM plan is missing or invalid, the system returns `Uncertain`.

`hybrid` is for coverage debugging. It tries LLM planning first but may fall
back to deterministic planning. `deterministic` skips the LLM and is used for
engine tests, dataset coverage, and fast regression checks.

Cache keys are scoped by planning mode.

## Formal IR

The frontend converts the question into a Formal IR:

```text
entities
quantities
symbolic_quantities
symbolic_relations
relations
constraints
states
events
goals
topology_graph
canonical_structures
```

The parser normalizes by physics role, not by example text. Labels such as
`ABC`, `MNQ`, `qA`, `qM`, `q1`, and `q′` are local surface labels that must map
to canonical roles before solving.

## LLM Contract

The LLM receives only a compact prompt:

```text
question summary
grounded facts
targets
geometry/topology hints
route-local formula menu
plan cards or operation templates
answer-format contract
```

Preferred compact output:

```json
{"status":"ok","plan_card_id":"p1"}
```

The card is selected by the model, then code expands it into executable steps.
This keeps the model in control of the reasoning path while reducing token
cost and malformed JSON risk.

Full structured output is also supported:

```json
{
  "status": "ok",
  "steps": [
    {
      "step_id": "s1",
      "operation": "construct_geometry",
      "geometry_constructor_id": "equilateral_triangle_vertex",
      "inputs": {"facts": "formal_ir.geometry"},
      "output": "geom",
      "depends_on": [],
      "public_cot": "Construct accepted geometry."
    },
    {
      "step_id": "s2",
      "operation": "compute_pairwise_force",
      "formula_id": "coulomb_force_triangle_sides",
      "principle_id": "coulomb_core",
      "inputs": {"geometry": "geom", "facts": "formal_ir.charges"},
      "output": "goal:1",
      "depends_on": ["s1"],
      "public_cot": "Resolve vector contributions."
    }
  ]
}
```

Allowed LLM authority:

- choose plan card or executable operations;
- choose known `formula_id`, `principle_id`, `geometry_constructor_id`, or
  `logic_rule_id`;
- provide short public `public_cot` action labels.

Forbidden LLM authority:

- final numeric answers;
- new formulas, constants, units, coordinates, diagrams, or code;
- hidden/free-form chain-of-thought;
- dataset answer or dataset CoT retrieval;
- bypassing compiler, engines, verifier, answer checker, or cache.

`public_cot` is a public action label, not hidden reasoning. It must avoid
arithmetic, equations, coordinates, final-answer wording, and new facts.

## Knowledge Layer

`knowledge/registries.py` owns reusable physics constraints. A formula card
declares:

```text
formula_id
task_type
principle_id
required_dimensions
target_dimension
target_unit
expression or deterministic branch marker
premise
execution_branch
```

`knowledge/constraint_graph.py` builds a bounded graph:

```text
known dimensions -> candidate formulas/principles -> target dimensions
```

Only a route-local, compact formula menu is shown to the LLM. The full registry
stays in code for compiler and verifier checks.

## Engines

Engine selection comes from the compiler-approved plan:

| Plan signal | Engine |
| --- | --- |
| ordered target branches | `engines/multi_output.py` |
| conceptual or yes/no rule | `engines/logic_engine.py` |
| scalar registry relation | `engines/equation_engine.py` |
| geometry/vector/superposition | `engines/spatial_engine.py` |
| small coupled symbolic system | `engines/algebraic_engine.py` |

Engines execute only code-owned registry relations. They normalize to SI
internally, construct geometry from templates, and return `Uncertain` rather
than guessing when grounding is ambiguous.

## Generalization Policy

Allowed generalization:

- dimension and target matching;
- entity/state scoped bindings;
- route-local formula families;
- topology constructors;
- geometry constructors;
- symbolic symmetry reductions derived from vector laws;
- proportional/conceptual rules grounded by known physical principles;
- uncertainty propagation and fail-closed verification.

Forbidden patterns:

- row-ID logic;
- exact question text branches;
- dataset-answer retrieval during inference;
- one-off formula cards for a single worksheet item;
- scalar fallback when a spatial plan was accepted;
- executing free-form LLM text.

The source test suite includes an audit that rejects embedded dataset row IDs
or known problem-instance strings in runtime source.

## Verification And XAI

Verification checks:

- compiler accepted the plan;
- solver executed a target-compatible registry path;
- unit and dimension compatibility;
- residuals where applicable;
- physical-domain constraints;
- vector and multi-path consistency when available;
- final answer string consistency.

The proof is a certified DAG:

```text
goal
  <- accepted plan steps
  <- facts/constants/unit conversions
  <- registry constraints
  -> result
  -> verifier
```

Explanations are generated from this proof DAG and execution trace, not from
free-form LLM prose.

## File Map

Core:

- `src/xai_pipeline/core/api.py`: request boundary.
- `src/xai_pipeline/core/pipeline.py`: orchestration, planning mode, dispatch,
  verification, response.

Frontend:

- `src/xai_pipeline/frontend/semantic_parser.py`: deterministic extraction.
- `src/xai_pipeline/frontend/canonical.py`: label, point, side, topology, and
  component normalization.
- `src/xai_pipeline/frontend/semantic_ir.py`: IR dataclasses.

Knowledge:

- `src/xai_pipeline/knowledge/registries.py`: formulas, principles, operations,
  constants, templates.
- `src/xai_pipeline/knowledge/constraint_graph.py`: route and candidate graph.
- `src/xai_pipeline/knowledge/formula_catalog.py`: route-local prompt menu.
- `src/xai_pipeline/knowledge/units.py`: SI and output-unit conversion.
- `src/xai_pipeline/knowledge/language.py`: shared wording/factor helpers.

Planning:

- `src/xai_pipeline/planning/local_llm.py`: LoRA runtime, Apple MPS config,
  compact prompt, JSON extraction, front repair, plan repair.
- `src/xai_pipeline/planning/solve_plan.py`: plan schema and deterministic
  debug planner.
- `src/xai_pipeline/planning/plan_compiler.py`: allowlist validation, DAG
  validation, engine order.
- `src/xai_pipeline/planning/answer_formats.py`: output contract.

Engines:

- `src/xai_pipeline/engines/equation_engine.py`: scalar/symbolic registry
  execution.
- `src/xai_pipeline/engines/spatial_engine.py`: geometry constructors and
  vector superposition.
- `src/xai_pipeline/engines/algebraic_engine.py`: CAS-lite equation subsets.
- `src/xai_pipeline/engines/logic_engine.py`: implicit facts and conceptual
  reasoning.
- `src/xai_pipeline/engines/multi_output.py`: ordered target branches.

Verification/XAI:

- `src/xai_pipeline/verification/verifier.py`: trust gate and confidence.
- `src/xai_pipeline/verification/answer_check.py`: final string consistency.
- `src/xai_pipeline/xai/trace.py`: proof DAG certificate.
- `src/xai_pipeline/xai/explanation.py`: trace-derived explanation.

Runtime/tests:

- `manual_question_test.py`: one-question runner with real LLM support.
- `run_50_dataset_llm_cot.py`: batch CSV runner.
- `tests/test_core_boundaries.py`: XAI/governance/edge tests.
- `tests/test_pipeline_dataset.py`: full dataset pipeline tests.
- `tests/test_real_dataset_front_pipeline.py`: full dataset frontend tests.
- `tests/test_real_llm_apple_mps.py`: opt-in real local LLM smoke test.

## Commands

Fast deterministic validation:

```bash
PYTHONPATH=src .venv/bin/python -m compileall -q src tests manual_question_test.py run_50_dataset_llm_cot.py
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python run_50_dataset_llm_cot.py \
  --no-llm --planning-mode deterministic --limit 0 \
  --output Physics_Problems_Text_Only_1347_llm.csv --request-timeout -1
git diff --check
```

Apple M1 Pro real LLM smoke:

```bash
XAI_RUN_REAL_LLM_TESTS=1 PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_real_llm_apple_mps.py
```

Manual Apple MPS run:

```bash
PYTHONPATH=src .venv/bin/python manual_question_test.py \
  --enable-llm \
  --planning-mode llm_required \
  --apple-mps \
  --max-new-tokens 32 \
  --llm-generate-max-time 60 \
  --llm-hard-timeout 0 \
  --require-llm-used \
  --require-llm-applied \
  --show-llm-raw \
  "A resistor R = 10 Ω has voltage U = 20 V. Find the current."
```

Expected dataset state after deleting the three underspecified rows:

```text
rows: 1347
verified: 1347
Uncertain: 0
errors: 0
```
