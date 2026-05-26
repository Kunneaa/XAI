# Physics XAI Challenge Summary

This repository is scoped to the **Physics** part of the XAI challenge only.

The goal is to build an explainable Physics QA system that returns accurate answers with concise, verifiable explanations while using only open-source LLM components with model size <= 8B parameters.

## Evaluation Goals

Systems are evaluated on three dimensions:

- **P1: Correctness of Answers** - accuracy and precision of final Physics answers.
- **P2: Quality of Explanation** - clear, coherent, and verifiable natural-language justification.
- **P3: Depth of Reasoning** - evidence of structured reasoning, formulas, derivations, unit conversions, and verification.

## Mandatory Rules

1. **Every answer must include an explanation**.
   - Explanations should be concise, interpretable, and verifiable.

2. **Only open-source LLMs are allowed**.
   - Any LLM used in semantic parsing or bounded repair must be open-source.
   - Maximum model size: **8B parameters**.

3. **External data usage must be fully disclosed**.
   - The primary local data file is `Physics_Problems_Text_Only.csv`.
   - Any additional data used for fine-tuning, symbolic components, or evaluation must be declared.

## Prohibited

- Using closed-source/commercial LLMs such as GPT, Claude, or Gemini.
- Returning unverified numerical answers directly from an LLM.
- Hiding or failing to disclose external training/fine-tuning data.

## Physics Dataset

- **File**: `Physics_Problems_Text_Only.csv`
- **Current local scale**: 1,347 text-only problems
- **Columns**: `id`, `question`, `cot`, `answer`, `unit`
- **Missing fields**: none in the current local file
- **Duplicate ids/questions**: none in the current local file
- **Domain**: electric circuits and electrostatics, including resistance, voltage, current, power, capacitance, electric fields, Coulomb force, energy, LC/RLC circuits, inductance, and magnetic fields.
- **Evaluation-time input**: question only

## Physics Answer Types

The system must support:

- Numeric answers with units
- Symbolic expressions
- Conceptual answers
- Yes/No answers
- Multi-output answers such as `A; A`, `cm; %`, or `μC; μJ`
- Vector/electrostatic geometry problems

Common units in the local file include:

```text
N, V/m, -, V, J, Ω, A, W, μF, nC, pF, %, mJ, nJ, Hz, H, mH, N/C, cm, C
```

## Proposed System Design

The current system design is documented in:

- `Physics_XAI_Core.md`

Core rule:

```text
Deterministic code normalizes facts and units.
Fine-tuned local LLM may propose a schema-bound Structured Solve Plan.
Plan compiler validates registry-backed executable steps.
Deterministic code solves. Verifier decides confidence.
```

The current local model artifact in `models/` is:

```text
adapter: models/deepseek-r1-distill-qwen-7b-exact-lora
type: PEFT LoRA SFT adapter
base model required at runtime: models/DeepSeek-R1-Distill-Qwen-7B
tokenizer: Qwen2Tokenizer
```

The repository currently contains the fine-tuned adapter, not the full base model.
Runtime loading must attach this adapter to the matching open-source base model
declared in `adapter_config.json`. If the base model is absent, the system must
disable LLM-assisted paths and continue with deterministic-only behavior.

The fine-tuned model may be used for structured solve-plan proposal, semantic
audit, or bounded repair when that path is enabled. It must not be used as the
final numerical calculator.

Current core architecture:

```text
runtime/cache
-> frontend/semantic_parser
-> frontend/canonical structure normalization
-> engines/logic_engine
-> knowledge/constraint_graph
-> knowledge/formula_catalog
-> planning/solve_plan
-> planning/plan_compiler
   -> if invalid: strict error packet -> one local LLM re-plan attempt
-> grounded conceptual solver when applicable
-> engines/equation_engine
-> engines/spatial_engine
-> verification/verifier
-> xai/trace / xai/explanation
-> answer checker
-> runtime/telemetry / response
```

## Anti-Hallucination Requirements

- Fine-tuned LLM output must be treated as an untrusted proposal until validated.
- LLM "CoT" must be represented as per-step `public_cot` labels inside a Structured Solve Plan, not free-form reasoning.
- The fine-tuned LLM must not compute final numerical answers.
- The fine-tuned LLM must not invent formulas, constants, units, diagrams, assumptions, or code.
- No raw natural-language LLM reasoning may be executed.
- Formula IDs, principle IDs, geometry templates, implicit rules, task types, and answer types must come from code-owned allowlists.
- Numerical answers must come from deterministic formula execution, CAS-lite registry equation graphs, or deterministic geometry code.
- Every high-confidence numerical answer must pass unit checks and verifier checks.
- Explanations should be generated from the Proof DAG and execution trace, not from free-form LLM reasoning.
- Public reasoning must describe the structure of the proof path: semantic facts, constraint graph selection, deterministic execution, unit handling, and verifier acceptance.
- Each question type uses a code-owned output-format contract: numeric scalar, dimensionless numeric, symbolic expression, conceptual text, yes/no, ordered multi-output, or controlled fallback.
- Repeated quantities must be scoped by entity/state before contradiction checks.
- Surface labels must be canonicalized before solving, so `ABC`, `MNQ`, `PQR`, `q1/q2`, and `qM/qN` can map to the same structural problem when the evidence supports it.
- Compound symbolic relations such as `LCω² = 1` must not become hidden-unit quantities.
- Dataset-shaped formula cards such as one-off quadrature/segment patterns must not stay in the active registry.
- Geometry-specific symbolic answers must be derived from general vector laws plus deterministic coordinate constructors, not stored as one-off formula cards.
- Direct scalar circuit formulas must abstain when multiple components exist without canonical topology.
- Explicit simple series/parallel topology may be solved by deterministic topology rules; ambiguous branch/node topology still abstains.
- Conceptual and Yes/No answers must be grounded by derived logic facts or registry-owned SI-unit facts.
- Cheap redundant formula paths should be compared when available.
- Multi-output answers should be solved as ordered target branches, not collapsed into one scalar target.
- Local LLM plan proposal/refinement/repair must fail closed unless a schema-bound local backend is available.
- Invalid LLM plans may be repaired once from a compiler error packet; unvalidated repaired plans are never executed.
- Proof DAGs should carry reproducible audit certificates.

Current deterministic smoke status:

```text
unit/full tests: 56 passed, 1353 subtests passed
dataset batch: 1347 / 1347 verified, 0 Uncertain, 0 errors
active executable registry: 151 formula IDs, 195 total registry cards
registry family coverage: 11 law families, 0 uncovered executable IDs
```

Manual local model runner:

```text
PYTHONPATH=src .venv/bin/python3 manual_question_test.py --llm-status
PYTHONPATH=src .venv/bin/python3 manual_question_test.py --llm-probe --max-new-tokens 16 --llm-hard-timeout 45
PYTHONPATH=src .venv/bin/python3 manual_question_test.py --direct-llm --max-new-tokens 64 --llm-hard-timeout 70 "A resistor R = 10 Ω has voltage U = 20 V. Find the current I."
PYTHONPATH=src .venv/bin/python3 manual_question_test.py --direct-llm --timeout -1 --llm-generate-max-time 0 --llm-hard-timeout 0 --max-new-tokens 256 --output-json /private/tmp/manual_result.json "A resistor R = 10 Ω has voltage U = 20 V. Find the current I."
```

The LoRA adapter is present at `models/deepseek-r1-distill-qwen-7b-exact-lora`.
Actual generation also requires the local base model directory
`models/DeepSeek-R1-Distill-Qwen-7B` and optional dependencies listed in
`requirements.txt`. On Apple Silicon, use `--apple-mps` to select `mps`,
`float16`, JSON prefill, and JSON early stop. In `llm_required` mode the system
fails closed if the model is unavailable, too slow, or emits invalid JSON; in
`hybrid` mode deterministic planning may be used for coverage debugging.

## API Output Schema

Required fields:

- `answer`
- `explanation`

Recommended fields:

- `cot`
- `premises`
- `confidence`
- `metadata`

Example:

```json
{
  "answer": "0.045 J",
  "explanation": "Use W = 1/2 C U^2. Convert C = 100 μF = 100e-6 F. Substitute U = 30 V, so W = 0.045 J.",
  "cot": [
    "Identify capacitance and voltage.",
    "Convert capacitance to SI units.",
    "Apply the verified capacitor energy relation."
  ],
  "premises": [
    "W = 1/2 C U^2",
    "1 μF = 10^-6 F"
  ],
  "confidence": 0.95
}
```

## Practical Compliance Checklist

- [ ] Every response includes both `answer` and `explanation`.
- [ ] All LLM components are open-source and <= 8B parameters.
- [ ] No closed-source models are used anywhere in the pipeline.
- [ ] The fine-tuned local LLM is used only for structured solve-plan proposal, semantic audit, or bounded repair when enabled.
- [ ] Final numerical answers are produced by deterministic code, not by the fine-tuned LLM.
- [ ] Unit conversion and answer verification are deterministic.
- [ ] Unverified deterministic candidates are exposed only as audit artifacts, never as final answers.
- [ ] Proof traces include a certificate/digest for replay and regression checks.
- [ ] External data used for training or fine-tuning is documented.
- [ ] API output conforms to the required schema.
