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
   - Any LLM used in planning, extraction, explanation polishing, or routing must be open-source.
   - Maximum model size: **8B parameters**.

3. **External data usage must be fully disclosed**.
   - Any external datasets used for fine-tuning, retrieval, symbolic components, or evaluation must be declared.

## Prohibited

- Using closed-source/commercial LLMs such as GPT, Claude, or Gemini.
- Returning unverified numerical answers directly from an LLM.
- Hiding or failing to disclose external training/fine-tuning data.

## Physics Dataset

- **File**: `Physics_Problems_Text_Only.csv`
- **Current local scale**: 1,350 text-only problems
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
Qwen plans. Deterministic code solves. Verifier decides confidence.
```

The local model in `models/` is Qwen2.5-7B-Instruct. It should be used as a planner/router/extractor, not as the final numerical calculator.

Recommended architecture:

```text
cache
-> normalizer
-> implicit KB
-> deterministic router
-> fast solver
-> retrieval helper when needed
-> Qwen planner when needed
-> schema validator
-> unit converter
-> deterministic executor
-> verifier
-> trace-based explanation
-> answer checker
-> response
```

## Anti-Hallucination Requirements

- Qwen output must be treated as an untrusted proposal until validated.
- Qwen must not compute final numerical answers.
- Qwen must not invent formulas, constants, units, diagrams, assumptions, or code.
- Formula IDs, principle IDs, geometry templates, implicit rules, task types, and answer types must come from code-owned allowlists.
- Numerical answers must come from deterministic formula execution, SymPy, bounded numerical fallback, or deterministic geometry code.
- Every high-confidence numerical answer must pass unit checks and verifier checks.
- Explanations should be generated from execution trace, not from free-form LLM reasoning.

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
- [ ] Qwen is used only for planning/extraction/routing or guarded wording polish.
- [ ] Final numerical answers are produced by deterministic code, not by Qwen.
- [ ] Unit conversion and answer verification are deterministic.
- [ ] External data used for training, retrieval, or fine-tuning is documented.
- [ ] API output conforms to the required schema.
