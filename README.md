# XAI Challenge Summary

This repository contains two datasets and a concise guide to the core requirements of the XAI competition.

## Competition Goals

Systems are evaluated on three dimensions:

- **P1: Correctness of Answers** - accuracy and precision of final answers.
- **P2: Quality of Explanation** - clear, coherent, and verifiable natural-language justification.
- **P3: Depth of Reasoning** - evidence of structured reasoning (e.g., premises, FOL, derivations, stepwise logic).

## Mandatory Rules

1. **Every answer must include an explanation**.
   - Explanations should be concise, interpretable, and verifiable.

2. **Only open-source LLMs are allowed**.
   - Any LLM used in the system (answering, reasoning, NL-to-logic, etc.) must be open-source.
   - Maximum model size: **8B parameters**.

3. **External data usage must be fully disclosed**.
   - All external datasets used for fine-tuning LLMs or symbolic components must be declared.

## Prohibited

- Using closed-source/commercial LLMs (e.g., GPT, Claude, Gemini).
- Hiding or failing to disclose external training/fine-tuning data.

Violations can lead to **disqualification**.

## Encouraged Approach

- Integrate a symbolic reasoning component (e.g., Z3 or custom engine) to verify results and strengthen explainability.
- Symbolic reasoning is encouraged, not mandatory.

## Dataset Overview

### 1) Logic-Based Educational Queries

- **File**: `Logic_Based_Educational_Queries.json`
- **Scale**: 464 records, 913 questions.
- **Domain**: university policies and regulations (grading, enrollment, scholarships, requirements, etc.).
- **Question types**: Multiple Choice, Yes/No/Uncertain, open-ended.
- **Provided fields include**:
  - Premises in natural language (`premises-NL`)
  - Premises in FOL
  - Questions
  - Ground-truth answers
  - Human-written explanations
- **Evaluation-time input**: question + natural-language premises.
- Teams may process premises in any way (prompt context, FOL conversion, symbolic solving, etc.).

### 2) Physics Problems

- **File**: `Physics_Problems_Text_Only.csv`
- **Scale**: 5,520 text-only problems.
- **Domain**: electric circuits and electrostatics (resistance, voltage, current, power, capacitance, electric fields, energy).
- **Nature**: numerical, multi-step computation.
- **Dataset annotations**: step-by-step CoT and final numerical answer with unit.
- **Evaluation-time input**: **question only** (no extra context provided).
- Source references used to build this dataset will be announced at the kick-off workshop.

## Test Format

The official test set combines both dataset types:

- Type 1: question + premises-NL
- Type 2: question only

Question formats may include:

- Multiple choice
- Yes/No/Uncertain
- Open-ended reasoning
- Numerical computation

Topic distribution percentages will be announced at the kick-off workshop.

## Evaluation Process

- **Phase 1 & 2 (Selection)**:
  - Automatic scoring against ground truth
  - Committee review of explanation quality
- **Final Round**:
  - Live run on unseen queries
  - Challenge Chairs directly evaluate answer quality, explanation quality, and reasoning depth
- **Final score**:
  - Weighted combination of P1, P2, P3
  - Exact weights released with official dataset release

## Submission Requirements

Each team must submit:

1. An **API endpoint**
2. A **1-page solution description** including:
   - approach
   - models used
   - datasets used for training

### API Output Schema

Required fields:

- `answer`
- `explanation`

Optional but encouraged fields (help with reasoning-depth evaluation):

- `fol`
- `cot`
- `premises`
- `confidence`

Example:

```json
{
  "answer": "B",
  "explanation": "The voltage across R2 is calculated using ...",
  "fol": "∀x (Resistor(x) → HasVoltage(x, V))",
  "cot": [
    "Step 1: Identify the circuit topology ...",
    "Step 2: Apply Kirchhoff's voltage law ...",
    "Step 3: Solve for the unknown voltage ..."
  ],
  "premises": [
    "Ohm's law: V = IR",
    "KVL: sum of voltages in a loop = 0"
  ],
  "confidence": 0.92
}
```

> Note: The final submission format may be refined at the kick-off workshop.

## Practical Compliance Checklist

- [ ] Every response includes both `answer` and `explanation`.
- [ ] All LLM components are open-source and <= 8B parameters.
- [ ] No closed-source models are used anywhere in the pipeline.
- [ ] All external training/fine-tuning data are documented.
- [ ] Reasoning artifacts (FOL/steps/premises/confidence) are included when possible.
- [ ] API output conforms to required JSON fields.

