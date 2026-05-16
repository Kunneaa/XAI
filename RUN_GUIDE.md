# RUN GUIDE

This project is now **core-only** (no API layer). It focuses on the hybrid reasoning pipeline for the two competition datasets.

## 1) What To Run

Main files you will run:

- Core notebook:
  - `notebooks/core_pipeline.ipynb`
- Core pipeline modules:
  - `src/xai_pipeline/pipeline.py`
  - `src/xai_pipeline/logic_engine.py`
  - `src/xai_pipeline/physics_engine.py`
  - `src/xai_pipeline/retrieval.py`
  - `src/xai_pipeline/llm_client.py`
  - `src/xai_pipeline/prompt_registry.py`
- Training/evaluation scripts:
  - `training/prepare_logic_sft_data.py`
  - `training/train_sft.py`
  - `training/benchmark_core.py`
  - `training/benchmark_prompts.py`

## 2) Environment Setup

```bash
cd /Users/kunne/Kunne/Project/XAI
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3) Required Data Files

Expected at project root:

- `Logic_Based_Educational_Queries.train.json`
- `Logic_Based_Educational_Queries.test.json`
- `Physics_Problems_Text_Only.train.csv`
- `Physics_Problems_Text_Only.test.csv`

## 4) Runtime Configuration

Set environment variables before running:

- `XAI_USE_LLM`:
  - `1` = enable LLM
  - `0` = disable LLM (solver-first fallback mode)
- `XAI_MODEL_ID` (default: `Qwen/Qwen2.5-7B-Instruct`)
- `XAI_MODEL_PATH` (optional local model/checkpoint path)
- `XAI_DATA_SPLIT` (`train` or `test`)
- `XAI_PROMPT_VERSION` (`v1` or `v2`)

Example:

```bash
export XAI_USE_LLM=1
export XAI_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export XAI_MODEL_PATH=/absolute/path/to/model-or-checkpoint
export XAI_DATA_SPLIT=test
export XAI_PROMPT_VERSION=v2
```

Notes:
- Model guard enforces open-source <=8B policy heuristically by model name.
- If local path is invalid, LLM init fails and `LLM error` will show reason.

## 5) Run Core Notebook

Open and run:

- `notebooks/core_pipeline.ipynb`

Notebook flow:
1. Setup env defaults
2. Initialize `XAIPipeline`
3. Run sample logic + physics predictions


## 6) Prepare Logic SFT Data

```bash
cd /Users/kunne/Kunne/Project/XAI
source .venv/bin/activate
python training/prepare_logic_sft_data.py
```

Outputs:
- `training/logic_nl2fol_sft.jsonl`
- `training/logic_explainer_sft.jsonl`

## 7) Train SFT Checkpoint (Skeleton)

Example run:

```bash
cd /Users/kunne/Kunne/Project/XAI && source .venv/bin/activate && python3.12 training/train_sft.py --data training/logic_nl2fol_sft.jsonl --out training/checkpoints/logic_nl2fol --epochs 1 --bs 1 --lr 2e-5 --use-cpu
```

To use trained checkpoint in inference:

```bash
export XAI_MODEL_PATH=/Users/kunne/Kunne/Project/XAI/training/checkpoints/logic_nl2fol
```

## 8) Benchmark

Core benchmark:

```bash
cd /Users/kunne/Kunne/Project/XAI
source .venv/bin/activate
python training/benchmark_core.py
```

Prompt version A/B benchmark:

```bash
cd /Users/kunne/Kunne/Project/XAI
source .venv/bin/activate
python training/benchmark_prompts.py
```

## 9) Current Architecture (Quick)

- Dataset 1 (Logic):
  - NL->FOL (LLM translator, regex fallback) -> Z3 entailment -> explanation generation
- Dataset 2 (Physics):
  - Hybrid retrieval (BM25 + semantic) -> plan -> code generation/execution -> fallback
- Unified output always includes:
  - `answer`, `explanation`, optional reasoning fields (`fol`, `cot`, `premises`, `confidence`) and `meta`
