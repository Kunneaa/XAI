import argparse
import json
import os
import inspect
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_text(row):
    return (
        "### Instruction\n" + row["instruction"] + "\n\n"
        "### Input\n" + row["input"] + "\n\n"
        "### Output\n" + row["output"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    ap.add_argument("--max-len", type=int, default=512, help="Max token length")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--eval-ratio", type=float, default=0.1, help="Validation split ratio")
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--early-stopping-patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-from-checkpoint", default=None)
    ap.add_argument("--use-cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = ap.parse_args()

    use_cpu = args.use_cpu or not torch.cuda.is_available()
    use_cuda = not use_cpu

    # Convert to absolute path if relative
    model_path = Path(args.model).resolve()
    device_map = "cpu" if use_cpu else "auto"

    # T4 supports fp16 well; bf16 is not supported.
    model_dtype = torch.float32 if use_cpu else torch.float16

    # Check if it's a local path, load locally; otherwise load from HF
    if model_path.exists():
        tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=model_dtype,
            device_map=device_map,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=model_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id

    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    rows = load_jsonl(Path(args.data))
    ds = Dataset.from_list([{"text": build_text(r)} for r in rows])

    def tok_fn(batch):
        x = tok(batch["text"], truncation=True, max_length=args.max_len)
        x["labels"] = x["input_ids"].copy()
        return x

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    targs_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=0.5,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=use_cuda,
        bf16=False,
        dataloader_pin_memory=use_cuda,
        optim="adamw_torch_fused" if use_cuda else "adamw_torch",
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    # Transformers compatibility: some versions use eval_strategy/save_strategy,
    # while others use evaluation_strategy/save_strategy.
    if "evaluation_strategy" in ta_params:
        targs_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_params:
        targs_kwargs["eval_strategy"] = "steps"

    if "save_strategy" in ta_params:
        targs_kwargs["save_strategy"] = "steps"

    targs = TrainingArguments(**targs_kwargs)

    device_label = "cpu"
    if use_cuda:
        device_label = f"cuda ({torch.cuda.get_device_name(0)})"
    print(f"[train_sft] device={device_label}")
    print(
        f"[train_sft] bs={args.bs}, grad_accum={args.grad_accum}, "
        f"max_len={args.max_len}, epochs={args.epochs}, lr={args.lr}"
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
