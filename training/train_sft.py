import argparse
import json
import os
import platform
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
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
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--use-cpu", action="store_true", help="Force CPU instead of GPU (recommended for Mac)")
    args = ap.parse_args()

    # On Mac, prefer CPU for stability
    is_mac = platform.system() == "Darwin"
    use_cpu = args.use_cpu or is_mac
    
    # Disable MPS for stability
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Convert to absolute path if relative
    model_path = Path(args.model).resolve()
    
    # Determine device
    device_map = "cpu" if use_cpu else "auto"
    
    # Check if it's a local path, load locally; otherwise load from HF
    if model_path.exists():
        tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), 
            torch_dtype="float32",
            device_map=device_map,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype="float32",
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id

    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    rows = load_jsonl(Path(args.data))
    ds = Dataset.from_list([{"text": build_text(r)} for r in rows])

    def tok_fn(batch):
        x = tok(batch["text"], truncation=True, max_length=256)
        x["labels"] = x["input_ids"].copy()
        return x

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        report_to="none",
        gradient_accumulation_steps=1,
        max_grad_norm=0.5,
        weight_decay=0.01,
        warmup_steps=50,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
