#!/usr/bin/env python3
"""
Parameter-efficient (LoRA) fine-tuning for Track-1 STOP/CONTINUE on TraceTriage prefix data.

Why this script:
- Keeps the same data protocol and metrics as existing full fine-tune scripts.
- Reduces trainable params via LoRA for higher efficiency on CPU-only servers.
- Exports a merged HuggingFace model for compatibility with existing eval scripts.
"""

import argparse
import csv
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import run_track1_baselines as b


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_input(row: Dict[str, str], label_field: str) -> str:
    metrics = (row.get("prefix_metrics") or "").strip()
    excerpt = (row.get("prefix_log_excerpt") or "").strip()
    platform = row.get("platform", "")
    design = row.get("design", "")
    ckpt = row.get("checkpoint", "")
    elapsed = row.get("prefix_elapsed_seconds", "")
    run_params_text = b.format_run_params_text_for_model_input(row)
    return (
        "You are an EDA triage model.\n"
        "Task: Decide whether to STOP or CONTINUE for this EDA checkpoint.\n"
        "Goal: Maximize net runtime savings while controlling false stops on successful runs.\n"
        "Decision policy: Treat STOP and CONTINUE as equally valid baseline decisions.\n"
        f"platform={platform}\n"
        f"design={design}\n"
        f"checkpoint={ckpt}\n"
        f"elapsed_seconds={elapsed}\n"
        f"run_params={run_params_text}\n"
        f"prefix_metrics={metrics}\n"
        f"log_excerpt={excerpt}\n"
        f"Output policy target: CONTINUE for {label_field}=1, STOP for {label_field}=0."
    )


def load_rows(path: str, label_field: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            label = (r.get(label_field) or "").strip()
            split = (r.get("split") or "").strip()
            if split not in {"train", "val", "test"}:
                continue
            if label not in {"0", "1"}:
                continue
            rows.append(r)
    return rows


def build_split(rows: List[Dict[str, str]], split: str, max_samples: int, seed: int) -> List[Dict[str, str]]:
    subset = [r for r in rows if r.get("split") == split]
    rng = random.Random(seed)
    rng.shuffle(subset)
    if max_samples > 0:
        subset = subset[:max_samples]
    return subset


def to_hf_dataset(rows: List[Dict[str, str]], label_field: str) -> Dataset:
    records = []
    for r in rows:
        label = int(r[label_field])
        records.append(
            {
                "text": format_input(r, label_field=label_field),
                "label": label,
            }
        )
    return Dataset.from_list(records)


def parse_target_modules(s: str) -> List[str]:
    xs = [x.strip() for x in (s or "").split(",")]
    return [x for x in xs if x]


def parse_torch_dtype(name: str) -> torch.dtype | None:
    x = (name or "auto").strip().lower()
    if x in {"auto", ""}:
        return None
    if x in {"fp16", "float16", "half"}:
        return torch.float16
    if x in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if x in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported --torch-dtype: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-field", default="strict_success")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module suffixes for LoRA injection.",
    )

    ap.add_argument("--use-gradient-checkpointing", action="store_true")
    ap.add_argument("--save-merged", action="store_true", default=True)
    ap.add_argument("--ddp-find-unused-parameters", action="store_true")
    ap.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
        help="Base model load dtype. Use bfloat16/float16 to reduce memory on larger models.",
    )
    ap.add_argument("--use-fp16", action="store_true")
    ap.add_argument("--use-bf16", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_rows(args.manifest, label_field=str(args.label_field or "strict_success"))
    train_rows = build_split(rows, "train", args.max_train_samples, args.seed)
    val_rows = build_split(rows, "val", args.max_val_samples, args.seed)

    print(
        f"Loaded rows: total={len(rows)} train={len(train_rows)} val={len(val_rows)} "
        f"label_field={args.label_field}",
        flush=True,
    )

    train_ds = to_hf_dataset(train_rows, label_field=str(args.label_field))
    val_ds = to_hf_dataset(val_rows, label_field=str(args.label_field))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"], num_proc=args.num_workers or None)
    val_ds = val_ds.map(tok, batched=True, remove_columns=["text"], num_proc=args.num_workers or None)

    model_dtype = parse_torch_dtype(args.torch_dtype)
    model_load_kwargs = {"num_labels": 2}
    if model_dtype is not None:
        model_load_kwargs["torch_dtype"] = model_dtype
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_load_kwargs)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    target_modules = parse_target_modules(args.lora_target_modules)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }

    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size * 2),
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        bf16=bool(args.use_bf16),
        fp16=bool(args.use_fp16),
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=bool(args.ddp_find_unused_parameters),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics, flush=True)

    adapter_dir = os.path.join(args.output_dir, "adapter_model")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    if args.save_merged:
        merged_dir = os.path.join(args.output_dir, "best_model")
        print(f"Merging LoRA adapter into base model -> {merged_dir}", flush=True)
        fresh_load_kwargs = {"num_labels": 2}
        if model_dtype is not None:
            fresh_load_kwargs["torch_dtype"] = model_dtype
        fresh_base = AutoModelForSequenceClassification.from_pretrained(args.model_name, **fresh_load_kwargs)
        fresh_base.config.pad_token_id = tokenizer.pad_token_id
        peft_model = PeftModel.from_pretrained(fresh_base, adapter_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)

    cfg = {
        "manifest": args.manifest,
        "model_name": args.model_name,
        "label_field": args.label_field,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
        },
        "precision": {
            "torch_dtype": args.torch_dtype,
            "use_fp16": bool(args.use_fp16),
            "use_bf16": bool(args.use_bf16),
        },
        "ddp_find_unused_parameters": bool(args.ddp_find_unused_parameters),
        "metrics": metrics,
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
