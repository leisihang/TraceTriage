#!/usr/bin/env python3
"""
Minimal CPU fine-tuning for Track-1 STOP/CONTINUE on TraceTriage prefix data.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import run_track1_baselines as b


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_input(
    row: Dict[str, str],
    label_field: str,
    prompt_style: str = "strong",
    log_excerpt_override: str | None = None,
    run_params_override: str | None = None,
    metrics_override: str | None = None,
) -> str:
    if metrics_override is None:
        metrics_raw = (row.get("prefix_metrics") or "").strip()
    else:
        metrics_raw = str(metrics_override)
    if log_excerpt_override is None:
        excerpt = (row.get("prefix_log_excerpt") or "").strip()
    else:
        excerpt = str(log_excerpt_override)
    platform = row.get("platform", "")
    design = row.get("design", "")
    objective = row.get("objective_hint", "")
    ckpt = row.get("checkpoint", "")
    elapsed = row.get("prefix_elapsed_seconds", "")
    if run_params_override is None:
        run_params_text = b.format_run_params_text_for_model_input(row)
    else:
        run_params_text = str(run_params_override)
    style = str(prompt_style or "strong").strip().lower()
    if style == "expert":
        policy = (
            "You are an EDA triage expert for cost-aware early stopping.\n"
            "Goal: maximize net saved runtime with evidence-grounded STOP/CONTINUE decisions.\n"
            "Rule set (priority order):\n"
            "R1 Hard-fail rule: logs with ERROR/FATAL/segmentation/killed/timeout indicate STOP.\n"
            "R2 Routing-collapse risk signal (recommendation, not mandatory): checkpoint==grt and "
            "(overflow>0.10 AND congestion>0.80) indicates elevated failure risk.\n"
            "R3 Timing-collapse risk signal (recommendation, not mandatory): worst_slack<-10.0 and "
            "(overflow>0.15 OR congestion>0.15) indicates elevated failure risk.\n"
            "R2/R3 are risk recommendations, not certainty. Do not decide mechanically from a single trigger.\n"
            "When strong counter-evidence exists, CONTINUE is allowed.\n"
        )
    elif style == "strong":
        metrics_small = b.compact_metrics_for_llm(metrics_raw)
        if run_params_override is None:
            run_params_json = json.dumps(b.compact_run_params_with_missing_for_llm(row), ensure_ascii=False)
        else:
            run_params_json = str(run_params_override)
        sig = b.extract_log_signals(excerpt)
        return (
            "You are an EDA triage specialist for high-cost workflows.\n"
            "Goal: minimize wasted runtime while avoiding false stops on likely-successful runs.\n"
            "Decision policy: choose STOP only when risk of failure is clearly high.\n"
            "Output STRICT JSON only with keys:\n"
            "{\"decision\":\"STOP|CONTINUE\",\"confidence\":0.0-1.0,\"success_probability\":0.0-1.0,"
            "\"risk_tags\":[\"...\"],\"evidence\":[\"...\"]}\n"
            "Use concise risk_tags and include 1-3 evidence entries quoted from log text.\n\n"
            f"Platform: {platform}\n"
            f"Design: {design}\n"
            f"Objective: {objective}\n"
            f"Checkpoint: {ckpt}\n"
            f"Prefix elapsed seconds: {elapsed}\n"
            f"Metrics Compact JSON: {json.dumps(metrics_small, ensure_ascii=False)}\n"
            f"Run Params JSON: {run_params_json}\n"
            f"Log Signals JSON: {json.dumps(sig, ensure_ascii=False)}\n"
            f"Log Excerpt:\n{excerpt}\n"
        )
    else:
        policy = (
            "Task: Decide whether to STOP or CONTINUE for this EDA checkpoint.\n"
            "Goal: maximize net runtime savings while controlling false stops.\n"
        )

    return (
        policy
        + f"platform={platform}\n"
        + f"design={design}\n"
        + f"checkpoint={ckpt}\n"
        + f"elapsed_seconds={elapsed}\n"
        + f"run_params={run_params_text}\n"
        + f"prefix_metrics={metrics_raw}\n"
        + f"log_excerpt={excerpt}\n"
        + f"Output policy target: CONTINUE for {label_field}=1, STOP for {label_field}=0."
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


def build_semantic_excerpt_list(
    rows: List[Dict[str, str]],
    semantic_mode: str,
    semantic_shuffle_seed: int,
    split_tag: str,
    semantic_source: str,
    semantic_max_chars: int,
) -> List[str]:
    mode = str(semantic_mode or "full").strip().lower()
    source = str(semantic_source or "excerpt_only").strip().lower()
    ex = [_resolve_excerpt(r, source=source, max_chars=int(semantic_max_chars)) for r in rows]
    if mode == "full":
        return ex
    if mode == "no_log":
        return ["" for _ in ex]
    if mode == "shuffled_log":
        n = len(ex)
        if n <= 1:
            return ex
        idx = list(range(n))
        # Deterministic shuffle for reproducibility across runs/splits.
        salt = f"{semantic_shuffle_seed}:{split_tag}:{n}".encode("utf-8")
        hv = hashlib.sha1(salt).hexdigest()
        seed = int(hv[:8], 16)
        rng = random.Random(seed)
        rng.shuffle(idx)
        # Avoid identity mapping when possible.
        if all(i == idx[i] for i in range(n)):
            idx = idx[1:] + idx[:1]
        return [ex[idx[i]] for i in range(n)]
    raise ValueError(f"Unsupported semantic_mode: {semantic_mode}")


def _resolve_excerpt(row: Dict[str, str], source: str, max_chars: int) -> str:
    ex = str((row.get("prefix_log_excerpt") or "").strip())
    if ex:
        return ex[:max_chars]
    if source != "excerpt_or_path":
        return ""
    path = str((row.get("prefix_log_path") or row.get("prefix_log") or "").strip())
    if not path or (not os.path.isfile(path)):
        return ""
    return _extract_log_excerpt_from_path(path=path, max_chars=max_chars)


def _extract_log_excerpt_from_path(path: str, max_chars: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return ""
    pat = re.compile(
        r"(WARN|ERROR|FATAL|violation|overflow|congestion|slack|WNS|TNS|DRC|failed|timeout|killed|segmentation)",
        re.IGNORECASE,
    )
    focus = [ln.strip() for ln in lines if pat.search(ln)]
    tail = [ln.strip() for ln in lines[-40:]]
    merged: List[str] = []
    seen = set()
    for ln in (focus[-120:] + tail):
        if not ln or ln in seen:
            continue
        seen.add(ln)
        merged.append(ln)
    if merged:
        txt = "\n".join(merged)
    else:
        txt = ("".join(lines[:20]) + "\n...\n" + "".join(lines[-40:])).strip()
    return txt[:max_chars]


def build_feature_overrides(
    rows: List[Dict[str, str]],
    feature_mode: str,
    feature_shuffle_seed: int,
    split_tag: str,
) -> tuple[List[str], List[str]]:
    mode = str(feature_mode or "full").strip().lower()
    run_params = [b.format_run_params_text_for_model_input(r) for r in rows]
    metrics = [str((r.get("prefix_metrics") or "").strip()) for r in rows]
    n = len(rows)

    def _permute(vals: List[str], salt_key: str) -> List[str]:
        if n <= 1:
            return vals[:]
        idx = list(range(n))
        salt = f"{feature_shuffle_seed}:{split_tag}:{salt_key}:{n}".encode("utf-8")
        hv = hashlib.sha1(salt).hexdigest()
        seed = int(hv[:8], 16)
        rng = random.Random(seed)
        rng.shuffle(idx)
        if all(i == idx[i] for i in range(n)):
            idx = idx[1:] + idx[:1]
        return [vals[idx[i]] for i in range(n)]

    if mode == "full":
        return run_params, metrics
    if mode == "no_run_params":
        return ["" for _ in range(n)], metrics
    if mode == "no_metrics":
        return run_params, ["" for _ in range(n)]
    if mode == "shuffled_run_params":
        return _permute(run_params, "run_params"), metrics
    if mode == "shuffled_metrics":
        return run_params, _permute(metrics, "metrics")
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def to_hf_dataset(
    rows: List[Dict[str, str]],
    label_field: str,
    prompt_style: str,
    semantic_mode: str,
    semantic_shuffle_seed: int,
    split_tag: str,
    semantic_source: str,
    semantic_max_chars: int,
    feature_mode: str,
    feature_shuffle_seed: int,
) -> Dataset:
    ex = build_semantic_excerpt_list(
        rows=rows,
        semantic_mode=semantic_mode,
        semantic_shuffle_seed=semantic_shuffle_seed,
        split_tag=split_tag,
        semantic_source=semantic_source,
        semantic_max_chars=semantic_max_chars,
    )
    run_params, metrics = build_feature_overrides(
        rows=rows,
        feature_mode=feature_mode,
        feature_shuffle_seed=feature_shuffle_seed,
        split_tag=split_tag,
    )
    records = []
    for i, r in enumerate(rows):
        label = int(r[label_field])
        records.append(
            {
                "text": format_input(
                    r,
                    label_field=label_field,
                    prompt_style=prompt_style,
                    log_excerpt_override=ex[i],
                    run_params_override=run_params[i],
                    metrics_override=metrics[i],
                ),
                "label": label,
            }
        )
    return Dataset.from_list(records)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-field", default="actionable_success")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-train-samples", type=int, default=2000)
    ap.add_argument("--max-val-samples", type=int, default=300)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--save-total-limit", type=int, default=10)
    ap.add_argument("--metric-for-best-model", default="macro_f1")
    ap.add_argument("--use-class-weights", action="store_true")
    ap.add_argument("--prompt-style", choices=["basic", "strong", "expert"], default="strong")
    ap.add_argument("--semantic-mode", choices=["full", "no_log", "shuffled_log"], default="full")
    ap.add_argument("--semantic-shuffle-seed", type=int, default=42)
    ap.add_argument("--semantic-source", choices=["excerpt_only", "excerpt_or_path"], default="excerpt_only")
    ap.add_argument("--semantic-max-chars", type=int, default=1800)
    ap.add_argument(
        "--feature-mode",
        choices=["full", "no_run_params", "no_metrics", "shuffled_run_params", "shuffled_metrics"],
        default="full",
    )
    ap.add_argument("--feature-shuffle-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_rows(args.manifest, label_field=str(args.label_field or "actionable_success"))
    train_rows = build_split(rows, "train", args.max_train_samples, args.seed)
    val_rows = build_split(rows, "val", args.max_val_samples, args.seed)

    print(
        f"Loaded rows: total={len(rows)} train={len(train_rows)} val={len(val_rows)} "
        f"label_field={args.label_field}"
    )
    train_pos = sum(1 for r in train_rows if str(r.get(str(args.label_field), "")).strip() == "1")
    train_neg = sum(1 for r in train_rows if str(r.get(str(args.label_field), "")).strip() == "0")
    print(
        f"Train label counts: pos1={train_pos} neg0={train_neg} "
        f"neg_rate={(train_neg / max(1, (train_pos + train_neg))):.4f}"
    )

    train_ds = to_hf_dataset(
        train_rows,
        label_field=str(args.label_field),
        prompt_style=str(args.prompt_style),
        semantic_mode=str(args.semantic_mode),
        semantic_shuffle_seed=int(args.semantic_shuffle_seed),
        split_tag="train",
        semantic_source=str(args.semantic_source),
        semantic_max_chars=int(args.semantic_max_chars),
        feature_mode=str(args.feature_mode),
        feature_shuffle_seed=int(args.feature_shuffle_seed),
    )
    val_ds = to_hf_dataset(
        val_rows,
        label_field=str(args.label_field),
        prompt_style=str(args.prompt_style),
        semantic_mode=str(args.semantic_mode),
        semantic_shuffle_seed=int(args.semantic_shuffle_seed),
        split_tag="val",
        semantic_source=str(args.semantic_source),
        semantic_max_chars=int(args.semantic_max_chars),
        feature_mode=str(args.feature_mode),
        feature_shuffle_seed=int(args.feature_shuffle_seed),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p1, r1, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=1, zero_division=0
        )
        p0, r0, f1_stop, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=0, zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": float(acc),
            "precision": float(p1),
            "recall": float(r1),
            "f1": float(f1),
            "precision_stop": float(p0),
            "recall_stop": float(r0),
            "f1_stop": float(f1_stop),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "macro_f1": float(f1_macro),
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
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        report_to=[],
        bf16=False,
        fp16=False,
    )

    class_weights = None
    if args.use_class_weights:
        total = train_pos + train_neg
        if train_pos > 0 and train_neg > 0:
            # label 0 is STOP(negative), label 1 is CONTINUE(positive)
            w0 = total / (2.0 * train_neg)
            w1 = total / (2.0 * train_pos)
            class_weights = torch.tensor([w0, w1], dtype=torch.float32)
            print(f"Using class weights: w0={w0:.6f}, w1={w1:.6f}")

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))


if __name__ == "__main__":
    main()
