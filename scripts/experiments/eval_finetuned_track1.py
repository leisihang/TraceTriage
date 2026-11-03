#!/usr/bin/env python3
"""
Evaluate a fine-tuned sequence-classification model on Track-1 with the same
threshold/metric protocol as run_track1_baselines.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import run_track1_baselines as b


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
        prefix = (
            "You are an EDA triage expert for cost-aware early stopping.\n"
            "Goal: maximize net saved runtime with evidence-grounded STOP/CONTINUE decisions.\n"
            "R1 hard-fail logs imply STOP.\n"
            "R2 routing-collapse (grt with overflow>0.10 and congestion>0.80) is a risk recommendation.\n"
            "R3 timing-collapse (worst_slack<-10.0 and overflow>0.15 or congestion>0.15) is a risk recommendation.\n"
            "R2/R3 are not certainty; avoid mechanical trigger-only decisions and consider counter-evidence.\n"
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
        prefix = (
            "You are an EDA triage model.\n"
            "Task: Decide whether to STOP or CONTINUE for this EDA checkpoint.\n"
        )
    return (
        prefix
        + f"platform={platform}\n"
        + f"design={design}\n"
        + f"checkpoint={ckpt}\n"
        + f"elapsed_seconds={elapsed}\n"
        + f"run_params={run_params_text}\n"
        + f"prefix_metrics={metrics_raw}\n"
        + f"log_excerpt={excerpt}\n"
        + f"Output policy target: CONTINUE for {label_field}=1, STOP for {label_field}=0."
    )


def predict_success_prob(
    rows: List[Dict[str, str]],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    label_field: str,
    device: torch.device,
    prompt_style: str,
    semantic_mode: str,
    semantic_shuffle_seed: int,
    split_tag: str,
    semantic_source: str,
    semantic_max_chars: int,
    feature_mode: str,
    feature_shuffle_seed: int,
) -> Tuple[List[float], float]:
    if not rows:
        return [], 0.0

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
    texts = [
        format_input(
            r,
            label_field=label_field,
            prompt_style=prompt_style,
            log_excerpt_override=ex[i],
            run_params_override=run_params[i],
            metrics_override=metrics[i],
        )
        for i, r in enumerate(rows)
    ]
    probs: List[float] = []

    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            bt = texts[i : i + batch_size]
            enc = tokenizer(
                bt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            p = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy().tolist()
            probs.extend(float(x) for x in p)
    t1 = time.time()

    avg_infer = (t1 - t0) / max(1, len(rows))
    return probs, float(avg_infer)


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
        salt = f"{semantic_shuffle_seed}:{split_tag}:{n}".encode("utf-8")
        hv = hashlib.sha1(salt).hexdigest()
        seed = int(hv[:8], 16)
        rng = random.Random(seed)
        rng.shuffle(idx)
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


def write_metrics_table(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _parse_lambda_list(spec: str) -> List[float]:
    vals: List[float] = []
    for tok in (spec or "").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            vals.append(max(0.0, float(t)))
        except Exception:
            continue
    # Stable de-dup while preserving order.
    out: List[float] = []
    seen = set()
    for v in vals:
        k = f"{v:.10g}"
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate fine-tuned model on Track-1 main protocol.")
    ap.add_argument("--track1-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default="finetuned_qwen2.5_0.5b")
    ap.add_argument("--label-field", default="actionable_success")
    ap.add_argument("--threshold-metric", default="avg_weighted_net_saved_seconds_with_inference")
    ap.add_argument("--max-success-false-stop-rate", type=float, default=0.10)
    ap.add_argument("--lambda-false-stop", type=float, default=1.0)
    ap.add_argument(
        "--lambda-false-stop-list",
        default="",
        help="Optional comma-separated lambda list, e.g. '1,2,5,10'. "
        "When provided, performs one inference pass and evaluates all lambdas.",
    )
    ap.add_argument("--disable-threshold-constraint", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=1536)
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
    ap.add_argument("--inference-cost-mode", choices=["measured", "zero", "fixed"], default="measured")
    ap.add_argument("--inference-cost-fixed", type=float, default=0.0)
    ap.add_argument("--export-threshold-frontier", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_csv = os.path.join(args.track1_dir, "splits", "train.csv")
    val_csv = os.path.join(args.track1_dir, "splits", "val.csv")
    test_csv = os.path.join(args.track1_dir, "splits", "test.csv")
    if not (os.path.isfile(train_csv) and os.path.isfile(val_csv) and os.path.isfile(test_csv)):
        raise SystemExit("Missing split CSV files under track1/splits.")

    b.TARGET_LABEL_FIELD = str(args.label_field or "actionable_success")

    train_rows = b.load_rows(train_csv, require_binary_label=True, label_field=b.TARGET_LABEL_FIELD)
    val_rows = b.load_rows(val_csv, require_binary_label=True, label_field=b.TARGET_LABEL_FIELD)
    test_rows = b.load_rows(test_csv, require_binary_label=True, label_field=b.TARGET_LABEL_FIELD)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    _, train_infer = predict_success_prob(
        train_rows[: min(256, len(train_rows))],
        tokenizer,
        model,
        batch_size=max(1, args.batch_size),
        max_length=args.max_length,
        label_field=b.TARGET_LABEL_FIELD,
        device=device,
        prompt_style=args.prompt_style,
        semantic_mode=args.semantic_mode,
        semantic_shuffle_seed=int(args.semantic_shuffle_seed),
        split_tag="train_probe",
        semantic_source=args.semantic_source,
        semantic_max_chars=int(args.semantic_max_chars),
        feature_mode=args.feature_mode,
        feature_shuffle_seed=int(args.feature_shuffle_seed),
    )
    p_val, val_infer = predict_success_prob(
        val_rows,
        tokenizer,
        model,
        batch_size=max(1, args.batch_size),
        max_length=args.max_length,
        label_field=b.TARGET_LABEL_FIELD,
        device=device,
        prompt_style=args.prompt_style,
        semantic_mode=args.semantic_mode,
        semantic_shuffle_seed=int(args.semantic_shuffle_seed),
        split_tag="val",
        semantic_source=args.semantic_source,
        semantic_max_chars=int(args.semantic_max_chars),
        feature_mode=args.feature_mode,
        feature_shuffle_seed=int(args.feature_shuffle_seed),
    )
    p_test, test_infer = predict_success_prob(
        test_rows,
        tokenizer,
        model,
        batch_size=max(1, args.batch_size),
        max_length=args.max_length,
        label_field=b.TARGET_LABEL_FIELD,
        device=device,
        prompt_style=args.prompt_style,
        semantic_mode=args.semantic_mode,
        semantic_shuffle_seed=int(args.semantic_shuffle_seed),
        split_tag="test",
        semantic_source=args.semantic_source,
        semantic_max_chars=int(args.semantic_max_chars),
        feature_mode=args.feature_mode,
        feature_shuffle_seed=int(args.feature_shuffle_seed),
    )

    if args.inference_cost_mode == "zero":
        infer_val = 0.0
        infer_test = 0.0
    elif args.inference_cost_mode == "fixed":
        infer_val = float(args.inference_cost_fixed)
        infer_test = float(args.inference_cost_fixed)
    else:
        infer_val = float(val_infer)
        infer_test = float(test_infer)

    lambda_list = _parse_lambda_list(args.lambda_false_stop_list)
    if not lambda_list:
        lambda_list = [max(0.0, float(args.lambda_false_stop))]

    sweep_rows: List[Dict[str, float]] = []
    metrics_table_rows: List[Dict[str, str]] = []
    primary_thr = 0.0
    primary_m: Dict[str, float] = {}

    for i, lambda_fp in enumerate(lambda_list):
        thr = b.choose_best_threshold(
            val_rows,
            p_val,
            metric=args.threshold_metric,
            max_success_false_stop_rate=args.max_success_false_stop_rate,
            enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
            lambda_false_stop=lambda_fp,
            inference_cost_per_row=infer_val,
        )
        m = b.evaluate_predictions(
            test_rows,
            p_test,
            thr,
            lambda_false_stop=lambda_fp,
            inference_cost_per_row=infer_test,
        )
        m = b.enrich_with_oracle_headroom(test_rows, m, lambda_false_stop=lambda_fp)
        m["model_name"] = args.model_name
        m["inference_cost_mode"] = args.inference_cost_mode
        m["inference_cost_val_measured"] = float(val_infer)
        m["inference_cost_test_measured"] = float(test_infer)
        m["inference_cost_train_probe"] = float(train_infer)
        m["lambda_false_stop"] = float(lambda_fp)

        sweep_rows.append(
            {
                "lambda_false_stop": float(lambda_fp),
                "n": float(m.get("n", 0.0)),
                "oracle_s": float(m.get("oracle_avg_saved_seconds", 0.0)),
                "recall": float(m.get("fail_recall_stop", 0.0)),
                "fsr": float(m.get("success_false_stop_rate", 0.0)),
                "net_s": float(m.get("avg_weighted_net_saved_seconds_with_inference", 0.0)),
                "stop_threshold": float(thr),
                "headroom": float(m.get("headroom_ratio_vs_oracle", 0.0)),
                "inference_cost_test": float(infer_test),
            }
        )
        metrics_table_rows.append(
            {
                "model": args.model_name,
                "lambda_false_stop": str(lambda_fp),
                **{k: str(v) for k, v in m.items()},
            }
        )

        if i == 0:
            primary_thr = thr
            primary_m = m

    b.write_predictions_csv(
        os.path.join(args.output_dir, f"predictions_{args.model_name}_test.csv"),
        test_rows,
        p_test,
        primary_thr,
    )

    if args.export_threshold_frontier:
        frontier = b.scan_threshold_frontier(
            test_rows,
            p_test,
            lambda_false_stop=float(lambda_list[0]),
            inference_cost_per_row=infer_test,
            max_success_false_stop_rate=args.max_success_false_stop_rate,
            enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
        )
        b.write_frontier_csv(
            os.path.join(args.output_dir, f"frontier_{args.model_name}_test.csv"),
            frontier,
        )

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "track1_dir": args.track1_dir,
        "output_dir": args.output_dir,
        "label_field": b.TARGET_LABEL_FIELD,
        "threshold_metric": args.threshold_metric,
        "max_success_false_stop_rate": args.max_success_false_stop_rate,
        "threshold_constraint_enabled": (not args.disable_threshold_constraint),
        "reward_config": {
            "lambda_false_stop": float(lambda_list[0]),
            "lambda_false_stop_list": [float(x) for x in lambda_list],
            "inference_cost_mode": args.inference_cost_mode,
            "inference_cost_fixed": args.inference_cost_fixed,
            "prompt_style": args.prompt_style,
            "semantic_mode": args.semantic_mode,
            "semantic_shuffle_seed": int(args.semantic_shuffle_seed),
            "semantic_source": args.semantic_source,
            "semantic_max_chars": int(args.semantic_max_chars),
            "feature_mode": args.feature_mode,
            "feature_shuffle_seed": int(args.feature_shuffle_seed),
        },
        "data_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "metrics": {args.model_name: primary_m},
        "lambda_sweep": sweep_rows,
    }

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    write_metrics_table(
        os.path.join(args.output_dir, "metrics_table.csv"),
        metrics_table_rows,
    )
    write_metrics_table(
        os.path.join(args.output_dir, "lambda_sweep_metrics.csv"),
        [{k: str(v) for k, v in r.items()} for r in sweep_rows],
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
