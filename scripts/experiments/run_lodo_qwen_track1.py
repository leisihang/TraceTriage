#!/usr/bin/env python3
"""
Purpose:
  Run Leave-One-Dataset-Out (LODO) evaluation for Track-1 with fine-tuned Qwen.

Protocol:
  - For each target dataset (one fold), train on the other datasets' source splits.
  - Use target val to choose stop threshold (same as eval_finetuned_track1.py protocol).
  - Report target test metrics and macro summary with bootstrap CI.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import run_track1_baselines as b  # noqa: E402


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ds_name(ds_dir: str) -> str:
    return os.path.basename(os.path.normpath(ds_dir))


def parse_source_splits(s: str) -> List[str]:
    out = []
    for x in (s or "").split(","):
        x = x.strip()
        if x in {"train", "val", "test"}:
            out.append(x)
    return out or ["train", "val"]


def parse_platform_design(ds_dir: str) -> Tuple[str, str]:
    bn = ds_name(ds_dir)
    parts = bn.split("_")
    if len(parts) >= 3 and parts[-1] == "main":
        return parts[0], parts[1]
    test_csv = os.path.join(ds_dir, "track1_strict", "splits", "test.csv")
    if os.path.isfile(test_csv):
        rows = b.load_rows(test_csv, require_binary_label=False, label_field="strict_success")
        if rows:
            return rows[0].get("platform", ""), rows[0].get("design", "")
    return "", ""


def load_split_rows(ds_dir: str, split: str, label_field: str) -> List[Dict[str, str]]:
    p = os.path.join(ds_dir, "track1_strict", "splits", f"{split}.csv")
    if not os.path.isfile(p):
        return []
    rows = b.load_rows(p, require_binary_label=True, label_field=label_field)
    for r in rows:
        r["original_split"] = str(r.get("split", "")).strip()
        r["source_file_split"] = split
        r["split"] = split
        r["source_dataset"] = ds_name(ds_dir)
    return rows


def _row_key(row: Dict[str, str]) -> str:
    pid = str(row.get("prefix_id", "")).strip()
    if pid:
        return pid
    return str(row.get("record_key", "")).strip()


def build_split_audit(
    source_rows: List[Dict[str, str]],
    target_dir: str,
    label_field: str,
) -> Dict[str, object]:
    target_test_rows = load_split_rows(target_dir, "test", label_field)
    train_keys = {_row_key(r) for r in source_rows if r.get("split") == "train" and _row_key(r)}
    val_keys = {_row_key(r) for r in source_rows if r.get("split") == "val" and _row_key(r)}
    test_keys = {_row_key(r) for r in target_test_rows if _row_key(r)}
    train_val_overlap = sorted(train_keys & val_keys)
    train_overlap = sorted(train_keys & test_keys)
    val_overlap = sorted(val_keys & test_keys)
    audit = {
        "target_dataset": ds_name(target_dir),
        "source_rows": len(source_rows),
        "source_train_rows": sum(1 for r in source_rows if r.get("split") == "train"),
        "source_val_rows": sum(1 for r in source_rows if r.get("split") == "val"),
        "target_test_rows": len(target_test_rows),
        "source_train_val_overlap": len(train_val_overlap),
        "source_train_target_test_overlap": len(train_overlap),
        "source_val_target_test_overlap": len(val_overlap),
        "source_train_val_overlap_examples": train_val_overlap[:20],
        "source_train_target_test_overlap_examples": train_overlap[:20],
        "source_val_target_test_overlap_examples": val_overlap[:20],
    }
    if train_val_overlap or train_overlap or val_overlap:
        raise RuntimeError(
            f"LODO leakage audit failed for {ds_name(target_dir)}: "
            f"source_train∩source_val={len(train_val_overlap)}, "
            f"source_train∩target_test={len(train_overlap)}, "
            f"source_val∩target_test={len(val_overlap)}"
        )
    return audit


def write_rows_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    seen = set(fields)
    for r in rows[1:]:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def bootstrap_ci_mean(vals: List[float], iters: int, seed: int) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(max(1, iters)):
        s = 0.0
        for _j in range(n):
            s += vals[rng.randrange(n)]
        samples.append(s / n)
    samples.sort()
    lo_i = int(0.025 * len(samples))
    hi_i = int(0.975 * len(samples))
    lo_i = min(max(lo_i, 0), len(samples) - 1)
    hi_i = min(max(hi_i, 0), len(samples) - 1)
    return float(samples[lo_i]), float(samples[hi_i])


def run_cmd(cmd: List[str], log_path: str) -> None:
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(cmd) + "\n")
        lf.flush()
        subprocess.run(cmd, check=True, stdout=lf, stderr=lf)


def read_eval_metrics(eval_dir: str, model_name: str) -> Dict[str, float]:
    p = os.path.join(eval_dir, "metrics_summary.json")
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    metrics = obj.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    m = metrics.get(model_name, {})
    if not isinstance(m, dict):
        return {}
    return m


def _metric_scalar_to_str(v: object) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(float(v))
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LODO Qwen fine-tune/eval on Track-1 datasets.")
    ap.add_argument("--dataset-dir-glob", action="append", required=True, help="Glob(s) for *_main dirs.")
    ap.add_argument(
        "--target-dataset",
        action="append",
        default=[],
        help="Optional dataset basename(s) to run as LODO target fold(s); defaults to all.",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-field", default="strict_success")
    ap.add_argument("--source-splits", default="train,val", help="Comma list: train,val,test")
    ap.add_argument("--threshold-metric", default="avg_weighted_net_saved_seconds_with_inference")
    ap.add_argument("--max-success-false-stop-rate", type=float, default=0.10)
    ap.add_argument("--lambda-false-stop", type=float, default=1.0)
    ap.add_argument("--disable-threshold-constraint", action="store_true")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
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
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--train-script", default=os.path.join(SCRIPT_DIR, "train_small_llm_track1.py"))
    ap.add_argument("--eval-script", default=os.path.join(SCRIPT_DIR, "eval_finetuned_track1.py"))
    ap.add_argument("--eval-model-name", default="finetuned_lodo_qwen")
    ap.add_argument("--inference-cost-mode", choices=["measured", "zero", "fixed"], default="fixed")
    ap.add_argument("--inference-cost-fixed", type=float, default=0.01845288702419826)
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    b.TARGET_LABEL_FIELD = str(args.label_field)
    source_splits = parse_source_splits(args.source_splits)

    ds_dirs: List[str] = []
    for pat in args.dataset_dir_glob:
        ds_dirs.extend([p for p in glob.glob(pat) if os.path.isdir(p)])
    ds_dirs = sorted(set(ds_dirs))

    ok_dirs: List[str] = []
    for d in ds_dirs:
        train_csv = os.path.join(d, "track1_strict", "splits", "train.csv")
        val_csv = os.path.join(d, "track1_strict", "splits", "val.csv")
        test_csv = os.path.join(d, "track1_strict", "splits", "test.csv")
        if os.path.isfile(train_csv) and os.path.isfile(val_csv) and os.path.isfile(test_csv):
            ok_dirs.append(d)
    if len(ok_dirs) < 2:
        raise SystemExit("Need at least 2 datasets with track1_strict train/val/test splits for LODO sources.")

    target_filter = {str(x).strip() for x in (args.target_dataset or []) if str(x).strip()}
    if target_filter:
        target_dirs = [d for d in ok_dirs if ds_name(d) in target_filter]
    else:
        target_dirs = list(ok_dirs)
    if not target_dirs:
        raise SystemExit("No target datasets matched --target-dataset in available dirs.")
    if len(target_dirs) == 1 and len(ok_dirs) < 2:
        raise SystemExit("Need at least 2 datasets in total to form source/target for LODO.")

    os.makedirs(args.output_dir, exist_ok=True)
    fold_rows: List[Dict[str, str]] = []

    for target_dir in target_dirs:
        target_name = ds_name(target_dir)
        platform, design = parse_platform_design(target_dir)
        fold_dir = os.path.join(args.output_dir, target_name)
        os.makedirs(fold_dir, exist_ok=True)
        log_path = os.path.join(fold_dir, "runner.log")
        done_json = os.path.join(fold_dir, "metrics_summary.json")

        source_rows: List[Dict[str, str]] = []
        for src_dir in ok_dirs:
            if src_dir == target_dir:
                continue
            for sp in source_splits:
                source_rows.extend(load_split_rows(src_dir, sp, args.label_field))

        source_train = sum(1 for r in source_rows if (r.get("split") or "") == "train")
        source_val = sum(1 for r in source_rows if (r.get("split") or "") == "val")

        if not source_rows or source_train == 0 or source_val == 0:
            fold_rows.append(
                {
                    "target_dataset": target_name,
                    "platform": platform,
                    "design": design,
                    "status": "skipped_missing_source_rows",
                    "source_rows": str(len(source_rows)),
                    "source_train_rows": str(source_train),
                    "source_val_rows": str(source_val),
                }
            )
            continue

        audit = build_split_audit(source_rows, target_dir, args.label_field)

        manifest_path = os.path.join(fold_dir, "source_manifest.csv")
        write_rows_csv(manifest_path, source_rows)
        with open(os.path.join(fold_dir, "split_audit.json"), "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)

        output_train = os.path.join(fold_dir, "train_output")
        model_dir = os.path.join(output_train, "best_model")
        eval_dir = os.path.join(fold_dir, "eval_main_protocol")
        track1_target = os.path.join(target_dir, "track1_strict")

        if args.skip_existing and os.path.isfile(done_json):
            with open(done_json, "r", encoding="utf-8") as f:
                one = json.load(f)
            fold_rows.append(one)
            continue

        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"[{now_utc_iso()}] target={target_name}\n")
            lf.write(f"source_rows={len(source_rows)} source_train={source_train} source_val={source_val}\n")
            lf.write(
                "split_audit "
                f"target_test={audit['target_test_rows']} "
                f"train_val_overlap={audit['source_train_val_overlap']} "
                f"train_test_overlap={audit['source_train_target_test_overlap']} "
                f"val_test_overlap={audit['source_val_target_test_overlap']}\n"
            )

        train_cmd = [
            args.python_bin,
            args.train_script,
            "--manifest",
            manifest_path,
            "--output-dir",
            output_train,
            "--label-field",
            args.label_field,
            "--model-name",
            args.model_name,
            "--max-train-samples",
            str(args.max_train_samples),
            "--max-val-samples",
            str(args.max_val_samples),
            "--max-length",
            str(args.max_length),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum",
            str(args.grad_accum),
            "--seed",
            str(args.seed),
            "--prompt-style",
            args.prompt_style,
            "--semantic-mode",
            args.semantic_mode,
            "--semantic-shuffle-seed",
            str(args.semantic_shuffle_seed),
            "--semantic-source",
            args.semantic_source,
            "--semantic-max-chars",
            str(args.semantic_max_chars),
            "--feature-mode",
            args.feature_mode,
            "--feature-shuffle-seed",
            str(args.feature_shuffle_seed),
        ]

        eval_cmd = [
            args.python_bin,
            args.eval_script,
            "--track1-dir",
            track1_target,
            "--model-dir",
            model_dir,
            "--output-dir",
            eval_dir,
            "--model-name",
            args.eval_model_name,
            "--label-field",
            args.label_field,
            "--threshold-metric",
            args.threshold_metric,
            "--max-success-false-stop-rate",
            str(args.max_success_false_stop_rate),
            "--lambda-false-stop",
            str(args.lambda_false_stop),
            "--max-length",
            str(args.max_length),
            "--prompt-style",
            args.prompt_style,
            "--semantic-mode",
            args.semantic_mode,
            "--semantic-shuffle-seed",
            str(args.semantic_shuffle_seed),
            "--semantic-source",
            args.semantic_source,
            "--semantic-max-chars",
            str(args.semantic_max_chars),
            "--feature-mode",
            args.feature_mode,
            "--feature-shuffle-seed",
            str(args.feature_shuffle_seed),
            "--inference-cost-mode",
            args.inference_cost_mode,
            "--inference-cost-fixed",
            str(args.inference_cost_fixed),
        ]
        if args.disable_threshold_constraint:
            eval_cmd.append("--disable-threshold-constraint")

        status = "ok"
        err = ""
        try:
            run_cmd(train_cmd, log_path)
            run_cmd(eval_cmd, log_path)
            metrics = read_eval_metrics(eval_dir, args.eval_model_name)
        except subprocess.CalledProcessError as e:
            status = "failed"
            metrics = {}
            err = f"command_failed_returncode_{e.returncode}"

        row = {
            "target_dataset": target_name,
            "platform": platform,
            "design": design,
            "status": status,
            "source_rows": str(len(source_rows)),
            "source_train_rows": str(source_train),
            "source_val_rows": str(source_val),
            "source_train_val_overlap": str(audit["source_train_val_overlap"]),
            "source_train_target_test_overlap": str(audit["source_train_target_test_overlap"]),
            "source_val_target_test_overlap": str(audit["source_val_target_test_overlap"]),
            "threshold": str(float(metrics.get("stop_threshold", 0.0))) if metrics else "",
            "accuracy": str(float(metrics.get("accuracy", 0.0))) if metrics else "",
            "fail_recall_stop": str(float(metrics.get("fail_recall_stop", 0.0))) if metrics else "",
            "success_false_stop_rate": str(float(metrics.get("success_false_stop_rate", 0.0))) if metrics else "",
            "stop_rate": str(float(metrics.get("stop_rate", 0.0))) if metrics else "",
            "net_with_infer": str(float(metrics.get("avg_weighted_net_saved_seconds_with_inference", 0.0))) if metrics else "",
            "oracle": str(float(metrics.get("oracle_avg_weighted_net_saved_seconds", 0.0))) if metrics else "",
            "headroom_ratio_vs_oracle": str(float(metrics.get("headroom_ratio_vs_oracle", 0.0))) if metrics else "",
            "error": err,
        }
        # Keep all eval metrics (including newly added stage metrics) in per-fold CSV/JSON.
        # Prefix with metric__ to avoid collision with base fields.
        if metrics:
            for mk, mv in metrics.items():
                row[f"metric__{mk}"] = _metric_scalar_to_str(mv)
        fold_rows.append(row)

        with open(done_json, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

    ok = [r for r in fold_rows if r.get("status") == "ok"]

    def vals(key: str) -> List[float]:
        out: List[float] = []
        for r in ok:
            try:
                out.append(float(r.get(key, "0") or 0.0))
            except Exception:
                out.append(0.0)
        return out

    metric_keys = [
        "accuracy",
        "fail_recall_stop",
        "success_false_stop_rate",
        "stop_rate",
        "net_with_infer",
        "oracle",
        "headroom_ratio_vs_oracle",
    ]
    macro: Dict[str, float] = {}
    ci95: Dict[str, Dict[str, float]] = {}
    for k in metric_keys:
        v = vals(k)
        macro[k] = float(sum(v) / len(v)) if v else 0.0
        lo, hi = bootstrap_ci_mean(v, iters=int(args.bootstrap_iters), seed=int(args.seed) + int.from_bytes(__import__("hashlib").sha256(str(k).encode()).digest()[:8], "little") % 10007)
        ci95[k] = {"lo": lo, "hi": hi}

    worst_fold = None
    if ok:
        worst_fold = min(ok, key=lambda r: float(r.get("net_with_infer", "0") or 0.0))

    summary = {
        "generated_at": now_utc_iso(),
        "dataset_count": len(target_dirs),
        "folds_total": len(fold_rows),
        "folds_ok": len(ok),
        "label_field": args.label_field,
        "source_splits": source_splits,
        "source_split_semantics": "split_csv_files",
        "threshold_metric": args.threshold_metric,
        "max_success_false_stop_rate": args.max_success_false_stop_rate,
        "threshold_constraint_enabled": (not args.disable_threshold_constraint),
        "lambda_false_stop": args.lambda_false_stop,
        "model_name": args.model_name,
        "train_config": {
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "seed": args.seed,
            "prompt_style": args.prompt_style,
            "semantic_mode": args.semantic_mode,
            "semantic_shuffle_seed": args.semantic_shuffle_seed,
            "semantic_source": args.semantic_source,
            "semantic_max_chars": args.semantic_max_chars,
            "feature_mode": args.feature_mode,
            "feature_shuffle_seed": args.feature_shuffle_seed,
        },
        "inference_cost_mode": args.inference_cost_mode,
        "inference_cost_fixed": args.inference_cost_fixed,
        "macro_mean": macro,
        "macro_mean_ci95_bootstrap": ci95,
        "worst_fold_by_net": worst_fold,
    }

    with open(os.path.join(args.output_dir, "lodo_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fields = [
        "target_dataset",
        "platform",
        "design",
        "status",
        "source_rows",
        "source_train_rows",
        "source_val_rows",
        "source_train_val_overlap",
        "source_train_target_test_overlap",
        "source_val_target_test_overlap",
        "threshold",
        "accuracy",
        "fail_recall_stop",
        "success_false_stop_rate",
        "stop_rate",
        "net_with_infer",
        "oracle",
        "headroom_ratio_vs_oracle",
        "error",
    ]
    extra_metric_fields = sorted(
        {
            k
            for r in fold_rows
            for k in r.keys()
            if k.startswith("metric__") and k not in fields
        }
    )
    fields.extend(extra_metric_fields)
    with open(os.path.join(args.output_dir, "lodo_folds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in fold_rows:
            w.writerow(r)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
