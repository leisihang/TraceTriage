#!/usr/bin/env python3
"""
Purpose:
  Run per-dataset (single-dataset) Qwen fine-tune + eval on Track-1 datasets.

Protocol:
  - For each dataset directory, train on its own train/val split.
  - Use its val split to choose threshold.
  - Evaluate on its test split with eval_finetuned_track1.py protocol.
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
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ds_name(ds_dir: str) -> str:
    return os.path.basename(os.path.normpath(ds_dir))


def bootstrap_ci_mean(vals: List[float], iters: int, seed: int) -> tuple[float, float]:
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


def count_binary_rows(csv_path: str, label_field: str) -> int:
    if not os.path.isfile(csv_path):
        return 0
    n = 0
    with open(csv_path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        rd = csv.DictReader(f)
        for r in rd:
            y = str(r.get(label_field, "")).strip()
            if y in {"0", "1"}:
                n += 1
    return n


def _row_key(row: Dict[str, str]) -> str:
    pid = str(row.get("prefix_id", "")).strip()
    if pid:
        return pid
    return str(row.get("record_key", "")).strip()


def build_manifest_from_split_files(
    source_manifest: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    label_field: str,
    out_manifest: str,
) -> Dict[str, int]:
    if not os.path.isfile(source_manifest):
        raise FileNotFoundError(f"source_manifest not found: {source_manifest}")
    if not os.path.isfile(train_csv) or not os.path.isfile(val_csv) or not os.path.isfile(test_csv):
        raise FileNotFoundError("split csv missing for split_files mode")

    def read_keys(path: str) -> set[str]:
        keys: set[str] = set()
        with open(path, "r", newline="", encoding="utf-8", errors="ignore") as f:
            rd = csv.DictReader(f)
            for r in rd:
                y = str(r.get(label_field, "")).strip()
                if y not in {"0", "1"}:
                    continue
                k = _row_key(r)
                if k:
                    keys.add(k)
        return keys

    train_keys = read_keys(train_csv)
    val_keys = read_keys(val_csv)
    test_keys = read_keys(test_csv)

    overlap_tv = len(train_keys & val_keys)
    overlap_tt = len(train_keys & test_keys)
    overlap_vt = len(val_keys & test_keys)
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            "split_files overlap detected: "
            f"train∩val={overlap_tv}, train∩test={overlap_tt}, val∩test={overlap_vt}"
        )

    split_map: Dict[str, str] = {}
    for k in train_keys:
        split_map[k] = "train"
    for k in val_keys:
        split_map[k] = "val"
    for k in test_keys:
        split_map[k] = "test"

    kept = 0
    kept_train = 0
    kept_val = 0
    kept_test = 0
    missing = 0
    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)

    with open(source_manifest, "r", newline="", encoding="utf-8", errors="ignore") as fin:
        rd = csv.DictReader(fin)
        fieldnames = list(rd.fieldnames or [])
        if "split" not in fieldnames:
            fieldnames.append("split")
        with open(out_manifest, "w", newline="", encoding="utf-8") as fout:
            wr = csv.DictWriter(fout, fieldnames=fieldnames)
            wr.writeheader()
            for r in rd:
                y = str(r.get(label_field, "")).strip()
                if y not in {"0", "1"}:
                    continue
                k = _row_key(r)
                if not k or k not in split_map:
                    missing += 1
                    continue
                nr = dict(r)
                nr["split"] = split_map[k]
                wr.writerow(nr)
                kept += 1
                if nr["split"] == "train":
                    kept_train += 1
                elif nr["split"] == "val":
                    kept_val += 1
                elif nr["split"] == "test":
                    kept_test += 1

    return {
        "kept_rows": kept,
        "kept_train": kept_train,
        "kept_val": kept_val,
        "kept_test": kept_test,
        "missing_from_split_files": missing,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run single-dataset Qwen fine-tune/eval on Track-1 datasets.")
    ap.add_argument("--dataset-dir-glob", action="append", required=True, help="Glob(s) for dataset dirs.")
    ap.add_argument(
        "--target-dataset",
        action="append",
        default=[],
        help="Optional dataset basename(s) to run; defaults to all.",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-field", default="strict_success")
    ap.add_argument("--threshold-metric", default="avg_weighted_net_saved_seconds_with_inference")
    ap.add_argument("--max-success-false-stop-rate", type=float, default=0.10)
    ap.add_argument("--lambda-false-stop", type=float, default=1.0)
    ap.add_argument("--disable-threshold-constraint", action="store_true")
    ap.add_argument("--model-name", required=True)
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
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--train-script", default=os.path.join(SCRIPT_DIR, "train_small_llm_track1.py"))
    ap.add_argument("--eval-script", default=os.path.join(SCRIPT_DIR, "eval_finetuned_track1.py"))
    ap.add_argument("--eval-model-name", default="finetuned_single_qwen")
    ap.add_argument("--inference-cost-mode", choices=["measured", "zero", "fixed"], default="fixed")
    ap.add_argument("--inference-cost-fixed", type=float, default=0.01845288702419826)
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--train-split-source",
        choices=["manifest_column", "split_csv_files"],
        default="manifest_column",
        help="Use manifest split column (legacy) or rebuild split from splits/{train,val,test}.csv",
    )
    args = ap.parse_args()

    ds_dirs: List[str] = []
    for pat in args.dataset_dir_glob:
        ds_dirs.extend([p for p in glob.glob(pat) if os.path.isdir(p)])
    ds_dirs = sorted(set(ds_dirs))

    ok_dirs: List[str] = []
    for d in ds_dirs:
        tr = os.path.join(d, "track1_strict", "splits", "train.csv")
        va = os.path.join(d, "track1_strict", "splits", "val.csv")
        te = os.path.join(d, "track1_strict", "splits", "test.csv")
        if os.path.isfile(tr) and os.path.isfile(va) and os.path.isfile(te):
            ok_dirs.append(d)
    if not ok_dirs:
        raise SystemExit("No dataset dirs with track1_strict/splits/{train,val,test}.csv found.")

    target_filter = {str(x).strip() for x in (args.target_dataset or []) if str(x).strip()}
    if target_filter:
        target_dirs = [d for d in ok_dirs if ds_name(d) in target_filter]
    else:
        target_dirs = list(ok_dirs)
    if not target_dirs:
        raise SystemExit("No target datasets matched --target-dataset in available dirs.")

    os.makedirs(args.output_dir, exist_ok=True)
    fold_rows: List[Dict[str, str]] = []

    for ds in target_dirs:
        name = ds_name(ds)
        fold_dir = os.path.join(args.output_dir, name)
        os.makedirs(fold_dir, exist_ok=True)
        log_path = os.path.join(fold_dir, "runner.log")
        done_json = os.path.join(fold_dir, "metrics_summary.json")

        track1_dir = os.path.join(ds, "track1_strict")
        manifest = os.path.join(track1_dir, "track1_prefix_manifest.csv")
        train_csv = os.path.join(track1_dir, "splits", "train.csv")
        val_csv = os.path.join(track1_dir, "splits", "val.csv")
        test_csv = os.path.join(track1_dir, "splits", "test.csv")
        manifest_for_train = manifest

        n_train = count_binary_rows(train_csv, args.label_field)
        n_val = count_binary_rows(val_csv, args.label_field)
        n_test = count_binary_rows(test_csv, args.label_field)
        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            fold_rows.append(
                {
                    "dataset": name,
                    "status": "skipped_missing_rows",
                    "train_rows": str(n_train),
                    "val_rows": str(n_val),
                    "test_rows": str(n_test),
                    "error": "",
                }
            )
            continue

        if args.skip_existing and os.path.isfile(done_json):
            with open(done_json, "r", encoding="utf-8") as f:
                one = json.load(f)
            fold_rows.append(one)
            continue

        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"[{now_utc_iso()}] dataset={name}\n")
            lf.write(f"train_rows={n_train} val_rows={n_val} test_rows={n_test}\n")

        if args.train_split_source == "split_csv_files":
            manifest_for_train = os.path.join(fold_dir, "source_manifest_split_files.csv")
            mstats = build_manifest_from_split_files(
                source_manifest=manifest,
                train_csv=train_csv,
                val_csv=val_csv,
                test_csv=test_csv,
                label_field=args.label_field,
                out_manifest=manifest_for_train,
            )
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(
                    "[split_source=split_csv_files] "
                    f"manifest_rows={mstats['kept_rows']} "
                    f"train={mstats['kept_train']} val={mstats['kept_val']} test={mstats['kept_test']} "
                    f"missing={mstats['missing_from_split_files']}\n"
                )

        output_train = os.path.join(fold_dir, "train_output")
        model_dir = os.path.join(output_train, "best_model")
        eval_dir = os.path.join(fold_dir, "eval_main_protocol")

        train_cmd = [
            args.python_bin,
            args.train_script,
            "--manifest",
            manifest_for_train,
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
            str(args.prompt_style),
            "--semantic-mode",
            str(args.semantic_mode),
            "--semantic-shuffle-seed",
            str(args.semantic_shuffle_seed),
            "--semantic-source",
            str(args.semantic_source),
            "--semantic-max-chars",
            str(args.semantic_max_chars),
            "--feature-mode",
            str(args.feature_mode),
            "--feature-shuffle-seed",
            str(args.feature_shuffle_seed),
        ]

        eval_cmd = [
            args.python_bin,
            args.eval_script,
            "--track1-dir",
            track1_dir,
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
            "--prompt-style",
            str(args.prompt_style),
            "--semantic-mode",
            str(args.semantic_mode),
            "--semantic-shuffle-seed",
            str(args.semantic_shuffle_seed),
            "--semantic-source",
            str(args.semantic_source),
            "--semantic-max-chars",
            str(args.semantic_max_chars),
            "--feature-mode",
            str(args.feature_mode),
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
            "dataset": name,
            "status": status,
            "train_rows": str(n_train),
            "val_rows": str(n_val),
            "test_rows": str(n_test),
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
        "threshold_metric": args.threshold_metric,
        "max_success_false_stop_rate": args.max_success_false_stop_rate,
        "threshold_constraint_enabled": (not args.disable_threshold_constraint),
        "lambda_false_stop": args.lambda_false_stop,
        "model_name": args.model_name,
        "prompt_style": args.prompt_style,
        "semantic_mode": args.semantic_mode,
        "semantic_shuffle_seed": args.semantic_shuffle_seed,
        "semantic_source": args.semantic_source,
        "semantic_max_chars": args.semantic_max_chars,
        "feature_mode": args.feature_mode,
        "feature_shuffle_seed": args.feature_shuffle_seed,
        "train_config": {
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "seed": args.seed,
            "train_split_source": args.train_split_source,
        },
        "inference_cost_mode": args.inference_cost_mode,
        "inference_cost_fixed": args.inference_cost_fixed,
        "macro_mean": macro,
        "macro_mean_ci95_bootstrap": ci95,
        "worst_fold_by_net": worst_fold,
    }

    with open(os.path.join(args.output_dir, "single_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fields = [
        "dataset",
        "status",
        "train_rows",
        "val_rows",
        "test_rows",
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
    with open(os.path.join(args.output_dir, "single_folds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in fold_rows:
            w.writerow(r)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
