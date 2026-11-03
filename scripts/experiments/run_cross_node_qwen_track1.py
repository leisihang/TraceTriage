#!/usr/bin/env python3
"""
Purpose:
  Run cross-node transfer for Track-1 with fine-tuned Qwen.

Protocol:
  - Train one model using all datasets from --source-platform.
  - Evaluate on each dataset from --target-platform with target-val thresholding.
  - Summarize per-target folds and macro statistics.
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

import run_track1_baselines as b  # noqa: E402


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ds_name(ds_dir: str) -> str:
    return os.path.basename(os.path.normpath(ds_dir))


def parse_source_splits(s: str) -> List[str]:
    out: List[str] = []
    for x in (s or "").split(","):
        x = x.strip()
        if x in {"train", "val", "test"}:
            out.append(x)
    return out or ["train", "val"]


def read_platform_from_test_csv(ds_dir: str) -> str:
    test_csv = os.path.join(ds_dir, "track1_strict", "splits", "test.csv")
    if not os.path.isfile(test_csv):
        return ""
    rows = b.load_rows(test_csv, require_binary_label=False, label_field="strict_success")
    if not rows:
        return ""
    return str(rows[0].get("platform", "")).strip()


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


def build_cross_node_split_audit(
    source_rows: List[Dict[str, str]],
    target_dirs: List[str],
    label_field: str,
) -> Dict[str, object]:
    target_test_rows: List[Dict[str, str]] = []
    for td in target_dirs:
        target_test_rows.extend(load_split_rows(td, "test", label_field))
    train_keys = {_row_key(r) for r in source_rows if r.get("split") == "train" and _row_key(r)}
    val_keys = {_row_key(r) for r in source_rows if r.get("split") == "val" and _row_key(r)}
    test_keys = {_row_key(r) for r in target_test_rows if _row_key(r)}
    train_val_overlap = sorted(train_keys & val_keys)
    train_overlap = sorted(train_keys & test_keys)
    val_overlap = sorted(val_keys & test_keys)
    audit = {
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
            "cross-node leakage audit failed: "
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Run cross-node Qwen transfer on Track-1.")
    ap.add_argument("--dataset-dir-glob", action="append", required=True, help="Glob(s) for dataset dirs.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--source-platform", required=True, help="Source platform, e.g. asap7 or sky130hd")
    ap.add_argument("--target-platform", required=True, help="Target platform, e.g. sky130hd or asap7")
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
    ap.add_argument("--eval-model-name", default="finetuned_cross_node_qwen")
    ap.add_argument("--inference-cost-mode", choices=["measured", "zero", "fixed"], default="fixed")
    ap.add_argument("--inference-cost-fixed", type=float, default=0.01845288702419826)
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    b.TARGET_LABEL_FIELD = str(args.label_field)
    source_splits = parse_source_splits(args.source_splits)
    src_platform = str(args.source_platform).strip()
    tgt_platform = str(args.target_platform).strip()

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

    src_dirs: List[str] = []
    tgt_dirs: List[str] = []
    for d in ok_dirs:
        p = read_platform_from_test_csv(d)
        if p == src_platform:
            src_dirs.append(d)
        if p == tgt_platform:
            tgt_dirs.append(d)

    if not src_dirs:
        raise SystemExit(f"No source datasets found for source_platform={src_platform}.")
    if not tgt_dirs:
        raise SystemExit(f"No target datasets found for target_platform={tgt_platform}.")

    os.makedirs(args.output_dir, exist_ok=True)
    train_root = os.path.join(args.output_dir, f"train_{src_platform}_to_{tgt_platform}")
    os.makedirs(train_root, exist_ok=True)
    train_log = os.path.join(train_root, "train_runner.log")
    source_manifest = os.path.join(train_root, "source_manifest.csv")
    model_dir = os.path.join(train_root, "train_output", "best_model")

    # Build one source manifest from all source-platform datasets.
    source_rows: List[Dict[str, str]] = []
    for sd in src_dirs:
        for sp in source_splits:
            source_rows.extend(load_split_rows(sd, sp, args.label_field))

    src_train = sum(1 for r in source_rows if (r.get("split") or "") == "train")
    src_val = sum(1 for r in source_rows if (r.get("split") or "") == "val")
    if not source_rows or src_train == 0 or src_val == 0:
        raise SystemExit(
            f"Invalid source rows for {src_platform}: total={len(source_rows)} train={src_train} val={src_val}"
        )
    split_audit = build_cross_node_split_audit(source_rows, tgt_dirs, args.label_field)
    write_rows_csv(source_manifest, source_rows)
    with open(os.path.join(train_root, "split_audit.json"), "w", encoding="utf-8") as f:
        json.dump(split_audit, f, indent=2)

    done_train_flag = os.path.join(train_root, "train_done.json")
    if not (args.skip_existing and os.path.isfile(done_train_flag) and os.path.isdir(model_dir)):
        with open(train_log, "a", encoding="utf-8") as lf:
            lf.write(f"[{now_utc_iso()}] source_platform={src_platform} target_platform={tgt_platform}\n")
            lf.write(
                f"source_datasets={len(src_dirs)} source_rows={len(source_rows)} "
                f"source_train={src_train} source_val={src_val}\n"
            )
            lf.write(
                "split_audit "
                f"target_test={split_audit['target_test_rows']} "
                f"train_val_overlap={split_audit['source_train_val_overlap']} "
                f"train_test_overlap={split_audit['source_train_target_test_overlap']} "
                f"val_test_overlap={split_audit['source_val_target_test_overlap']}\n"
            )

        train_cmd = [
            args.python_bin,
            args.train_script,
            "--manifest",
            source_manifest,
            "--output-dir",
            os.path.join(train_root, "train_output"),
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
        run_cmd(train_cmd, train_log)
        with open(done_train_flag, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": now_utc_iso(),
                    "source_platform": src_platform,
                    "target_platform": tgt_platform,
                    "source_dataset_count": len(src_dirs),
                    "source_rows": len(source_rows),
                    "source_train_rows": src_train,
                    "source_val_rows": src_val,
                    "source_split_semantics": "split_csv_files",
                    "prompt_style": args.prompt_style,
                    "semantic_mode": args.semantic_mode,
                    "semantic_source": args.semantic_source,
                    "semantic_max_chars": args.semantic_max_chars,
                    "feature_mode": args.feature_mode,
                    "source_train_val_overlap": split_audit["source_train_val_overlap"],
                    "source_train_target_test_overlap": split_audit["source_train_target_test_overlap"],
                    "source_val_target_test_overlap": split_audit["source_val_target_test_overlap"],
                },
                f,
                indent=2,
            )

    fold_rows: List[Dict[str, str]] = []
    for td in tgt_dirs:
        target_name = ds_name(td)
        fold_dir = os.path.join(args.output_dir, target_name)
        os.makedirs(fold_dir, exist_ok=True)
        log_path = os.path.join(fold_dir, "runner.log")
        done_json = os.path.join(fold_dir, "metrics_summary.json")
        track1_target = os.path.join(td, "track1_strict")

        if args.skip_existing and os.path.isfile(done_json):
            with open(done_json, "r", encoding="utf-8") as f:
                row = json.load(f)
            fold_rows.append(row)
            continue

        eval_cmd = [
            args.python_bin,
            args.eval_script,
            "--track1-dir",
            track1_target,
            "--model-dir",
            model_dir,
            "--output-dir",
            os.path.join(fold_dir, "eval_main_protocol"),
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
        metrics: Dict[str, float] = {}
        try:
            run_cmd(eval_cmd, log_path)
            metrics = read_eval_metrics(os.path.join(fold_dir, "eval_main_protocol"), args.eval_model_name)
        except subprocess.CalledProcessError as e:
            status = "failed"
            err = f"command_failed_returncode_{e.returncode}"

        row = {
            "target_dataset": target_name,
            "source_platform": src_platform,
            "target_platform": tgt_platform,
            "status": status,
            "source_train_val_overlap": str(split_audit["source_train_val_overlap"]),
            "source_train_target_test_overlap": str(split_audit["source_train_target_test_overlap"]),
            "source_val_target_test_overlap": str(split_audit["source_val_target_test_overlap"]),
            "threshold": str(float(metrics.get("stop_threshold", 0.0))) if metrics else "",
            "accuracy": str(float(metrics.get("accuracy", 0.0))) if metrics else "",
            "fail_recall_stop": str(float(metrics.get("fail_recall_stop", 0.0))) if metrics else "",
            "success_false_stop_rate": str(float(metrics.get("success_false_stop_rate", 0.0))) if metrics else "",
            "stop_rate": str(float(metrics.get("stop_rate", 0.0))) if metrics else "",
            "net_with_infer": str(float(metrics.get("avg_weighted_net_saved_seconds_with_inference", 0.0)))
            if metrics
            else "",
            "oracle": str(float(metrics.get("oracle_avg_weighted_net_saved_seconds", 0.0))) if metrics else "",
            "headroom_ratio_vs_oracle": str(float(metrics.get("headroom_ratio_vs_oracle", 0.0))) if metrics else "",
            "error": err,
        }
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
        "source_platform": src_platform,
        "target_platform": tgt_platform,
        "source_dataset_count": len(src_dirs),
        "target_dataset_count": len(tgt_dirs),
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
        "eval_model_name": args.eval_model_name,
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
        "macro_mean": macro,
        "macro_mean_ci95_bootstrap": ci95,
        "worst_fold_by_net": worst_fold,
    }

    with open(os.path.join(args.output_dir, "cross_node_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fields = [
        "target_dataset",
        "source_platform",
        "target_platform",
        "status",
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
    with open(os.path.join(args.output_dir, "cross_node_folds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in fold_rows:
            w.writerow(r)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
