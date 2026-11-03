#!/usr/bin/env python3
"""
Purpose:
  Run Leave-One-Dataset-Out (LODO) evaluation for Track-1 using supervised baseline
  (xgboost preferred) under the same threshold protocol.

Protocol:
  - Train on source datasets (default source splits: train+val).
  - Calibrate threshold on target dataset val split.
  - Evaluate on target dataset test split.
  - Report per-fold metrics and macro summary with bootstrap CI.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import run_track1_baselines as b  # noqa: E402


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_source_splits(s: str) -> List[str]:
    out = []
    for x in (s or "").split(","):
        x = x.strip()
        if x in {"train", "val", "test"}:
            out.append(x)
    return out or ["train", "val"]


def ds_name(ds_dir: str) -> str:
    return os.path.basename(os.path.normpath(ds_dir))


def parse_platform_design(ds_dir: str) -> Tuple[str, str]:
    bn = ds_name(ds_dir)
    parts = bn.split("_")
    if len(parts) >= 3 and parts[-1] == "main":
        return parts[0], parts[1]
    # fallback from test row
    test_csv = os.path.join(ds_dir, "track1_strict", "splits", "test.csv")
    rows = b.load_rows(test_csv, require_binary_label=False, label_field="strict_success") if os.path.isfile(test_csv) else []
    if rows:
        return rows[0].get("platform", ""), rows[0].get("design", "")
    return "", ""


def load_split_rows(ds_dir: str, split: str, label_field: str) -> List[Dict[str, str]]:
    p = os.path.join(ds_dir, "track1_strict", "splits", f"{split}.csv")
    if not os.path.isfile(p):
        return []
    return b.load_rows(p, require_binary_label=True, label_field=label_field)


def write_predictions(path: str, rows: List[Dict[str, str]], probs: List[float], threshold: float, label_field: str) -> None:
    fields = [
        "prefix_id",
        "platform",
        "design",
        "checkpoint",
        "run_id",
        "split",
        "eval_label_field",
        "eval_label_value",
        "success_prob",
        "pred_stop",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, p in zip(rows, probs):
            w.writerow(
                {
                    "prefix_id": r.get("prefix_id", ""),
                    "platform": r.get("platform", ""),
                    "design": r.get("design", ""),
                    "checkpoint": r.get("checkpoint", ""),
                    "run_id": r.get("run_id", ""),
                    "split": r.get("split", ""),
                    "eval_label_field": label_field,
                    "eval_label_value": r.get(label_field, ""),
                    "success_prob": f"{float(p):.8f}",
                    "pred_stop": "1" if float(p) < float(threshold) else "0",
                }
            )


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
    hi_i = min(max(hi_i, 0), len(samples) - 1)
    lo_i = min(max(lo_i, 0), len(samples) - 1)
    return float(samples[lo_i]), float(samples[hi_i])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LODO xgboost baseline on Track-1 datasets.")
    ap.add_argument("--dataset-dir-glob", action="append", required=True, help="Glob(s) for *_main dirs.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-field", default="strict_success")
    ap.add_argument("--source-splits", default="train,val", help="Comma list: train,val,test")
    ap.add_argument("--target-val-split", default="val")
    ap.add_argument("--target-test-split", default="test")
    ap.add_argument("--threshold-metric", default="avg_weighted_net_saved_seconds_with_inference")
    ap.add_argument("--max-success-false-stop-rate", type=float, default=0.10)
    ap.add_argument("--lambda-false-stop", type=float, default=1.0)
    ap.add_argument("--disable-threshold-constraint", action="store_true")
    ap.add_argument("--include-missing-indicators", action="store_true")
    ap.add_argument("--supervised-model", choices=["auto", "xgboost", "hgb", "logreg"], default="xgboost")
    ap.add_argument(
        "--feature-mode",
        choices=["param_plus_prefix", "prefix_only", "param_only"],
        default="param_plus_prefix",
    )
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    b.TARGET_LABEL_FIELD = str(args.label_field)

    ds_dirs: List[str] = []
    for pat in args.dataset_dir_glob:
        ds_dirs.extend([p for p in glob.glob(pat) if os.path.isdir(p)])
    ds_dirs = sorted(set(ds_dirs))

    # Keep only dirs with required split files.
    ok_dirs: List[str] = []
    for d in ds_dirs:
        tr = os.path.join(d, "track1_strict", "splits", f"{args.target_test_split}.csv")
        va = os.path.join(d, "track1_strict", "splits", f"{args.target_val_split}.csv")
        if os.path.isfile(tr) and os.path.isfile(va):
            ok_dirs.append(d)

    if len(ok_dirs) < 2:
        raise SystemExit("Need at least 2 datasets with track1_strict val/test splits.")

    os.makedirs(args.output_dir, exist_ok=True)

    source_splits = parse_source_splits(args.source_splits)
    lam = max(0.0, float(args.lambda_false_stop))

    fold_rows: List[Dict[str, str]] = []

    for target_dir in ok_dirs:
        target_name = ds_name(target_dir)
        target_platform, target_design = parse_platform_design(target_dir)
        fold_dir = os.path.join(args.output_dir, target_name)
        os.makedirs(fold_dir, exist_ok=True)

        source_rows: List[Dict[str, str]] = []
        for src_dir in ok_dirs:
            if src_dir == target_dir:
                continue
            for sp in source_splits:
                source_rows.extend(load_split_rows(src_dir, sp, args.label_field))

        target_val_rows = load_split_rows(target_dir, args.target_val_split, args.label_field)
        target_test_rows = load_split_rows(target_dir, args.target_test_split, args.label_field)

        if not source_rows or not target_val_rows or not target_test_rows:
            fold_rows.append(
                {
                    "target_dataset": target_name,
                    "platform": target_platform,
                    "design": target_design,
                    "status": "skipped_missing_rows",
                    "source_train_rows": str(len(source_rows)),
                    "target_val_rows": str(len(target_val_rows)),
                    "target_test_rows": str(len(target_test_rows)),
                }
            )
            continue

        p_train, p_val, p_test, selected_model = b.train_supervised_model(
            source_rows,
            target_val_rows,
            target_test_rows,
            include_missing_indicators=bool(args.include_missing_indicators),
            preferred_model=str(args.supervised_model),
            feature_mode=str(args.feature_mode),
            seed=int(args.seed),
        )

        thr = b.choose_best_threshold(
            target_val_rows,
            p_val,
            metric=str(args.threshold_metric),
            max_success_false_stop_rate=float(args.max_success_false_stop_rate),
            enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
            lambda_false_stop=lam,
            inference_cost_per_row=0.0,
        )

        m = b.evaluate_predictions(
            target_test_rows,
            p_test,
            float(thr),
            lambda_false_stop=lam,
            inference_cost_per_row=0.0,
        )
        m = b.enrich_with_oracle_headroom(target_test_rows, m, lambda_false_stop=lam)

        frontier = b.scan_threshold_frontier(
            target_test_rows,
            p_test,
            lambda_false_stop=lam,
            inference_cost_per_row=0.0,
            max_success_false_stop_rate=float(args.max_success_false_stop_rate),
            enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
        )
        b.write_frontier_csv(os.path.join(fold_dir, "frontier_xgb_target_test.csv"), frontier)
        write_predictions(
            os.path.join(fold_dir, "predictions_xgb_target_test.csv"),
            target_test_rows,
            p_test,
            float(thr),
            args.label_field,
        )

        fold_json = {
            "generated_at": now_utc_iso(),
            "target_dataset": target_name,
            "target_platform": target_platform,
            "target_design": target_design,
            "target_dir": target_dir,
            "source_dataset_count": len(ok_dirs) - 1,
            "source_splits": source_splits,
            "source_train_rows": len(source_rows),
            "target_val_rows": len(target_val_rows),
            "target_test_rows": len(target_test_rows),
            "label_field": args.label_field,
            "selected_supervised_model": selected_model,
            "feature_mode": str(args.feature_mode),
            "threshold_metric": args.threshold_metric,
            "max_success_false_stop_rate": args.max_success_false_stop_rate,
            "threshold_constraint_enabled": (not args.disable_threshold_constraint),
            "lambda_false_stop": lam,
            "threshold_from_target_val": float(thr),
            "metrics_test": m,
        }
        with open(os.path.join(fold_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(fold_json, f, indent=2)

        fold_rows.append(
            {
                "target_dataset": target_name,
                "platform": target_platform,
                "design": target_design,
                "status": "ok",
                "selected_model": selected_model,
                "source_train_rows": str(len(source_rows)),
                "target_val_rows": str(len(target_val_rows)),
                "target_test_rows": str(len(target_test_rows)),
                "threshold": str(float(thr)),
                "accuracy": str(m.get("accuracy", 0.0)),
                "fail_recall_stop": str(m.get("fail_recall_stop", 0.0)),
                "success_false_stop_rate": str(m.get("success_false_stop_rate", 0.0)),
                "stop_rate": str(m.get("stop_rate", 0.0)),
                "net_with_infer": str(m.get("avg_weighted_net_saved_seconds_with_inference", 0.0)),
                "oracle": str(m.get("oracle_avg_weighted_net_saved_seconds", 0.0)),
                "headroom_ratio_vs_oracle": str(m.get("headroom_ratio_vs_oracle", 0.0)),
            }
        )

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
        "dataset_count": len(ok_dirs),
        "folds_total": len(fold_rows),
        "folds_ok": len(ok),
        "label_field": args.label_field,
        "source_splits": source_splits,
        "target_val_split": args.target_val_split,
        "target_test_split": args.target_test_split,
        "threshold_metric": args.threshold_metric,
        "max_success_false_stop_rate": args.max_success_false_stop_rate,
        "threshold_constraint_enabled": (not args.disable_threshold_constraint),
        "lambda_false_stop": lam,
        "supervised_model_preference": args.supervised_model,
        "feature_mode": args.feature_mode,
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
        "selected_model",
        "source_train_rows",
        "target_val_rows",
        "target_test_rows",
        "threshold",
        "accuracy",
        "fail_recall_stop",
        "success_false_stop_rate",
        "stop_rate",
        "net_with_infer",
        "oracle",
        "headroom_ratio_vs_oracle",
    ]
    with open(os.path.join(args.output_dir, "lodo_folds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in fold_rows:
            w.writerow(r)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
