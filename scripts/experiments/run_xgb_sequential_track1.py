#!/usr/bin/env python3
"""
Run XGBoost Track-1 experiments under run-level sequential protocol.

Protocol:
- Decision points: init -> place -> cts -> grt -> drt
- For each run, evaluate checkpoints in order; first STOP ends the run.
- Utility counted once per run:
    fail & STOP at ck: +remaining_seconds(ck)
    success & STOP at ck: -lambda_false_stop * remaining_seconds(ck)
    no STOP: 0
  then minus inference_cost_per_row * (#evaluated prefixes for that run)

Modes:
- single: train/val/test within each dataset
- lodo: train on other datasets (source_splits), val/test on target dataset
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import run_track1_baselines as b  # noqa: E402


CK_ORDER = {"init": 0, "place": 1, "cts": 2, "grt": 3, "drt": 4}
CK_LIST = ["init", "place", "cts", "grt", "drt"]


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


def load_split_rows(ds_dir: str, split: str, label_field: str) -> List[Dict[str, str]]:
    p = os.path.join(ds_dir, "track1_strict", "splits", f"{split}.csv")
    if not os.path.isfile(p):
        return []
    return b.load_rows(p, require_binary_label=True, label_field=label_field)


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


def _remaining(row: Dict[str, str]) -> float:
    return max(0.0, float(b.maybe_float(row.get("remaining_seconds", "")) or 0.0))


def _run_key(row: Dict[str, str]) -> str:
    k = (row.get("record_key", "") or "").strip()
    if k:
        return k
    return (row.get("run_id", "") or "").strip()


def build_run_groups(rows: List[Dict[str, str]], probs: List[float]) -> List[List[Tuple[Dict[str, str], float]]]:
    groups: Dict[str, List[Tuple[Dict[str, str], float]]] = defaultdict(list)
    for r, p in zip(rows, probs):
        groups[_run_key(r)].append((r, float(p)))

    out: List[List[Tuple[Dict[str, str], float]]] = []
    for _, items in groups.items():
        items_sorted = sorted(items, key=lambda rp: CK_ORDER.get((rp[0].get("checkpoint", "") or "").strip(), 99))
        out.append(items_sorted)
    return out


def run_level_eval(
    rows: List[Dict[str, str]],
    probs: List[float],
    thresholds: Dict[str, float],
    lambda_false_stop: float,
    inference_cost_per_row: float,
) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    run_groups = build_run_groups(rows, probs)

    n_runs = len(run_groups)
    if n_runs == 0:
        return {
            "n_runs": 0,
            "accuracy": 0.0,
            "fail_recall_stop": 0.0,
            "success_false_stop_rate": 0.0,
            "stop_rate": 0.0,
            "avg_inference_steps": 0.0,
            "avg_weighted_net_saved_seconds_with_inference": 0.0,
            "oracle_avg_weighted_net_saved_seconds": 0.0,
            "headroom_ratio_vs_oracle": 0.0,
            "stop_precision_on_stopped": 0.0,
        }, []

    tp = fp = tn = fn = 0
    net_sum = 0.0
    oracle_sum = 0.0
    total_steps = 0
    stopped_runs = 0

    pred_rows: List[Dict[str, str]] = []

    for items in run_groups:
        # run-level label from any prefix row in run.
        y_success = b.target_success(items[0][0])  # 1 success, 0 fail
        # oracle: stop at checkpoint with max remaining on fail, else 0
        if y_success == 0:
            oracle_sum += max(_remaining(r) for r, _ in items)

        stopped = False
        steps = 0
        contrib = 0.0

        for r, p in items:
            ck = (r.get("checkpoint", "") or "").strip()
            thr = float(thresholds.get(ck, 0.5))
            pred_stop = 1 if p < thr else 0
            steps += 1

            pred_rows.append(
                {
                    "prefix_id": r.get("prefix_id", ""),
                    "record_key": _run_key(r),
                    "checkpoint": ck,
                    "strict_success": r.get("strict_success", ""),
                    "remaining_seconds": r.get("remaining_seconds", ""),
                    "success_prob": f"{p:.8f}",
                    "threshold": f"{thr:.6f}",
                    "pred_stop": str(pred_stop),
                }
            )

            if pred_stop == 1:
                rem = _remaining(r)
                if y_success == 0:
                    contrib += rem
                else:
                    contrib -= lambda_false_stop * rem
                stopped = True
                break

        contrib -= float(inference_cost_per_row) * float(steps)
        net_sum += contrib
        total_steps += steps

        if stopped:
            stopped_runs += 1

        if y_success == 0 and stopped:
            tp += 1
        elif y_success == 0 and not stopped:
            fn += 1
        elif y_success == 1 and stopped:
            fp += 1
        else:
            tn += 1

    fail = tp + fn
    succ = fp + tn
    stop_total = tp + fp

    acc = (tp + tn) / n_runs
    fail_recall = (tp / fail) if fail else 0.0
    fsr = (fp / succ) if succ else 0.0
    stop_rate = stop_total / n_runs
    avg_steps = total_steps / n_runs
    net_avg = net_sum / n_runs
    oracle_avg = oracle_sum / n_runs
    headroom = (net_avg / oracle_avg) if oracle_avg > 0 else 0.0
    stop_precision = (tp / stop_total) if stop_total > 0 else 0.0

    m = {
        "n_runs": float(n_runs),
        "accuracy": float(acc),
        "fail_recall_stop": float(fail_recall),
        "success_false_stop_rate": float(fsr),
        "stop_rate": float(stop_rate),
        "avg_inference_steps": float(avg_steps),
        "avg_weighted_net_saved_seconds_with_inference": float(net_avg),
        "oracle_avg_weighted_net_saved_seconds": float(oracle_avg),
        "headroom_ratio_vs_oracle": float(headroom),
        "stop_precision_on_stopped": float(stop_precision),
    }
    return m, pred_rows


def choose_thresholds_coordinate_descent(
    rows_val: List[Dict[str, str]],
    probs_val: List[float],
    max_success_false_stop_rate: float,
    lambda_false_stop: float,
    inference_cost_per_row: float,
    enforce_constraint: bool,
    rounds: int = 3,
) -> Dict[str, float]:
    grid = [i / 100.0 for i in range(0, 96)]
    th = {ck: 0.5 for ck in CK_LIST}

    def better(m_new: Dict[str, float], m_old: Optional[Dict[str, float]]) -> bool:
        if m_old is None:
            return True
        return (
            float(m_new["avg_weighted_net_saved_seconds_with_inference"])
            > float(m_old["avg_weighted_net_saved_seconds_with_inference"]) + 1e-12
        ) or (
            abs(
                float(m_new["avg_weighted_net_saved_seconds_with_inference"])
                - float(m_old["avg_weighted_net_saved_seconds_with_inference"])
            )
            <= 1e-12
            and float(m_new["fail_recall_stop"]) > float(m_old["fail_recall_stop"])
        )

    for _ in range(max(1, rounds)):
        for ck in CK_LIST:
            best_m: Optional[Dict[str, float]] = None
            best_t = th[ck]
            fallback_m: Optional[Dict[str, float]] = None
            fallback_t = th[ck]
            for t in grid:
                cand = dict(th)
                cand[ck] = float(t)
                m, _ = run_level_eval(
                    rows_val,
                    probs_val,
                    thresholds=cand,
                    lambda_false_stop=lambda_false_stop,
                    inference_cost_per_row=inference_cost_per_row,
                )
                fsr = float(m["success_false_stop_rate"])
                # fallback: minimize FSR, tie by higher net
                if fallback_m is None:
                    fallback_m = m
                    fallback_t = t
                else:
                    if (fsr < float(fallback_m["success_false_stop_rate"]) - 1e-12) or (
                        abs(fsr - float(fallback_m["success_false_stop_rate"])) <= 1e-12
                        and float(m["avg_weighted_net_saved_seconds_with_inference"])
                        > float(fallback_m["avg_weighted_net_saved_seconds_with_inference"]) + 1e-12
                    ):
                        fallback_m = m
                        fallback_t = t

                if enforce_constraint and fsr > max_success_false_stop_rate:
                    continue
                if better(m, best_m):
                    best_m = m
                    best_t = t

            th[ck] = float(best_t if best_m is not None else fallback_t)

    return th


def fixed_thresholds(value: float) -> Dict[str, float]:
    v = max(0.0, min(0.95, float(value)))
    return {ck: v for ck in CK_LIST}


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


def run_single_dataset(
    ds_dir: str,
    output_dir: str,
    label_field: str,
    lambda_false_stop: float,
    max_success_false_stop_rate: float,
    disable_threshold_constraint: bool,
    include_missing_indicators: bool,
    supervised_model: str,
    feature_mode: str,
    inference_cost_per_row: float,
    fixed_threshold_value: Optional[float],
) -> Dict[str, object]:
    train_rows = load_split_rows(ds_dir, "train", label_field)
    val_rows = load_split_rows(ds_dir, "val", label_field)
    test_rows = load_split_rows(ds_dir, "test", label_field)
    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError(f"Missing splits in {ds_dir}")

    p_train, p_val, p_test, selected_model = b.train_supervised_model(
        train_rows,
        val_rows,
        test_rows,
        include_missing_indicators=include_missing_indicators,
        preferred_model=supervised_model,
        feature_mode=feature_mode,
    )

    if fixed_threshold_value is None:
        th = choose_thresholds_coordinate_descent(
            val_rows,
            p_val,
            max_success_false_stop_rate=max_success_false_stop_rate,
            lambda_false_stop=lambda_false_stop,
            inference_cost_per_row=inference_cost_per_row,
            enforce_constraint=(not disable_threshold_constraint),
            rounds=3,
        )
    else:
        th = fixed_thresholds(fixed_threshold_value)

    m_test, pred_test_rows = run_level_eval(
        test_rows,
        p_test,
        thresholds=th,
        lambda_false_stop=lambda_false_stop,
        inference_cost_per_row=inference_cost_per_row,
    )

    os.makedirs(output_dir, exist_ok=True)
    write_rows_csv(os.path.join(output_dir, "predictions_xgb_target_test.csv"), pred_test_rows)

    row = {
        "generated_at": now_utc_iso(),
        "dataset": ds_name(ds_dir),
        "dataset_dir": ds_dir,
        "status": "ok",
        "selected_model": selected_model,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "thresholds": th,
        "metrics_test": m_test,
    }
    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Run XGBoost with run-level sequential protocol.")
    ap.add_argument("--mode", choices=["single", "lodo"], required=True)
    ap.add_argument("--dataset-dir-glob", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-field", default="strict_success")
    ap.add_argument("--source-splits", default="train,val")
    ap.add_argument("--max-success-false-stop-rate", type=float, default=0.10)
    ap.add_argument("--lambda-false-stop", type=float, default=1.0)
    ap.add_argument("--disable-threshold-constraint", action="store_true")
    ap.add_argument("--include-missing-indicators", action="store_true")
    ap.add_argument("--supervised-model", choices=["auto", "xgboost", "hgb", "logreg"], default="xgboost")
    ap.add_argument("--feature-mode", choices=["param_plus_prefix", "prefix_only", "param_only"], default="param_plus_prefix")
    ap.add_argument("--inference-cost-per-row", type=float, default=0.0)
    ap.add_argument(
        "--fixed-threshold",
        type=float,
        default=-1.0,
        help="When >=0, skip threshold search and use this fixed threshold for all checkpoints.",
    )
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    b.TARGET_LABEL_FIELD = str(args.label_field)
    source_splits = parse_source_splits(args.source_splits)

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
        raise SystemExit("No datasets with track1_strict/splits/{train,val,test}.csv")

    os.makedirs(args.output_dir, exist_ok=True)
    fixed_threshold_value: Optional[float] = None
    if float(args.fixed_threshold) >= 0.0:
        fixed_threshold_value = float(args.fixed_threshold)

    if args.mode == "single":
        fold_rows: List[Dict[str, str]] = []
        for d in ok_dirs:
            name = ds_name(d)
            out = os.path.join(args.output_dir, name)
            try:
                obj = run_single_dataset(
                    d,
                    out,
                    label_field=args.label_field,
                    lambda_false_stop=float(args.lambda_false_stop),
                    max_success_false_stop_rate=float(args.max_success_false_stop_rate),
                    disable_threshold_constraint=bool(args.disable_threshold_constraint),
                    include_missing_indicators=bool(args.include_missing_indicators),
                    supervised_model=str(args.supervised_model),
                    feature_mode=str(args.feature_mode),
                    inference_cost_per_row=float(args.inference_cost_per_row),
                    fixed_threshold_value=fixed_threshold_value,
                )
                mt = obj.get("metrics_test", {})
                th = obj.get("thresholds", {})
                fold_rows.append(
                    {
                        "dataset": name,
                        "status": "ok",
                        "selected_model": str(obj.get("selected_model", "")),
                        "n_runs": str(float(mt.get("n_runs", 0.0))),
                        "accuracy": str(float(mt.get("accuracy", 0.0))),
                        "fail_recall_stop": str(float(mt.get("fail_recall_stop", 0.0))),
                        "success_false_stop_rate": str(float(mt.get("success_false_stop_rate", 0.0))),
                        "stop_rate": str(float(mt.get("stop_rate", 0.0))),
                        "avg_inference_steps": str(float(mt.get("avg_inference_steps", 0.0))),
                        "net_with_infer": str(float(mt.get("avg_weighted_net_saved_seconds_with_inference", 0.0))),
                        "oracle": str(float(mt.get("oracle_avg_weighted_net_saved_seconds", 0.0))),
                        "headroom_ratio_vs_oracle": str(float(mt.get("headroom_ratio_vs_oracle", 0.0))),
                        "stop_precision_on_stopped": str(float(mt.get("stop_precision_on_stopped", 0.0))),
                        "threshold_init": str(float(th.get("init", 0.0))),
                        "threshold_place": str(float(th.get("place", 0.0))),
                        "threshold_cts": str(float(th.get("cts", 0.0))),
                        "threshold_grt": str(float(th.get("grt", 0.0))),
                        "threshold_drt": str(float(th.get("drt", 0.0))),
                    }
                )
            except Exception as e:  # noqa: BLE001
                fold_rows.append({"dataset": name, "status": f"failed:{e}"})

        fields = list(fold_rows[0].keys()) if fold_rows else ["dataset", "status"]
        with open(os.path.join(args.output_dir, "single_folds.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in fold_rows:
                w.writerow(r)

        ok = [r for r in fold_rows if r.get("status") == "ok"]
        metric_keys = [
            "accuracy",
            "fail_recall_stop",
            "success_false_stop_rate",
            "stop_rate",
            "avg_inference_steps",
            "net_with_infer",
            "oracle",
            "headroom_ratio_vs_oracle",
            "stop_precision_on_stopped",
        ]
        macro = {}
        ci95 = {}
        for k in metric_keys:
            vals = [float(r.get(k, "0") or 0.0) for r in ok]
            macro[k] = float(sum(vals) / len(vals)) if vals else 0.0
            lo, hi = bootstrap_ci_mean(vals, args.bootstrap_iters, args.seed + int.from_bytes(__import__("hashlib").sha256(str(k).encode()).digest()[:8], "little") % 10007)
            ci95[k] = {"lo": lo, "hi": hi}

        summary = {
            "generated_at": now_utc_iso(),
            "mode": "single",
            "dataset_count": len(ok_dirs),
            "folds_total": len(fold_rows),
            "folds_ok": len(ok),
            "label_field": args.label_field,
            "max_success_false_stop_rate": args.max_success_false_stop_rate,
            "threshold_constraint_enabled": (not args.disable_threshold_constraint),
            "lambda_false_stop": args.lambda_false_stop,
            "supervised_model_preference": args.supervised_model,
            "feature_mode": args.feature_mode,
            "inference_cost_per_row": args.inference_cost_per_row,
            "fixed_threshold": (None if fixed_threshold_value is None else fixed_threshold_value),
            "macro_mean": macro,
            "macro_mean_ci95_bootstrap": ci95,
        }
        with open(os.path.join(args.output_dir, "single_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return

    # LODO mode
    if len(ok_dirs) < 2:
        raise SystemExit("Need at least 2 datasets for LODO.")

    fold_rows: List[Dict[str, str]] = []
    for target_dir in ok_dirs:
        target_name = ds_name(target_dir)
        fold_dir = os.path.join(args.output_dir, target_name)
        os.makedirs(fold_dir, exist_ok=True)

        source_rows: List[Dict[str, str]] = []
        for src_dir in ok_dirs:
            if src_dir == target_dir:
                continue
            for sp in source_splits:
                source_rows.extend(load_split_rows(src_dir, sp, args.label_field))

        val_rows = load_split_rows(target_dir, "val", args.label_field)
        test_rows = load_split_rows(target_dir, "test", args.label_field)

        if not source_rows or not val_rows or not test_rows:
            fold_rows.append(
                {
                    "target_dataset": target_name,
                    "status": "skipped_missing_rows",
                    "source_train_rows": str(len(source_rows)),
                    "target_val_rows": str(len(val_rows)),
                    "target_test_rows": str(len(test_rows)),
                }
            )
            continue

        try:
            p_train, p_val, p_test, selected_model = b.train_supervised_model(
                source_rows,
                val_rows,
                test_rows,
                include_missing_indicators=bool(args.include_missing_indicators),
                preferred_model=str(args.supervised_model),
                feature_mode=str(args.feature_mode),
            )
            if fixed_threshold_value is None:
                th = choose_thresholds_coordinate_descent(
                    val_rows,
                    p_val,
                    max_success_false_stop_rate=float(args.max_success_false_stop_rate),
                    lambda_false_stop=float(args.lambda_false_stop),
                    inference_cost_per_row=float(args.inference_cost_per_row),
                    enforce_constraint=(not args.disable_threshold_constraint),
                    rounds=3,
                )
            else:
                th = fixed_thresholds(fixed_threshold_value)
            mt, pred_rows = run_level_eval(
                test_rows,
                p_test,
                thresholds=th,
                lambda_false_stop=float(args.lambda_false_stop),
                inference_cost_per_row=float(args.inference_cost_per_row),
            )

            write_rows_csv(os.path.join(fold_dir, "predictions_xgb_target_test.csv"), pred_rows)
            with open(os.path.join(fold_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "generated_at": now_utc_iso(),
                        "target_dataset": target_name,
                        "target_dir": target_dir,
                        "selected_model": selected_model,
                        "thresholds": th,
                        "metrics_test": mt,
                    },
                    f,
                    indent=2,
                )

            fold_rows.append(
                {
                    "target_dataset": target_name,
                    "status": "ok",
                    "selected_model": selected_model,
                    "source_train_rows": str(len(source_rows)),
                    "target_val_rows": str(len(val_rows)),
                    "target_test_rows": str(len(test_rows)),
                    "accuracy": str(float(mt.get("accuracy", 0.0))),
                    "fail_recall_stop": str(float(mt.get("fail_recall_stop", 0.0))),
                    "success_false_stop_rate": str(float(mt.get("success_false_stop_rate", 0.0))),
                    "stop_rate": str(float(mt.get("stop_rate", 0.0))),
                    "avg_inference_steps": str(float(mt.get("avg_inference_steps", 0.0))),
                    "net_with_infer": str(float(mt.get("avg_weighted_net_saved_seconds_with_inference", 0.0))),
                    "oracle": str(float(mt.get("oracle_avg_weighted_net_saved_seconds", 0.0))),
                    "headroom_ratio_vs_oracle": str(float(mt.get("headroom_ratio_vs_oracle", 0.0))),
                    "stop_precision_on_stopped": str(float(mt.get("stop_precision_on_stopped", 0.0))),
                    "threshold_init": str(float(th.get("init", 0.0))),
                    "threshold_place": str(float(th.get("place", 0.0))),
                    "threshold_cts": str(float(th.get("cts", 0.0))),
                    "threshold_grt": str(float(th.get("grt", 0.0))),
                    "threshold_drt": str(float(th.get("drt", 0.0))),
                }
            )
        except Exception as e:  # noqa: BLE001
            fold_rows.append({"target_dataset": target_name, "status": f"failed:{e}"})

    fields = list(fold_rows[0].keys()) if fold_rows else ["target_dataset", "status"]
    with open(os.path.join(args.output_dir, "lodo_folds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in fold_rows:
            w.writerow(r)

    ok = [r for r in fold_rows if r.get("status") == "ok"]
    metric_keys = [
        "accuracy",
        "fail_recall_stop",
        "success_false_stop_rate",
        "stop_rate",
        "avg_inference_steps",
        "net_with_infer",
        "oracle",
        "headroom_ratio_vs_oracle",
        "stop_precision_on_stopped",
    ]
    macro = {}
    ci95 = {}
    for k in metric_keys:
        vals = [float(r.get(k, "0") or 0.0) for r in ok]
        macro[k] = float(sum(vals) / len(vals)) if vals else 0.0
        lo, hi = bootstrap_ci_mean(vals, args.bootstrap_iters, args.seed + int.from_bytes(__import__("hashlib").sha256(str(k).encode()).digest()[:8], "little") % 10007)
        ci95[k] = {"lo": lo, "hi": hi}

    worst_fold = None
    if ok:
        worst_fold = min(ok, key=lambda r: float(r.get("net_with_infer", "0") or 0.0))

    summary = {
        "generated_at": now_utc_iso(),
        "mode": "lodo",
        "dataset_count": len(ok_dirs),
        "folds_total": len(fold_rows),
        "folds_ok": len(ok),
        "label_field": args.label_field,
        "source_splits": source_splits,
        "max_success_false_stop_rate": args.max_success_false_stop_rate,
        "threshold_constraint_enabled": (not args.disable_threshold_constraint),
        "lambda_false_stop": args.lambda_false_stop,
        "supervised_model_preference": args.supervised_model,
        "feature_mode": args.feature_mode,
        "inference_cost_per_row": args.inference_cost_per_row,
        "fixed_threshold": (None if fixed_threshold_value is None else fixed_threshold_value),
        "macro_mean": macro,
        "macro_mean_ci95_bootstrap": ci95,
        "worst_fold_by_net": worst_fold,
    }
    with open(os.path.join(args.output_dir, "lodo_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
