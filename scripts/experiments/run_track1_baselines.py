#!/usr/bin/env python3
"""
Run Track-1 baselines:
  1) Rule-based triage
  2) XGBoost-like supervised baseline (xgboost if available, fallback to sklearn HGB)
  3) LLM triage (DeepSeek/OpenAI/Anthropic chat API)

Outputs:
  - metrics_summary.json
  - per_model_predictions_*.csv
  - metrics_table.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import math
import os
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


TARGET_LABEL_FIELD = "strict_success"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maybe_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def sigmoid(x: float) -> float:
    x = max(min(x, 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-x))


def parse_prefix_metrics(metrics_json: str) -> Dict[str, Optional[float]]:
    if not metrics_json:
        return {}
    try:
        obj = json.loads(metrics_json)
        out = {}
        for k, v in obj.items():
            if v is None:
                out[k] = None
            else:
                out[k] = maybe_float(v)
        return out
    except Exception:
        return {}


PREFIX_METRIC_KEYS = [
    "dpl_hpwl",
    "congestion",
    "overflow",
    "density_overflow",
    "worst_slack",
    "tns_metric",
    "design_area",
    "utilization",
    "rpt_wns",
    "rpt_tns",
    "rpt_worst_slack",
    "rpt_setup_skew",
    "rpt_has_violated_path",
    "rpt_exists",
    "rpt_nonempty",
    "prefix_elapsed_seconds",
]

RUN_PARAM_NUMERIC_KEYS = [
    "core_util",
    "lb_addon",
    "tns_param",
    "cts_size",
    "cts_diameter",
    "cell_pad_global",
    "cell_pad_detail",
    "clock_period",
    "clock_uncertainty",
    "clock_transition",
]

RUN_PARAM_BOOL_KEYS = [
    "synth_flatten",
    "enable_dpo",
]


def maybe_bool01(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s == "":
        return None
    if s in {"1", "true", "yes", "y", "on"}:
        return 1.0
    if s in {"0", "false", "no", "n", "off"}:
        return 0.0
    v = maybe_float(s)
    if v is None:
        return None
    return 1.0 if float(v) != 0.0 else 0.0


def maybe_layer_index(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    m = re.search(r"(?:^|[^0-9])([0-9]+)(?:[^0-9]|$)", s)
    if m:
        try:
            return float(int(m.group(1)))
        except Exception:
            pass
    return maybe_float(s)


def load_rows(
    csv_path: str,
    require_binary_label: bool = True,
    label_field: str = "strict_success",
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if require_binary_label and r.get(label_field, "") not in ("0", "1"):
                continue
            rows.append(r)
    return rows


def extract_features(row: Dict[str, str], include_missing_indicators: bool = False) -> Dict[str, float]:
    return extract_features_with_mode(
        row=row,
        include_missing_indicators=include_missing_indicators,
        feature_mode="param_plus_prefix",
    )


def extract_features_with_mode(
    row: Dict[str, str],
    include_missing_indicators: bool = False,
    feature_mode: str = "param_plus_prefix",
) -> Dict[str, float]:
    mode = str(feature_mode or "param_plus_prefix").strip().lower()
    use_prefix = mode in {"param_plus_prefix", "prefix_only"}
    use_param = mode in {"param_plus_prefix", "param_only"}

    pm = parse_prefix_metrics(row.get("prefix_metrics", ""))
    feat: Dict[str, float] = {}
    if use_prefix:
        for k in PREFIX_METRIC_KEYS:
            v = pm.get(k)
            if v is None:
                feat[k] = 0.0
                if include_missing_indicators:
                    feat[f"{k}_missing"] = 1.0
            else:
                feat[k] = float(v)
                if include_missing_indicators:
                    feat[f"{k}_missing"] = 0.0

    # Static run parameters from config.
    if use_param:
        for k in RUN_PARAM_NUMERIC_KEYS:
            v = maybe_float(row.get(k, ""))
            if v is None:
                feat[k] = 0.0
                if include_missing_indicators:
                    feat[f"{k}_missing"] = 1.0
            else:
                feat[k] = float(v)
                if include_missing_indicators:
                    feat[f"{k}_missing"] = 0.0

        for k in RUN_PARAM_BOOL_KEYS:
            v = maybe_bool01(row.get(k, ""))
            if v is None:
                feat[k] = 0.0
                if include_missing_indicators:
                    feat[f"{k}_missing"] = 1.0
            else:
                feat[k] = float(v)
                if include_missing_indicators:
                    feat[f"{k}_missing"] = 0.0

        pin_layer = maybe_layer_index(row.get("pin_layer", ""))
        above_layer = maybe_layer_index(row.get("above_layer", ""))
        if pin_layer is None:
            feat["pin_layer_idx"] = 0.0
            if include_missing_indicators:
                feat["pin_layer_idx_missing"] = 1.0
        else:
            feat["pin_layer_idx"] = float(pin_layer)
            if include_missing_indicators:
                feat["pin_layer_idx_missing"] = 0.0
        if above_layer is None:
            feat["above_layer_idx"] = 0.0
            if include_missing_indicators:
                feat["above_layer_idx_missing"] = 1.0
        else:
            feat["above_layer_idx"] = float(above_layer)
            if include_missing_indicators:
                feat["above_layer_idx_missing"] = 0.0
        if pin_layer is None or above_layer is None:
            feat["layer_span"] = 0.0
            if include_missing_indicators:
                feat["layer_span_missing"] = 1.0
        else:
            feat["layer_span"] = float(above_layer - pin_layer)
            if include_missing_indicators:
                feat["layer_span_missing"] = 0.0

    # One-hot checkpoint.
    cp = row.get("checkpoint", "")
    for c in ("place", "cts", "grt", "drt"):
        feat[f"cp_{c}"] = 1.0 if cp == c else 0.0
    return feat


def target_success(row: Dict[str, str]) -> int:
    v = str(row.get(TARGET_LABEL_FIELD, "")).strip()
    if v not in ("0", "1"):
        v = str(row.get("strict_success", "0")).strip()
    return 1 if v == "1" else 0


def row_remaining_seconds(row: Dict[str, str]) -> float:
    p = maybe_float(row.get("prefix_elapsed_seconds", ""))
    f = maybe_float(row.get("final_wallclock_seconds", ""))
    if p is None or f is None:
        return 0.0
    return max(0.0, f - p)


def row_total_seconds(row: Dict[str, str]) -> float:
    f = maybe_float(row.get("final_wallclock_seconds", ""))
    if f is None:
        return 0.0
    return max(0.0, f)


def _adv_group_key(row: Dict[str, str]) -> str:
    design = row.get("design", "")
    objective = row.get("objective_hint", "")
    checkpoint = row.get("checkpoint", "")
    return f"{design}|{objective}|{checkpoint}"


def compute_group_relative_advantage(
    rows: List[Dict[str, str]],
    y: List[int],
    stop_decision: List[int],
    lambda_false_stop: float = 1.0,
) -> Dict[str, float]:
    group_vals: Dict[str, List[float]] = defaultdict(list)
    point_vals: List[float] = []
    point_groups: List[str] = []

    for i, r in enumerate(rows):
        remain = row_remaining_seconds(r)
        val = remain if y[i] == 0 else (-lambda_false_stop * remain)
        g = _adv_group_key(r)
        group_vals[g].append(val)
        point_vals.append(val)
        point_groups.append(g)

    group_ref: Dict[str, Tuple[float, float]] = {}
    all_vals: List[float] = []
    for vals in group_vals.values():
        all_vals.extend(vals)
    global_med = statistics.median(all_vals) if all_vals else 0.0
    global_mad = statistics.median([abs(v - global_med) for v in all_vals]) if all_vals else 1.0
    if global_mad < 1e-6:
        global_mad = 1.0

    for g, vals in group_vals.items():
        med = statistics.median(vals) if vals else global_med
        mad = statistics.median([abs(v - med) for v in vals]) if vals else global_mad
        if mad < 1e-6:
            med = global_med
            mad = global_mad
        group_ref[g] = (float(med), float(mad))

    contrib: List[float] = []
    stop_contrib: List[float] = []
    for i in range(len(rows)):
        if stop_decision[i] != 1:
            contrib.append(0.0)
            continue
        g = point_groups[i]
        med, mad = group_ref.get(g, (global_med, global_mad))
        a = (point_vals[i] - med) / (mad + 1e-6)
        contrib.append(a)
        stop_contrib.append(a)

    avg_adv = sum(contrib) / len(contrib) if contrib else 0.0
    avg_adv_on_stop = sum(stop_contrib) / len(stop_contrib) if stop_contrib else 0.0
    pos_rate_stop = (sum(1 for v in stop_contrib if v > 0.0) / len(stop_contrib)) if stop_contrib else 0.0
    return {
        "avg_group_relative_advantage": float(avg_adv),
        "avg_group_relative_advantage_on_stop": float(avg_adv_on_stop),
        "stop_positive_advantage_rate": float(pos_rate_stop),
    }


def evaluate_predictions(
    rows: List[Dict[str, str]],
    success_prob: List[float],
    stop_threshold: float,
    lambda_false_stop: float = 1.0,
    inference_cost_per_row: float = 0.0,
) -> Dict[str, float]:
    assert len(rows) == len(success_prob)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "fail_recall_stop": 0.0,
            "success_false_stop_rate": 0.0,
            "avg_saved_seconds": 0.0,
            "avg_wasted_seconds": 0.0,
            "avg_net_saved_seconds": 0.0,
            "avg_weighted_wasted_seconds": 0.0,
            "avg_weighted_net_saved_seconds": 0.0,
            "avg_inference_cost_seconds": 0.0,
            "avg_weighted_net_saved_seconds_with_inference": 0.0,
            "avg_saved_seconds_observed": 0.0,
            "avg_wasted_seconds_observed": 0.0,
            "avg_net_saved_seconds_observed": 0.0,
            "avg_weighted_net_saved_seconds_observed": 0.0,
            "cost_observed_rate": 0.0,
            "avg_net_saved_ratio": 0.0,
            "avg_group_relative_advantage": 0.0,
            "avg_group_relative_advantage_on_stop": 0.0,
            "stop_positive_advantage_rate": 0.0,
            "stop_rate": 0.0,
        }

    y = [target_success(r) for r in rows]  # 1 success, 0 fail
    y_pred_success = [1 if p >= 0.5 else 0 for p in success_prob]
    stop_decision = [1 if p < stop_threshold else 0 for p in success_prob]  # 1 STOP

    correct = sum(1 for a, b in zip(y, y_pred_success) if a == b)
    accuracy = correct / n

    fail_idx = [i for i, yy in enumerate(y) if yy == 0]
    succ_idx = [i for i, yy in enumerate(y) if yy == 1]

    fail_recall_stop = (
        sum(stop_decision[i] for i in fail_idx) / len(fail_idx) if fail_idx else 0.0
    )
    success_false_stop_rate = (
        sum(stop_decision[i] for i in succ_idx) / len(succ_idx) if succ_idx else 0.0
    )

    saved = []
    wasted = []
    weighted_wasted = []
    cost_observed = []
    for i, r in enumerate(rows):
        psec = maybe_float(r.get("prefix_elapsed_seconds", ""))
        fsec = maybe_float(r.get("final_wallclock_seconds", ""))
        has_cost = (psec is not None and fsec is not None)
        cost_observed.append(1 if has_cost else 0)

        remain = row_remaining_seconds(r) if has_cost else 0.0
        if stop_decision[i] == 1 and y[i] == 0:
            saved.append(remain)
            wasted.append(0.0)
            weighted_wasted.append(0.0)
        elif stop_decision[i] == 1 and y[i] == 1:
            saved.append(0.0)
            wasted.append(remain)
            weighted_wasted.append(lambda_false_stop * remain)
        else:
            saved.append(0.0)
            wasted.append(0.0)
            weighted_wasted.append(0.0)

    avg_saved = sum(saved) / n
    avg_wasted = sum(wasted) / n
    avg_weighted_wasted = sum(weighted_wasted) / n
    avg_net = avg_saved - avg_wasted
    avg_weighted_net = avg_saved - avg_weighted_wasted

    avg_infer = max(0.0, float(inference_cost_per_row))
    avg_weighted_net_with_infer = avg_weighted_net - avg_infer

    obs_n = sum(cost_observed)
    if obs_n > 0:
        avg_saved_obs = sum(v for v, o in zip(saved, cost_observed) if o == 1) / obs_n
        avg_wasted_obs = sum(v for v, o in zip(wasted, cost_observed) if o == 1) / obs_n
        avg_weighted_wasted_obs = sum(v for v, o in zip(weighted_wasted, cost_observed) if o == 1) / obs_n
    else:
        avg_saved_obs = 0.0
        avg_wasted_obs = 0.0
        avg_weighted_wasted_obs = 0.0
    avg_net_obs = avg_saved_obs - avg_wasted_obs
    avg_weighted_net_obs = avg_saved_obs - avg_weighted_wasted_obs

    net_ratio_vals: List[float] = []
    for i, r in enumerate(rows):
        tot = row_total_seconds(r)
        if tot <= 0.0:
            net_ratio_vals.append(0.0)
            continue
        local = 0.0
        if stop_decision[i] == 1 and y[i] == 0:
            local = row_remaining_seconds(r) / tot
        elif stop_decision[i] == 1 and y[i] == 1:
            local = -(lambda_false_stop * row_remaining_seconds(r)) / tot
        net_ratio_vals.append(local)
    avg_net_ratio = sum(net_ratio_vals) / n if n else 0.0

    adv = compute_group_relative_advantage(rows, y, stop_decision, lambda_false_stop=lambda_false_stop)

    return {
        "n": n,
        "accuracy": accuracy,
        "fail_recall_stop": fail_recall_stop,
        "success_false_stop_rate": success_false_stop_rate,
        "avg_saved_seconds": avg_saved,
        "avg_wasted_seconds": avg_wasted,
        "avg_net_saved_seconds": avg_net,
        "avg_weighted_wasted_seconds": avg_weighted_wasted,
        "avg_weighted_net_saved_seconds": avg_weighted_net,
        "avg_inference_cost_seconds": avg_infer,
        "avg_weighted_net_saved_seconds_with_inference": avg_weighted_net_with_infer,
        "avg_saved_seconds_observed": avg_saved_obs,
        "avg_wasted_seconds_observed": avg_wasted_obs,
        "avg_net_saved_seconds_observed": avg_net_obs,
        "avg_weighted_net_saved_seconds_observed": avg_weighted_net_obs,
        "cost_observed_rate": (obs_n / n if n else 0.0),
        "avg_net_saved_ratio": avg_net_ratio,
        "avg_group_relative_advantage": adv["avg_group_relative_advantage"],
        "avg_group_relative_advantage_on_stop": adv["avg_group_relative_advantage_on_stop"],
        "stop_positive_advantage_rate": adv["stop_positive_advantage_rate"],
        "stop_rate": sum(stop_decision) / n,
        "stop_threshold": stop_threshold,
    }


def evaluate_with_fixed_stop(
    rows: List[Dict[str, str]],
    stop_decision: List[int],
    lambda_false_stop: float = 1.0,
    inference_cost_per_row: float = 0.0,
    success_prob: Optional[List[float]] = None,
    stop_threshold: float = -1.0,
) -> Dict[str, float]:
    assert len(rows) == len(stop_decision)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "fail_recall_stop": 0.0,
            "success_false_stop_rate": 0.0,
            "avg_saved_seconds": 0.0,
            "avg_wasted_seconds": 0.0,
            "avg_net_saved_seconds": 0.0,
            "avg_weighted_wasted_seconds": 0.0,
            "avg_weighted_net_saved_seconds": 0.0,
            "avg_inference_cost_seconds": 0.0,
            "avg_weighted_net_saved_seconds_with_inference": 0.0,
            "avg_saved_seconds_observed": 0.0,
            "avg_wasted_seconds_observed": 0.0,
            "avg_net_saved_seconds_observed": 0.0,
            "avg_weighted_net_saved_seconds_observed": 0.0,
            "cost_observed_rate": 0.0,
            "avg_net_saved_ratio": 0.0,
            "avg_group_relative_advantage": 0.0,
            "avg_group_relative_advantage_on_stop": 0.0,
            "stop_positive_advantage_rate": 0.0,
            "stop_rate": 0.0,
            "stop_threshold": stop_threshold,
        }

    y = [target_success(r) for r in rows]
    if success_prob is not None and len(success_prob) == n:
        y_pred_success = [1 if p >= 0.5 else 0 for p in success_prob]
    else:
        y_pred_success = [0 if s == 1 else 1 for s in stop_decision]

    correct = sum(1 for a, b in zip(y, y_pred_success) if a == b)
    accuracy = correct / n

    fail_idx = [i for i, yy in enumerate(y) if yy == 0]
    succ_idx = [i for i, yy in enumerate(y) if yy == 1]
    fail_recall_stop = (
        sum(stop_decision[i] for i in fail_idx) / len(fail_idx) if fail_idx else 0.0
    )
    success_false_stop_rate = (
        sum(stop_decision[i] for i in succ_idx) / len(succ_idx) if succ_idx else 0.0
    )

    saved = []
    wasted = []
    weighted_wasted = []
    cost_observed = []
    for i, r in enumerate(rows):
        psec = maybe_float(r.get("prefix_elapsed_seconds", ""))
        fsec = maybe_float(r.get("final_wallclock_seconds", ""))
        has_cost = (psec is not None and fsec is not None)
        cost_observed.append(1 if has_cost else 0)

        remain = row_remaining_seconds(r) if has_cost else 0.0
        if stop_decision[i] == 1 and y[i] == 0:
            saved.append(remain)
            wasted.append(0.0)
            weighted_wasted.append(0.0)
        elif stop_decision[i] == 1 and y[i] == 1:
            saved.append(0.0)
            wasted.append(remain)
            weighted_wasted.append(lambda_false_stop * remain)
        else:
            saved.append(0.0)
            wasted.append(0.0)
            weighted_wasted.append(0.0)

    avg_saved = sum(saved) / n
    avg_wasted = sum(wasted) / n
    avg_weighted_wasted = sum(weighted_wasted) / n
    avg_net = avg_saved - avg_wasted
    avg_weighted_net = avg_saved - avg_weighted_wasted

    avg_infer = max(0.0, float(inference_cost_per_row))
    avg_weighted_net_with_infer = avg_weighted_net - avg_infer

    obs_n = sum(cost_observed)
    if obs_n > 0:
        avg_saved_obs = sum(v for v, o in zip(saved, cost_observed) if o == 1) / obs_n
        avg_wasted_obs = sum(v for v, o in zip(wasted, cost_observed) if o == 1) / obs_n
        avg_weighted_wasted_obs = sum(v for v, o in zip(weighted_wasted, cost_observed) if o == 1) / obs_n
    else:
        avg_saved_obs = 0.0
        avg_wasted_obs = 0.0
        avg_weighted_wasted_obs = 0.0
    avg_net_obs = avg_saved_obs - avg_wasted_obs
    avg_weighted_net_obs = avg_saved_obs - avg_weighted_wasted_obs

    net_ratio_vals: List[float] = []
    for i, r in enumerate(rows):
        tot = row_total_seconds(r)
        if tot <= 0.0:
            net_ratio_vals.append(0.0)
            continue
        local = 0.0
        if stop_decision[i] == 1 and y[i] == 0:
            local = row_remaining_seconds(r) / tot
        elif stop_decision[i] == 1 and y[i] == 1:
            local = -(lambda_false_stop * row_remaining_seconds(r)) / tot
        net_ratio_vals.append(local)
    avg_net_ratio = sum(net_ratio_vals) / n if n else 0.0

    adv = compute_group_relative_advantage(rows, y, stop_decision, lambda_false_stop=lambda_false_stop)

    return {
        "n": n,
        "accuracy": accuracy,
        "fail_recall_stop": fail_recall_stop,
        "success_false_stop_rate": success_false_stop_rate,
        "avg_saved_seconds": avg_saved,
        "avg_wasted_seconds": avg_wasted,
        "avg_net_saved_seconds": avg_net,
        "avg_weighted_wasted_seconds": avg_weighted_wasted,
        "avg_weighted_net_saved_seconds": avg_weighted_net,
        "avg_inference_cost_seconds": avg_infer,
        "avg_weighted_net_saved_seconds_with_inference": avg_weighted_net_with_infer,
        "avg_saved_seconds_observed": avg_saved_obs,
        "avg_wasted_seconds_observed": avg_wasted_obs,
        "avg_net_saved_seconds_observed": avg_net_obs,
        "avg_weighted_net_saved_seconds_observed": avg_weighted_net_obs,
        "cost_observed_rate": (obs_n / n if n else 0.0),
        "avg_net_saved_ratio": avg_net_ratio,
        "avg_group_relative_advantage": adv["avg_group_relative_advantage"],
        "avg_group_relative_advantage_on_stop": adv["avg_group_relative_advantage_on_stop"],
        "stop_positive_advantage_rate": adv["stop_positive_advantage_rate"],
        "stop_rate": sum(stop_decision) / n,
        "stop_threshold": stop_threshold,
    }


def fit_discrete_success_calibrator(
    rows_val: List[Dict[str, str]],
    success_prob_val: List[float],
    prior_strength: float = 2.0,
) -> Dict[str, object]:
    y_val = [target_success(r) for r in rows_val]
    buckets: Dict[float, List[float]] = defaultdict(list)
    for y, p in zip(y_val, success_prob_val):
        buckets[round(float(p), 6)].append(float(y))

    keys = sorted(buckets.keys())
    mapping: Dict[float, float] = {}
    prior = max(0.0, float(prior_strength))
    for k in keys:
        vals = buckets[k]
        n = len(vals)
        s = sum(vals)
        cal = (s + 0.5 * prior) / (n + prior) if n > 0 else k
        mapping[k] = max(0.0, min(1.0, float(cal)))
    return {"keys": keys, "mapping": mapping, "prior_strength": prior}


def apply_discrete_success_calibration(
    success_prob: List[float],
    calib: Dict[str, object],
) -> List[float]:
    keys: List[float] = list(calib.get("keys", []))  # type: ignore[assignment]
    mapping: Dict[float, float] = dict(calib.get("mapping", {}))  # type: ignore[assignment]
    if not keys:
        return [max(0.0, min(1.0, float(p))) for p in success_prob]

    out: List[float] = []
    for p in success_prob:
        q = round(float(p), 6)
        if q in mapping:
            out.append(float(mapping[q]))
            continue
        best = min(keys, key=lambda x: abs(x - q))
        out.append(float(mapping.get(best, best)))
    return out


def compute_oracle_upper_bound(
    rows: List[Dict[str, str]],
    lambda_false_stop: float = 1.0,
) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {
            "oracle_avg_saved_seconds": 0.0,
            "oracle_avg_weighted_net_saved_seconds": 0.0,
            "oracle_fail_count": 0.0,
            "oracle_fail_rate": 0.0,
        }

    saved = 0.0
    fail_count = 0
    for r in rows:
        y = target_success(r)
        if y == 0:
            fail_count += 1
            saved += row_remaining_seconds(r)

    avg_saved = saved / n
    # Oracle stops all and only fails, so weighted net == saved.
    avg_weighted_net = avg_saved
    return {
        "oracle_avg_saved_seconds": float(avg_saved),
        "oracle_avg_weighted_net_saved_seconds": float(avg_weighted_net),
        "oracle_fail_count": float(fail_count),
        "oracle_fail_rate": float(fail_count / n if n else 0.0),
    }


def enrich_with_oracle_headroom(
    rows: List[Dict[str, str]],
    m: Dict[str, float],
    lambda_false_stop: float = 1.0,
) -> Dict[str, float]:
    out = dict(m)
    oracle = compute_oracle_upper_bound(rows, lambda_false_stop=lambda_false_stop)
    out.update(oracle)
    den = float(oracle.get("oracle_avg_weighted_net_saved_seconds", 0.0))
    num = float(out.get("avg_weighted_net_saved_seconds_with_inference", 0.0))
    if den > 0:
        out["headroom_ratio_vs_oracle"] = num / den
    else:
        out["headroom_ratio_vs_oracle"] = 0.0
    return out


def scan_threshold_frontier(
    rows: List[Dict[str, str]],
    success_prob: List[float],
    lambda_false_stop: float,
    inference_cost_per_row: float,
    max_success_false_stop_rate: float,
    enforce_success_false_stop_constraint: bool,
    threshold_min: float = 0.0,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    t = threshold_min
    while t <= threshold_max + 1e-12:
        t2 = round(t, 4)
        m = evaluate_predictions(
            rows,
            success_prob,
            t2,
            lambda_false_stop=lambda_false_stop,
            inference_cost_per_row=inference_cost_per_row,
        )
        fsr = float(m.get("success_false_stop_rate", 1.0))
        feasible = (fsr <= max_success_false_stop_rate) if enforce_success_false_stop_constraint else True
        row = {"threshold": f"{t2:.4f}", "feasible_under_fsr": "1" if feasible else "0"}
        for k, v in m.items():
            row[k] = str(v)
        out.append(row)
        t += threshold_step
    return out


def write_frontier_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def choose_best_threshold(
    rows_val: List[Dict[str, str]],
    success_prob_val: List[float],
    metric: str = "avg_net_saved_seconds",
    max_success_false_stop_rate: float = 0.10,
    enforce_success_false_stop_constraint: bool = True,
    lambda_false_stop: float = 1.0,
    inference_cost_per_row: float = 0.0,
) -> float:
    best_t = 0.5
    best_v = -1e18
    best_fsr = 1e18

    fallback_t = 0.5
    fallback_fsr = 1e18
    fallback_v = -1e18

    for i in range(0, 96):
        t = i / 100.0
        m = evaluate_predictions(
            rows_val,
            success_prob_val,
            t,
            lambda_false_stop=lambda_false_stop,
            inference_cost_per_row=inference_cost_per_row,
        )
        v = float(m.get(metric, 0.0))
        fsr = float(m.get("success_false_stop_rate", 1.0))

        # Always track fallback: minimize false-stop first, then maximize target metric.
        if (fsr < fallback_fsr) or (abs(fsr - fallback_fsr) < 1e-12 and v > fallback_v):
            fallback_fsr = fsr
            fallback_v = v
            fallback_t = t

        if enforce_success_false_stop_constraint and fsr > max_success_false_stop_rate:
            continue

        # Feasible region: maximize metric, tie-break with lower false-stop.
        if (v > best_v) or (abs(v - best_v) < 1e-12 and fsr < best_fsr):
            best_v = v
            best_fsr = fsr
            best_t = t

    if best_v == -1e18:
        return fallback_t
    return best_t


def build_quantile_edges(values: List[float], n_bins: int) -> List[float]:
    if n_bins <= 1 or not values:
        return []
    arr = sorted(values)
    edges: List[float] = []
    n = len(arr)
    for i in range(1, n_bins):
        idx = int(round(i * n / n_bins))
        idx = max(1, min(n - 1, idx))
        edges.append(float(arr[idx]))
    # monotonic unique edges
    out: List[float] = []
    last = None
    for e in edges:
        if last is None or e > last:
            out.append(e)
            last = e
    return out


def assign_bin(v: float, edges: List[float]) -> int:
    for i, e in enumerate(edges):
        if v <= e:
            return i
    return len(edges)


def cgr_group_key(row: Dict[str, str], level: int, elapsed_edges: List[float]) -> str:
    design = row.get("design", "")
    objective = row.get("objective_hint", "")
    checkpoint = row.get("checkpoint", "")

    pe = maybe_float(row.get("prefix_elapsed_seconds", ""))
    elapsed_bin = "na" if pe is None else f"b{assign_bin(float(pe), elapsed_edges)}"

    if level == 0:
        return f"{design}|{objective}|{checkpoint}|{elapsed_bin}"
    if level == 1:
        return f"{design}|{objective}|{checkpoint}"
    return checkpoint


def build_cgr_model(
    train_rows: List[Dict[str, str]],
    elapsed_bins: int = 4,
    min_group_total: int = 80,
    min_group_each: int = 15,
    lambda_cost: float = 1.0,
    eta_uncertainty: float = 0.25,
    alpha: float = 1.0,
    beta: float = 1.0,
    risk_mix: float = 0.5,
) -> Dict[str, object]:
    elapsed_vals: List[float] = []
    for r in train_rows:
        pe = maybe_float(r.get("prefix_elapsed_seconds", ""))
        if pe is not None:
            elapsed_vals.append(float(pe))
    elapsed_edges = build_quantile_edges(elapsed_vals, elapsed_bins)

    raw_stats = [defaultdict(lambda: {
        "n": 0,
        "n_fail": 0,
        "n_succ": 0,
        "rem_fail_pos": [],
        "rem_succ_pos": [],
    }) for _ in range(3)]

    cp_costs: Dict[str, Dict[str, List[float]]] = {
        "place": {"fail": [], "succ": []},
        "cts": {"fail": [], "succ": []},
        "grt": {"fail": [], "succ": []},
        "drt": {"fail": [], "succ": []},
    }

    n_fail_all = 0
    n_total_all = 0

    for r in train_rows:
        y = target_success(r)
        rem = row_remaining_seconds(r)
        cp = r.get("checkpoint", "")

        n_total_all += 1
        if y == 0:
            n_fail_all += 1

        if rem > 0 and cp in cp_costs:
            if y == 0:
                cp_costs[cp]["fail"].append(rem)
            else:
                cp_costs[cp]["succ"].append(rem)

        for level in (0, 1, 2):
            key = cgr_group_key(r, level, elapsed_edges)
            st = raw_stats[level][key]
            st["n"] += 1
            if y == 0:
                st["n_fail"] += 1
                if rem > 0:
                    st["rem_fail_pos"].append(rem)
            else:
                st["n_succ"] += 1
                if rem > 0:
                    st["rem_succ_pos"].append(rem)

    all_succ_pos: List[float] = []
    all_fail_pos: List[float] = []
    for cp in cp_costs:
        all_succ_pos.extend(cp_costs[cp]["succ"])
        all_fail_pos.extend(cp_costs[cp]["fail"])

    global_succ_cost = statistics.median(all_succ_pos) if all_succ_pos else 1.0
    global_fail_cost = statistics.median(all_fail_pos) if all_fail_pos else global_succ_cost

    checkpoint_priors: Dict[str, Dict[str, float]] = {}
    for cp in ("place", "cts", "grt", "drt"):
        succ_vals = cp_costs[cp]["succ"]
        fail_vals = cp_costs[cp]["fail"]
        succ_med = statistics.median(succ_vals) if succ_vals else global_succ_cost
        fail_med = statistics.median(fail_vals) if fail_vals else succ_med
        checkpoint_priors[cp] = {"succ": float(succ_med), "fail": float(fail_med)}

    group_stats: List[Dict[str, Dict[str, float]]] = [{}, {}, {}]
    for level in (0, 1, 2):
        out_level: Dict[str, Dict[str, float]] = {}
        for key, st in raw_stats[level].items():
            med_fail = statistics.median(st["rem_fail_pos"]) if st["rem_fail_pos"] else None
            med_succ = statistics.median(st["rem_succ_pos"]) if st["rem_succ_pos"] else None
            out_level[key] = {
                "n": float(st["n"]),
                "n_fail": float(st["n_fail"]),
                "n_succ": float(st["n_succ"]),
                "med_fail": (float(med_fail) if med_fail is not None else None),
                "med_succ": (float(med_succ) if med_succ is not None else None),
            }
        group_stats[level] = out_level

    global_fail_rate = (n_fail_all / n_total_all) if n_total_all else 0.5

    return {
        "elapsed_edges": elapsed_edges,
        "group_stats": group_stats,
        "checkpoint_priors": checkpoint_priors,
        "global_succ_cost": float(global_succ_cost),
        "global_fail_cost": float(global_fail_cost),
        "global_fail_rate": float(global_fail_rate),
        "min_group_total": int(min_group_total),
        "min_group_each": int(min_group_each),
        "lambda_cost": float(lambda_cost),
        "eta_uncertainty": float(eta_uncertainty),
        "alpha": float(alpha),
        "beta": float(beta),
        "risk_mix": float(risk_mix),
    }


def cgr_resolve_group(row: Dict[str, str], model: Dict[str, object]) -> Tuple[Tuple[int, str], Dict[str, float]]:
    edges: List[float] = model["elapsed_edges"]  # type: ignore[assignment]
    group_stats: List[Dict[str, Dict[str, float]]] = model["group_stats"]  # type: ignore[assignment]
    min_total = int(model["min_group_total"])
    min_each = int(model["min_group_each"])

    # constrained best group
    for level in (0, 1, 2):
        key = cgr_group_key(row, level, edges)
        st = group_stats[level].get(key)
        if not st:
            continue
        if st["n"] >= min_total and st["n_fail"] >= min_each and st["n_succ"] >= min_each:
            return (level, key), st

    # fallback: first available by hierarchy
    for level in (0, 1, 2):
        key = cgr_group_key(row, level, edges)
        st = group_stats[level].get(key)
        if st:
            return (level, key), st

    return (2, row.get("checkpoint", "")), {
        "n": 0.0,
        "n_fail": 0.0,
        "n_succ": 0.0,
        "med_fail": None,
        "med_succ": None,
    }


def cgr_expected_stop_utility(row: Dict[str, str], model: Dict[str, object]) -> Tuple[float, Tuple[int, str]]:
    gid, st = cgr_resolve_group(row, model)

    cp = row.get("checkpoint", "")
    cp_priors: Dict[str, Dict[str, float]] = model["checkpoint_priors"]  # type: ignore[assignment]
    cp_cost = cp_priors.get(cp, {"succ": model["global_succ_cost"], "fail": model["global_fail_cost"]})

    n = float(st.get("n", 0.0))
    n_fail = float(st.get("n_fail", 0.0))
    alpha = float(model["alpha"])
    beta = float(model["beta"])

    if n > 0:
        p_fail = (n_fail + alpha) / (n + alpha + beta)
    else:
        p_fail = float(model["global_fail_rate"])

    # Mix group prior with row-level heuristic risk to avoid flat group-only scores.
    risk_mix = float(model.get("risk_mix", 0.5))
    risk_mix = max(0.0, min(1.0, risk_mix))
    row_fail_risk = 1.0 - rule_success_prob(row)
    p_fail = (1.0 - risk_mix) * p_fail + risk_mix * row_fail_risk
    p_fail = max(0.0, min(1.0, p_fail))

    med_fail = st.get("med_fail")
    med_succ = st.get("med_succ")
    c_fail = float(med_fail) if med_fail is not None else float(cp_cost["fail"])
    c_succ = float(med_succ) if med_succ is not None else float(cp_cost["succ"])

    uncertainty = math.sqrt(max(p_fail * (1.0 - p_fail), 0.0) / (n + 1.0))
    penalty_scale = c_fail + c_succ

    utility = (
        p_fail * c_fail
        - float(model["lambda_cost"]) * (1.0 - p_fail) * c_succ
        - float(model["eta_uncertainty"]) * uncertainty * penalty_scale
    )
    return utility, gid


def fit_cgr_reference(train_rows: List[Dict[str, str]], model: Dict[str, object]) -> None:
    group_vals: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    all_vals: List[float] = []
    for r in train_rows:
        u, gid = cgr_expected_stop_utility(r, model)
        group_vals[gid].append(u)
        all_vals.append(u)

    group_ref: Dict[Tuple[int, str], Dict[str, float]] = {}
    for gid, vals in group_vals.items():
        med = statistics.median(vals)
        mad = statistics.median([abs(x - med) for x in vals]) if vals else 0.0
        group_ref[gid] = {"median": float(med), "mad": float(mad)}

    global_med = statistics.median(all_vals) if all_vals else 0.0
    global_mad = statistics.median([abs(x - global_med) for x in all_vals]) if all_vals else 1.0
    if global_mad < 1e-6:
        global_mad = 1.0

    model["group_reference"] = group_ref
    model["global_reference"] = {"median": float(global_med), "mad": float(global_mad)}


def cgr_success_probabilities(rows: List[Dict[str, str]], model: Dict[str, object]) -> List[float]:
    group_ref: Dict[Tuple[int, str], Dict[str, float]] = model["group_reference"]  # type: ignore[assignment]
    global_ref: Dict[str, float] = model["global_reference"]  # type: ignore[assignment]

    probs: List[float] = []
    for r in rows:
        u, gid = cgr_expected_stop_utility(r, model)
        ref = group_ref.get(gid, global_ref)
        med = float(ref.get("median", global_ref["median"]))
        mad = float(ref.get("mad", global_ref["mad"]))
        # If group variance collapses, back off to global reference to keep ranking informative.
        if mad < 1e-6:
            med = float(global_ref["median"])
            mad = float(global_ref["mad"])
        a = (u - med) / (mad + 1e-6)
        # Higher relative utility for STOP -> lower success probability.
        probs.append(sigmoid(-a))
    return probs


def run_cgr_policy(
    train_rows: List[Dict[str, str]],
    val_rows: List[Dict[str, str]],
    test_rows: List[Dict[str, str]],
    elapsed_bins: int,
    min_group_total: int,
    min_group_each: int,
    lambda_cost: float,
    eta_uncertainty: float,
    alpha: float,
    beta: float,
    risk_mix: float,
) -> Tuple[List[float], List[float], List[float], str, Dict[str, object]]:
    model = build_cgr_model(
        train_rows=train_rows,
        elapsed_bins=elapsed_bins,
        min_group_total=min_group_total,
        min_group_each=min_group_each,
        lambda_cost=lambda_cost,
        eta_uncertainty=eta_uncertainty,
        alpha=alpha,
        beta=beta,
        risk_mix=risk_mix,
    )
    fit_cgr_reference(train_rows, model)

    p_train = cgr_success_probabilities(train_rows, model)
    p_val = cgr_success_probabilities(val_rows, model)
    p_test = cgr_success_probabilities(test_rows, model)

    # Direction calibration: ensure larger success_prob aligns with actual success labels.
    y_train = [target_success(r) for r in train_rows]
    if y_train:
        acc = sum(1 for yy, pp in zip(y_train, p_train) if (1 if pp >= 0.5 else 0) == yy) / len(y_train)
        acc_inv = sum(1 for yy, pp in zip(y_train, p_train) if (1 if (1.0 - pp) >= 0.5 else 0) == yy) / len(y_train)
        if acc_inv > acc:
            p_train = [1.0 - x for x in p_train]
            p_val = [1.0 - x for x in p_val]
            p_test = [1.0 - x for x in p_test]
            model["flipped"] = True
        else:
            model["flipped"] = False

    meta = {
        "elapsed_bins": elapsed_bins,
        "elapsed_edges": model.get("elapsed_edges", []),
        "min_group_total": min_group_total,
        "min_group_each": min_group_each,
        "lambda_cost": lambda_cost,
        "eta_uncertainty": eta_uncertainty,
        "alpha": alpha,
        "beta": beta,
        "risk_mix": risk_mix,
        "global_fail_rate": model.get("global_fail_rate", 0.0),
        "flipped": bool(model.get("flipped", False)),
    }
    return p_train, p_val, p_test, "cgr_policy", meta


def recommend_cgr_group_thresholds(
    train_rows: List[Dict[str, str]],
    elapsed_bins: int,
    target_coverage: float = 0.70,
) -> Tuple[int, int, Dict[str, float]]:
    if not train_rows:
        return 80, 15, {"coverage": 0.0}

    elapsed_vals: List[float] = []
    for r in train_rows:
        pe = maybe_float(r.get("prefix_elapsed_seconds", ""))
        if pe is not None:
            elapsed_vals.append(float(pe))
    elapsed_edges = build_quantile_edges(elapsed_vals, elapsed_bins)

    stats = [defaultdict(lambda: {"n": 0, "n_fail": 0, "n_succ": 0}) for _ in range(3)]
    for r in train_rows:
        y = target_success(r)
        for level in (0, 1, 2):
            key = cgr_group_key(r, level, elapsed_edges)
            st = stats[level][key]
            st["n"] += 1
            if y == 0:
                st["n_fail"] += 1
            else:
                st["n_succ"] += 1

    def coverage(min_total: int, min_each: int) -> float:
        ok = 0
        for r in train_rows:
            hit = False
            for level in (0, 1, 2):
                key = cgr_group_key(r, level, elapsed_edges)
                st = stats[level].get(key)
                if not st:
                    continue
                if st["n"] >= min_total and st["n_fail"] >= min_each and st["n_succ"] >= min_each:
                    hit = True
                    break
            if hit:
                ok += 1
        return ok / len(train_rows)

    candidates_total = [20, 30, 40, 60, 80, 120]
    candidates_each = [3, 5, 8, 10, 15]
    recs: List[Tuple[int, int, float]] = []
    for t in candidates_total:
        for e in candidates_each:
            if e * 2 > t:
                continue
            cov = coverage(t, e)
            recs.append((t, e, cov))

    recs.sort(key=lambda x: (x[2], x[0], x[1]), reverse=True)
    feasible = [x for x in recs if x[2] >= target_coverage]
    if feasible:
        feasible.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        t, e, cov = feasible[0]
    else:
        t, e, cov = recs[0]
    return t, e, {"coverage": float(cov), "target_coverage": float(target_coverage)}


def rule_success_prob(row: Dict[str, str]) -> float:
    pm = parse_prefix_metrics(row.get("prefix_metrics", ""))
    ws = pm.get("worst_slack")
    cong = pm.get("congestion")
    ovf = pm.get("overflow")
    dov = pm.get("density_overflow")
    cp = row.get("checkpoint", "")

    risk = 0.0
    if ws is not None:
        if ws < -0.10:
            risk += 2.5
        elif ws < -0.02:
            risk += 1.0
    else:
        risk += 0.5

    if ovf is not None:
        if ovf > 0.10:
            risk += 2.0
        elif ovf > 0.03:
            risk += 1.0
    if cong is not None:
        if cong > 0.20:
            risk += 1.5
        elif cong > 0.10:
            risk += 0.7
    if dov is not None and dov > 0.10:
        risk += 0.8

    if cp == "drt":
        risk += 0.6
    elif cp == "grt":
        risk += 0.3

    # Convert to success probability (high risk -> low success prob).
    return sigmoid(2.2 - risk)


def write_predictions_csv(
    out_csv: str,
    rows: List[Dict[str, str]],
    success_prob: List[float],
    stop_threshold: float,
    pred_stop_override: Optional[List[int]] = None,
) -> None:
    fields = [
        "prefix_id",
        "record_key",
        "checkpoint",
        "strict_success",
        "actionable_success",
        "eval_label_field",
        "eval_label_value",
        "success_prob",
        "pred_success",
        "pred_stop",
        "final_wallclock_seconds",
        "prefix_elapsed_seconds",
        "remaining_seconds",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (r, p) in enumerate(zip(rows, success_prob)):
            if pred_stop_override is not None and i < len(pred_stop_override):
                pred_stop = "1" if int(pred_stop_override[i]) == 1 else "0"
            else:
                pred_stop = "1" if p < stop_threshold else "0"
            w.writerow(
                {
                    "prefix_id": r.get("prefix_id", ""),
                    "record_key": r.get("record_key", ""),
                    "checkpoint": r.get("checkpoint", ""),
                    "strict_success": r.get("strict_success", ""),
                    "actionable_success": r.get("actionable_success", ""),
                    "eval_label_field": TARGET_LABEL_FIELD,
                    "eval_label_value": r.get(TARGET_LABEL_FIELD, ""),
                    "success_prob": f"{p:.6f}",
                    "pred_success": "1" if p >= 0.5 else "0",
                    "pred_stop": pred_stop,
                    "final_wallclock_seconds": r.get("final_wallclock_seconds", ""),
                    "prefix_elapsed_seconds": r.get("prefix_elapsed_seconds", ""),
                    "remaining_seconds": r.get("remaining_seconds", ""),
                }
            )


def _write_rows_csv(path: str, rows: List[Dict[str, Any]], preferred_fields: Optional[List[str]] = None) -> None:
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    if preferred_fields:
        for k in preferred_fields:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _stop_contribution_seconds(row: Dict[str, str], y_success: int, pred_stop: int, lambda_false_stop: float) -> float:
    remain = row_remaining_seconds(row)
    if pred_stop == 1 and y_success == 0:
        return remain
    if pred_stop == 1 and y_success == 1:
        return -lambda_false_stop * remain
    return 0.0


def export_llm_failure_analysis(
    output_dir: str,
    model_name: str,
    split_name: str,
    rows: List[Dict[str, str]],
    llm_success_prob: List[float],
    llm_pred_stop: List[int],
    llm_details: List[Dict[str, Any]],
    xgb_success_prob: List[float],
    xgb_threshold: float,
    lambda_false_stop: float,
) -> Dict[str, Any]:
    n = min(
        len(rows),
        len(llm_success_prob),
        len(llm_pred_stop),
        len(llm_details),
        len(xgb_success_prob),
    )
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    safe_split = re.sub(r"[^A-Za-z0-9_.-]+", "_", split_name)
    base = os.path.join(output_dir, f"llm_failure_{safe_model}_{safe_split}")

    case_rows: List[Dict[str, Any]] = []
    by_cp_case: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    risk_tag_counts: Dict[str, int] = defaultdict(int)
    evidence_hit_count = 0

    for i in range(n):
        r = rows[i]
        y_success = target_success(r)
        remain = row_remaining_seconds(r)
        checkpoint = r.get("checkpoint", "")
        llm_stop = 1 if int(llm_pred_stop[i]) == 1 else 0
        xgb_stop = 1 if float(xgb_success_prob[i]) < float(xgb_threshold) else 0

        if llm_stop == 1 and y_success == 1:
            llm_case = "false_stop"
        elif llm_stop == 0 and y_success == 0:
            llm_case = "missed_stop"
        elif llm_stop == 1 and y_success == 0:
            llm_case = "true_stop"
        else:
            llm_case = "true_continue"

        llm_net = _stop_contribution_seconds(r, y_success, llm_stop, lambda_false_stop=lambda_false_stop)
        xgb_net = _stop_contribution_seconds(r, y_success, xgb_stop, lambda_false_stop=lambda_false_stop)
        rel = "tie"
        if llm_net < xgb_net:
            rel = "xgb_better"
        elif llm_net > xgb_net:
            rel = "llm_better"

        det = llm_details[i] if i < len(llm_details) else {}
        tags = _normalize_str_list(det.get("risk_tags", []), max_items=10, max_len=120)
        evidence = _normalize_str_list(det.get("evidence", []), max_items=8, max_len=180)
        if evidence:
            evidence_hit_count += 1
        for t in tags:
            risk_tag_counts[t] += 1

        by_cp_case[checkpoint][llm_case] += 1

        case_rows.append(
            {
                "prefix_id": r.get("prefix_id", ""),
                "record_key": r.get("record_key", ""),
                "platform": r.get("platform", ""),
                "design": r.get("design", ""),
                "checkpoint": checkpoint,
                "objective_hint": r.get("objective_hint", ""),
                "eval_label_field": TARGET_LABEL_FIELD,
                "eval_label_value": r.get(TARGET_LABEL_FIELD, ""),
                "strict_success": r.get("strict_success", ""),
                "actionable_success": r.get("actionable_success", ""),
                "y_success": y_success,
                "remaining_seconds": remain,
                "llm_success_prob": float(llm_success_prob[i]),
                "llm_pred_stop": llm_stop,
                "llm_case": llm_case,
                "llm_net_seconds": llm_net,
                "xgb_success_prob": float(xgb_success_prob[i]),
                "xgb_pred_stop": xgb_stop,
                "xgb_net_seconds": xgb_net,
                "delta_xgb_minus_llm_seconds": (xgb_net - llm_net),
                "relative_quality": rel,
                "llm_confidence": _float_in_01(det.get("confidence", 0.5), default=0.5),
                "llm_decision_text": str(det.get("decision", "STOP" if llm_stop == 1 else "CONTINUE")),
                "llm_parse_mode": str(det.get("parse_mode", "")),
                "llm_source": str(det.get("source", "")),
                "llm_risk_tags": "|".join(tags),
                "llm_evidence": " || ".join(evidence),
            }
        )

    # Summary and grouped tables.
    case_counts: Dict[str, int] = defaultdict(int)
    for rr in case_rows:
        case_counts[str(rr.get("llm_case", ""))] += 1

    by_checkpoint_rows: List[Dict[str, Any]] = []
    for cp in sorted(by_cp_case.keys()):
        s = by_cp_case[cp]
        n_cp = sum(s.values())
        by_checkpoint_rows.append(
            {
                "checkpoint": cp,
                "n": n_cp,
                "false_stop_count": s.get("false_stop", 0),
                "missed_stop_count": s.get("missed_stop", 0),
                "true_stop_count": s.get("true_stop", 0),
                "true_continue_count": s.get("true_continue", 0),
                "false_stop_rate": (s.get("false_stop", 0) / n_cp if n_cp else 0.0),
                "missed_stop_rate": (s.get("missed_stop", 0) / n_cp if n_cp else 0.0),
            }
        )

    top_loss = sorted(case_rows, key=lambda x: float(x.get("llm_net_seconds", 0.0)))[:50]
    top_delta = sorted(case_rows, key=lambda x: float(x.get("delta_xgb_minus_llm_seconds", 0.0)), reverse=True)[:50]
    top_tags = sorted(risk_tag_counts.items(), key=lambda kv: kv[1], reverse=True)[:30]

    summary = {
        "model_name": model_name,
        "split_name": split_name,
        "n": n,
        "label_field": TARGET_LABEL_FIELD,
        "false_stop_count": int(case_counts.get("false_stop", 0)),
        "missed_stop_count": int(case_counts.get("missed_stop", 0)),
        "true_stop_count": int(case_counts.get("true_stop", 0)),
        "true_continue_count": int(case_counts.get("true_continue", 0)),
        "false_stop_rate": (case_counts.get("false_stop", 0) / n if n else 0.0),
        "missed_stop_rate": (case_counts.get("missed_stop", 0) / n if n else 0.0),
        "avg_llm_net_seconds": (sum(float(r.get("llm_net_seconds", 0.0)) for r in case_rows) / n if n else 0.0),
        "avg_xgb_net_seconds": (sum(float(r.get("xgb_net_seconds", 0.0)) for r in case_rows) / n if n else 0.0),
        "avg_delta_xgb_minus_llm_seconds": (
            sum(float(r.get("delta_xgb_minus_llm_seconds", 0.0)) for r in case_rows) / n if n else 0.0
        ),
        "xgb_better_count": int(sum(1 for r in case_rows if r.get("relative_quality") == "xgb_better")),
        "llm_better_count": int(sum(1 for r in case_rows if r.get("relative_quality") == "llm_better")),
        "evidence_nonempty_count": int(evidence_hit_count),
        "top_risk_tags": [{"tag": t, "count": c} for t, c in top_tags],
    }

    all_csv = f"{base}_all.csv"
    by_cp_csv = f"{base}_by_checkpoint.csv"
    top_loss_csv = f"{base}_top_false_stop_loss.csv"
    top_delta_csv = f"{base}_top_xgb_gain.csv"
    summary_json = f"{base}_summary.json"

    _write_rows_csv(
        all_csv,
        case_rows,
        preferred_fields=[
            "prefix_id",
            "platform",
            "design",
            "checkpoint",
            "eval_label_field",
            "eval_label_value",
            "y_success",
            "remaining_seconds",
            "llm_success_prob",
            "llm_pred_stop",
            "llm_case",
            "llm_net_seconds",
            "xgb_success_prob",
            "xgb_pred_stop",
            "xgb_net_seconds",
            "delta_xgb_minus_llm_seconds",
            "relative_quality",
            "llm_confidence",
            "llm_decision_text",
            "llm_risk_tags",
            "llm_evidence",
        ],
    )
    _write_rows_csv(by_cp_csv, by_checkpoint_rows)
    _write_rows_csv(top_loss_csv, top_loss)
    _write_rows_csv(top_delta_csv, top_delta)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "all_csv": all_csv,
        "by_checkpoint_csv": by_cp_csv,
        "top_loss_csv": top_loss_csv,
        "top_delta_csv": top_delta_csv,
        "summary_json": summary_json,
        "summary": summary,
    }


def train_supervised_model(
    train_rows: List[Dict[str, str]],
    val_rows: List[Dict[str, str]],
    test_rows: List[Dict[str, str]],
    include_missing_indicators: bool = False,
    preferred_model: str = "auto",
    feature_mode: str = "param_plus_prefix",
    seed: int = 42,
) -> Tuple[List[float], List[float], List[float], str]:
    # Supervised baseline selector:
    #   auto: xgboost -> hgb -> logreg_sgd
    #   xgboost/hgb/logreg: forced preference with graceful fallback to logreg.
    import numpy as np

    mode = str(feature_mode or "param_plus_prefix").strip().lower()
    use_prefix = mode in {"param_plus_prefix", "prefix_only"}
    use_param = mode in {"param_plus_prefix", "param_only"}

    base_numeric_keys: List[str] = []
    if use_prefix:
        base_numeric_keys.extend(list(PREFIX_METRIC_KEYS))
    if use_param:
        base_numeric_keys.extend(list(RUN_PARAM_NUMERIC_KEYS))
        base_numeric_keys.extend(list(RUN_PARAM_BOOL_KEYS))
        base_numeric_keys.extend(["pin_layer_idx", "above_layer_idx", "layer_span"])
    feature_keys = list(base_numeric_keys)
    if include_missing_indicators:
        for k in base_numeric_keys:
            feature_keys.append(f"{k}_missing")
    feature_keys.extend(["cp_place", "cp_cts", "cp_grt", "cp_drt"])

    def rows_to_matrix(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.zeros((len(rows), len(feature_keys)), dtype=float)
        y = np.zeros(len(rows), dtype=float)
        for i, r in enumerate(rows):
            feat = extract_features_with_mode(
                r,
                include_missing_indicators=include_missing_indicators,
                feature_mode=mode,
            )
            for j, k in enumerate(feature_keys):
                X[i, j] = float(feat.get(k, 0.0))
            y[i] = float(target_success(r))
        return X, y

    X_train_raw, y_train = rows_to_matrix(train_rows)
    X_val_raw, _ = rows_to_matrix(val_rows)
    X_test_raw, _ = rows_to_matrix(test_rows)

    # Tree models on raw features.
    model_pref = str(preferred_model or "auto").strip().lower()
    train_seed = int(seed)

    if model_pref in ("auto", "xgboost"):
        try:
            from xgboost import XGBClassifier  # type: ignore

            xgb = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=train_seed,
                n_jobs=8,
                tree_method="hist",
            )
            xgb.fit(X_train_raw, y_train)
            p_train = xgb.predict_proba(X_train_raw)[:, 1].tolist()
            p_val = xgb.predict_proba(X_val_raw)[:, 1].tolist()
            p_test = xgb.predict_proba(X_test_raw)[:, 1].tolist()
            return p_train, p_val, p_test, "xgboost"
        except Exception as e:
            print(f"[supervised] xgboost unavailable/failed ({e}); fallback.", flush=True)
            if model_pref == "xgboost":
                model_pref = "logreg"

    if model_pref in ("auto", "hgb"):
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore

            hgb = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=300,
                l2_regularization=1e-3,
                random_state=train_seed,
            )
            hgb.fit(X_train_raw, y_train)
            p_train = hgb.predict_proba(X_train_raw)[:, 1].tolist()
            p_val = hgb.predict_proba(X_val_raw)[:, 1].tolist()
            p_test = hgb.predict_proba(X_test_raw)[:, 1].tolist()
            return p_train, p_val, p_test, "hgb"
        except Exception as e:
            print(f"[supervised] hgb unavailable/failed ({e}); fallback to logreg.", flush=True)

    # Standardize (except one-hot checkpoint features at the tail).
    X_train = X_train_raw.copy()
    X_val = X_val_raw.copy()
    X_test = X_test_raw.copy()
    n_cont = len(feature_keys) - 4
    mean = X_train[:, :n_cont].mean(axis=0)
    std = X_train[:, :n_cont].std(axis=0)
    std[std < 1e-8] = 1.0
    X_train[:, :n_cont] = (X_train[:, :n_cont] - mean) / std
    X_val[:, :n_cont] = (X_val[:, :n_cont] - mean) / std
    X_test[:, :n_cont] = (X_test[:, :n_cont] - mean) / std

    # Mini-batch SGD
    rng = np.random.default_rng(train_seed)
    w = np.zeros(X_train.shape[1], dtype=float)
    b = 0.0
    lr = 0.05
    l2 = 1e-4
    epochs = 10
    batch = 2048

    for _ in range(epochs):
        idx = rng.permutation(X_train.shape[0])
        for st in range(0, X_train.shape[0], batch):
            ed = min(st + batch, X_train.shape[0])
            sel = idx[st:ed]
            xb = X_train[sel]
            yb = y_train[sel]

            z = xb @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            err = p - yb

            grad_w = (xb.T @ err) / len(sel) + l2 * w
            grad_b = float(err.mean())
            w -= lr * grad_w
            b -= lr * grad_b

    def predict_prob(X: np.ndarray) -> List[float]:
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        return p.tolist()

    p_train = predict_prob(X_train)
    p_val = predict_prob(X_val)
    p_test = predict_prob(X_test)
    return p_train, p_val, p_test, "logreg_sgd"


def _default_api_base(provider: str) -> str:
    p = str(provider or "deepseek").strip().lower()
    if p == "openai":
        return "https://api.openai.com/"
    if p == "anthropic":
        return "https://api.anthropic.com/v1/messages"
    return "https://api.deepseek.com/chat/completions"


def load_llm_api_key(provider: str, driver_or_optimize_path: str) -> str:
    # Priority 1: provider-specific env.
    p = str(provider or "deepseek").strip().lower()
    env_names = {
        "deepseek": ["DEEPSEEK_API_KEY", "LLM_API_KEY"],
        "openai": ["OPENAI_API_KEY", "LLM_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY", "LLM_API_KEY"],
    }.get(p, ["LLM_API_KEY"])
    for name in env_names:
        env_key = os.environ.get(name, "").strip()
        if env_key:
            return env_key

    # Priority 2: parse file for simple hardcoded patterns.
    if not os.path.isfile(driver_or_optimize_path):
        return ""
    try:
        with open(driver_or_optimize_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return ""

    pats = [
        re.compile(r"DEEPSEEK_API_KEY\s*=\s*[\"']([^\"']+)[\"']"),
        re.compile(r"OPENAI_API_KEY\s*=\s*[\"']([^\"']+)[\"']"),
        re.compile(r"ANTHROPIC_API_KEY\s*=\s*[\"']([^\"']+)[\"']"),
        re.compile(r"LLM_API_KEY\s*=\s*[\"']([^\"']+)[\"']"),
        re.compile(r"api_key\s*=\s*[\"']([^\"']+)[\"']"),
    ]
    for pat in pats:
        m = pat.search(txt)
        if m:
            v = m.group(1).strip()
            if v and "PUT YOUR KEY" not in v and "YOUR_KEY" not in v:
                return v
    return ""


def log_excerpt_for_llm(path: str, max_chars: int = 1800) -> str:
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return ""

    keep = []
    pat = re.compile(r"(WARN|ERROR|FATAL|violation|overflow|congestion|slack|WNS|TNS|DRC)", re.IGNORECASE)
    for ln in lines:
        if pat.search(ln):
            keep.append(ln.strip())
    if not keep:
        head = "".join(lines[:20])
        tail = "".join(lines[-20:])
        text = (head + "\n...\n" + tail).strip()
        return text[:max_chars]

    text = "\n".join(keep[-80:])
    return text[:max_chars]



def log_excerpt_for_llm_enhanced(path: str, max_chars: int = 2800) -> str:
    if not path or not os.path.isfile(path):
        return ""
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
        if not ln:
            continue
        if ln in seen:
            continue
        seen.add(ln)
        merged.append(ln)

    if not merged:
        head = "".join(lines[:20])
        tail_txt = "".join(lines[-40:])
        return (head + "\n...\n" + tail_txt).strip()[:max_chars]
    return "\n".join(merged)[:max_chars]


def extract_log_signals(text: str) -> Dict[str, float]:
    s = text or ""

    def c(pat: str) -> int:
        return len(re.findall(pat, s, flags=re.IGNORECASE))

    sig = {
        "warn_count": float(c(r"\bWARN(?:ING)?\b")),
        "error_count": float(c(r"\bERROR\b|\bError:\b")),
        "fatal_count": float(c(r"\bFATAL\b")),
        "drc_mentions": float(c(r"\bDRC\b|\bviolations?\b")),
        "overflow_mentions": float(c(r"\boverflow\b")),
        "timing_mentions": float(c(r"\bWNS\b|\bTNS\b|slack")),
        "congestion_mentions": float(c(r"\bcongestion\b")),
    }
    sig["hard_fail_flag"] = 1.0 if (sig["fatal_count"] > 0 or sig["error_count"] > 0) else 0.0
    return sig


def compact_metrics_for_llm(metrics_json: str) -> Dict[str, Optional[float]]:
    pm = parse_prefix_metrics(metrics_json)
    keys = [
        "dpl_hpwl",
        "congestion",
        "overflow",
        "density_overflow",
        "worst_slack",
        "tns_metric",
        "rpt_wns",
        "rpt_tns",
        "rpt_worst_slack",
        "rpt_setup_skew",
        "rpt_has_violated_path",
        "rpt_exists",
        "rpt_nonempty",
        "design_area",
        "utilization",
        "prefix_elapsed_seconds",
    ]
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        v = pm.get(k)
        if v is None:
            out[k] = None
        else:
            out[k] = float(round(v, 6))
    return out


def compact_run_params_for_llm(row: Dict[str, str]) -> Dict[str, str]:
    keys = list(RUN_PARAM_NUMERIC_KEYS) + list(RUN_PARAM_BOOL_KEYS) + [
        "pin_layer",
        "above_layer",
    ]
    out: Dict[str, str] = {}
    for k in keys:
        v = (row.get(k, "") or "").strip()
        if v != "":
            out[k] = v
    return out


def compact_run_params_with_missing_for_llm(row: Dict[str, str]) -> Dict[str, Optional[str]]:
    base = compact_run_params_for_llm(row)
    out: Dict[str, Optional[str]] = {}
    for k in (list(RUN_PARAM_NUMERIC_KEYS) + list(RUN_PARAM_BOOL_KEYS) + ["pin_layer", "above_layer"]):
        v = base.get(k)
        if v is None or str(v).strip() == "":
            out[k] = None
            out[f"{k}_missing"] = "1"
        else:
            out[k] = str(v)
            out[f"{k}_missing"] = "0"
    return out


def format_run_params_text_for_model_input(row: Dict[str, str]) -> str:
    run_params = compact_run_params_with_missing_for_llm(row)
    return json.dumps(run_params, ensure_ascii=False, sort_keys=True)


def build_llm_prompt(
    row: Dict[str, str],
    prompt_style: str,
    log_style: str,
    stage1_json: str = "",
) -> str:
    cp = row.get("checkpoint", "")
    metrics_raw = row.get("prefix_metrics", "{}")
    metrics_small = compact_metrics_for_llm(metrics_raw)
    platform = row.get("platform", "")
    design = row.get("design", "")
    objective = row.get("objective_hint", "")
    psec = row.get("prefix_elapsed_seconds", "")
    run_params = compact_run_params_with_missing_for_llm(row)

    if log_style == "enhanced":
        log_excerpt = log_excerpt_for_llm_enhanced(row.get("prefix_log", ""))
    else:
        log_excerpt = log_excerpt_for_llm(row.get("prefix_log", ""))
    sig = extract_log_signals(log_excerpt)

    if prompt_style == "weak":
        return (
            "You are an EDA triage model.\n"
            "Task: Decide whether to STOP or CONTINUE for this EDA checkpoint.\n"
            "Goal: Maximize net runtime savings while controlling false stops on successful runs.\n"
            "Decision policy: Treat STOP and CONTINUE as equally valid baseline decisions.\n"
            "Return strict JSON: {\"decision\":\"STOP|CONTINUE\",\"confidence\":0.0-1.0,"
            "\"success_probability\":0.0-1.0}.\n\n"
            f"Checkpoint: {cp}\n"
            f"Run Params JSON: {json.dumps(run_params, ensure_ascii=False)}\n"
            f"Metrics JSON: {metrics_raw}\n"
            f"Log Excerpt:\n{log_excerpt}\n"
        )

    if prompt_style == "strong":
        stage1_block = ""
        if stage1_json:
            stage1_block = f"Stage-1 Risk Extraction JSON:\n{stage1_json}\n\n"
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
            f"Checkpoint: {cp}\n"
            f"Prefix elapsed seconds: {psec}\n"
            f"Metrics Compact JSON: {json.dumps(metrics_small, ensure_ascii=False)}\n"
            f"Run Params JSON: {json.dumps(run_params, ensure_ascii=False)}\n"
            f"Log Signals JSON: {json.dumps(sig, ensure_ascii=False)}\n"
            f"{stage1_block}"
            f"Log Excerpt:\n{log_excerpt}\n"
        )

    if prompt_style == "expert":
        stage1_block = ""
        if stage1_json:
            stage1_block = f"Stage-1 Risk Extraction JSON:\n{stage1_json}\n\n"
        return (
            "You are an EDA triage expert for cost-aware early stopping.\n"
            "Goal: maximize net saved runtime with evidence-grounded STOP/CONTINUE decisions.\n"
            "Use the rule set below exactly; do not be over-conservative.\n\n"
            "Rule set (priority order):\n"
            "R1 Hard-fail rule: if logs show ERROR/FATAL/segmentation/killed/timeout => decision must be STOP.\n"
            "R2 Routing-collapse rule: if checkpoint in {grt,drt} and (overflow > 0.03 OR congestion > 0.10 OR explicit DRC/violation mentions) => prefer STOP unless strong recovery evidence exists.\n"
            "R3 Timing-collapse rule: if worst_slack < -0.10 and congestion/overflow is non-trivial => prefer STOP.\n"
            "R4 If none of R1-R3 hold, CONTINUE.\n"
            "Then output calibrated success_probability and confidence.\n\n"
            "Output STRICT JSON only:\n"
            "{\"decision\":\"STOP|CONTINUE\",\"confidence\":0.0-1.0,\"success_probability\":0.0-1.0,"
            "\"risk_tags\":[\"...\"],\"evidence\":[\"...\"]}\n"
            "Use concise risk_tags and include 1-3 evidence entries quoted from log text.\n\n"
            f"Platform: {platform}\n"
            f"Design: {design}\n"
            f"Objective: {objective}\n"
            f"Checkpoint: {cp}\n"
            f"Prefix elapsed seconds: {psec}\n"
            f"Metrics Compact JSON: {json.dumps(metrics_small, ensure_ascii=False)}\n"
            f"Run Params JSON: {json.dumps(run_params, ensure_ascii=False)}\n"
            f"Log Signals JSON: {json.dumps(sig, ensure_ascii=False)}\n"
            f"{stage1_block}"
            f"Log Excerpt:\n{log_excerpt}\n"
        )

    return (
        "Task: Decide whether to STOP_AND_REALLOCATE or CONTINUE for this EDA run checkpoint.\n"
        "Return strict JSON: {\"decision\":\"STOP|CONTINUE\",\"confidence\":0.0-1.0}.\n\n"
        f"Checkpoint: {cp}\n"
        f"Run Params JSON: {json.dumps(run_params, ensure_ascii=False)}\n"
        f"Metrics JSON: {metrics_raw}\n"
        f"Log Excerpt:\n{log_excerpt}\n"
    )


def build_rationale_retry_prompt(base_prompt: str, min_risk_tags: int, min_evidence: int) -> str:
    return (
        f"{base_prompt}\n\n"
        "Format correction (MANDATORY): return strict JSON only, with non-empty rationale arrays.\n"
        f"Constraints: risk_tags length >= {max(0, int(min_risk_tags))}, "
        f"evidence length >= {max(0, int(min_evidence))}.\n"
        "Each evidence item must be a concise, log-grounded phrase.\n"
    )


def run_llm_stage1_extraction(
    api_key: str,
    model_name: str,
    provider: str,
    api_base: str,
    openai_wire_api: str,
    openai_reasoning_effort: str,
    anthropic_version: str,
    row: Dict[str, str],
    log_style: str,
) -> str:
    cp = row.get("checkpoint", "")
    metrics_small = compact_metrics_for_llm(row.get("prefix_metrics", "{}"))
    run_params = compact_run_params_with_missing_for_llm(row)
    if log_style == "enhanced":
        log_excerpt = log_excerpt_for_llm_enhanced(row.get("prefix_log", ""))
    else:
        log_excerpt = log_excerpt_for_llm(row.get("prefix_log", ""))
    sig = extract_log_signals(log_excerpt)

    prompt = (
        "Extract structured risk information for this EDA checkpoint.\n"
        "Return STRICT JSON only with keys:\n"
        "{\"risk_score\":0.0-1.0,\"timing_risk\":0.0-1.0,\"routing_risk\":0.0-1.0,"
        "\"hard_fail_signals\":[\"...\"],\"summary\":\"...\"}\n\n"
        f"Checkpoint: {cp}\n"
        f"Metrics Compact JSON: {json.dumps(metrics_small, ensure_ascii=False)}\n"
        f"Run Params JSON: {json.dumps(run_params, ensure_ascii=False)}\n"
        f"Log Signals JSON: {json.dumps(sig, ensure_ascii=False)}\n"
        f"Log Excerpt:\n{log_excerpt}\n"
    )
    return llm_chat_completion(
        api_key=api_key,
        prompt=prompt,
        model=model_name,
        provider=provider,
        api_base=api_base,
        openai_wire_api=openai_wire_api,
        openai_reasoning_effort=openai_reasoning_effort,
        anthropic_version=anthropic_version,
    )


def llm_chat_completion(
    api_key: str,
    prompt: str,
    model: str,
    provider: str = "deepseek",
    api_base: str = "",
    openai_wire_api: str = "chat_completions",
    openai_reasoning_effort: str = "",
    anthropic_version: str = "2023-06-01",
    timeout_s: int = 60,
) -> str:
    p = str(provider or "deepseek").strip().lower()
    base = (str(api_base or "").strip() or _default_api_base(p)).strip()
    wire_api = str(openai_wire_api or "chat_completions").strip().lower()
    if wire_api in ("chat/completions", "chat-completions", "chat_completions", "chat"):
        wire_api = "chat_completions"
    elif wire_api in ("responses", "response"):
        wire_api = "responses"
    else:
        wire_api = "chat_completions"

    def _join_base(path: str) -> str:
        return f"{base.rstrip('/')}/{path.lstrip('/')}"

    endpoint = base
    if p in ("deepseek", "openai"):
        if p == "openai" and wire_api == "responses":
            endpoint = _join_base("/v1/responses")
            payload: Dict[str, Any] = {
                "model": model,
                "temperature": 0.0,
                "input": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": "You are an EDA triage assistant. Reply with strict JSON only."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt}
                        ],
                    },
                ],
            }
            if str(openai_reasoning_effort or "").strip():
                payload["reasoning"] = {"effort": str(openai_reasoning_effort).strip()}
        else:
            endpoint = _join_base("/v1/chat/completions")
            payload = {
                "model": model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": "You are an EDA triage assistant. Reply with strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
            }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="ignore"))
        if p == "openai" and wire_api == "responses":
            out_txt = obj.get("output_text")
            if isinstance(out_txt, str) and out_txt.strip():
                return out_txt
            out = obj.get("output")
            if isinstance(out, list):
                buf: List[str] = []
                for item in out:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content")
                    if not isinstance(content, list):
                        continue
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") in ("output_text", "text"):
                            t = blk.get("text")
                            if isinstance(t, str) and t:
                                buf.append(t)
                if buf:
                    return "\n".join(buf)
            raise KeyError(f"invalid_responses_output_text:{str(obj)[:500]}")
        else:
            choices = obj.get("choices")
            if not isinstance(choices, list) or not choices:
                raise KeyError(f"invalid_chat_response_missing_choices:{str(obj)[:500]}")
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = msg.get("content") if isinstance(msg, dict) else None
            if content is None:
                raise KeyError(f"invalid_chat_response_missing_content:{str(obj)[:500]}")
            return str(content)

    if p == "anthropic":
        payload = {
            "model": model,
            "max_tokens": 512,
            "temperature": 0.0,
            "system": "You are an EDA triage assistant. Reply with strict JSON only.",
            "messages": [{"role": "user", "content": prompt}],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": str(anthropic_version or "2023-06-01"),
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="ignore"))
        content = obj.get("content", [])
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    return str(blk.get("text", ""))
        raise KeyError("anthropic content.text not found")

    raise ValueError(f"Unsupported llm provider: {provider}")


def llm_chat_completion_with_retry(
    api_key: str,
    prompt: str,
    model: str,
    provider: str = "deepseek",
    api_base: str = "",
    openai_wire_api: str = "chat_completions",
    openai_reasoning_effort: str = "",
    anthropic_version: str = "2023-06-01",
    timeout_s: int = 60,
    max_retries: int = 4,
    retry_base_sleep_s: float = 0.8,
) -> str:
    # Retry transient API/network anomalies while preserving identical prompt/model settings.
    for attempt in range(max(1, int(max_retries))):
        try:
            return llm_chat_completion(
                api_key=api_key,
                prompt=prompt,
                model=model,
                provider=provider,
                api_base=api_base,
                openai_wire_api=openai_wire_api,
                openai_reasoning_effort=openai_reasoning_effort,
                anthropic_version=anthropic_version,
                timeout_s=timeout_s,
            )
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            ValueError,
            http.client.HTTPException,
            OSError,
        ):
            if attempt + 1 >= max(1, int(max_retries)):
                raise
            time.sleep(float(retry_base_sleep_s) * (2 ** attempt))
    raise RuntimeError("unreachable")


def _json_candidates_from_text(raw_content: str) -> List[Tuple[str, str]]:
    text = (raw_content or "").strip()
    if not text:
        return []
    candidates: List[Tuple[str, str]] = [("raw", text)]

    # Common LLM output shape: ```json ... ```
    if text.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text, count=1)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
        stripped = stripped.strip()
        if stripped:
            candidates.append(("fenced", stripped))

    # Last-resort: extract the first JSON object region.
    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        obj_slice = text[i : j + 1].strip()
        if obj_slice:
            candidates.append(("slice", obj_slice))

    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: List[Tuple[str, str]] = []
    for mode, cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        deduped.append((mode, cand))
    return deduped


def _try_parse_json_object(raw_content: str) -> Tuple[Optional[Dict[str, Any]], str]:
    for mode, cand in _json_candidates_from_text(raw_content):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj, f"json_{mode}"
        except Exception:
            continue
    return None, ""


def _float_in_01(v: Any, default: float = 0.5) -> float:
    try:
        x = float(v)
    except Exception:
        x = float(default)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _normalize_str_list(v: Any, max_items: int = 8, max_len: int = 200) -> List[str]:
    out: List[str] = []
    if isinstance(v, list):
        raw_items = v
    elif isinstance(v, str):
        raw_items = [x.strip() for x in re.split(r"[;\n|]+", v) if x.strip()]
    else:
        raw_items = []
    for x in raw_items:
        s = str(x).strip()
        if not s:
            continue
        if len(s) > max_len:
            s = s[:max_len]
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def parse_llm_payload(content: str) -> Tuple[int, float, Dict[str, Any]]:
    # Returns: (pred_stop, success_prob, details)
    # details contains parseable rationale fields for failure analysis:
    # decision/confidence/risk_tags/evidence/raw_content/parse_mode.
    raw_content = (content or "").strip()
    decision = ""
    conf = 0.5
    success_prob: Optional[float] = None
    risk_tags: List[str] = []
    evidence: List[str] = []
    parse_mode = "regex_fallback"

    obj, json_mode = _try_parse_json_object(raw_content)
    if isinstance(obj, dict):
        parse_mode = json_mode or "json"
        decision = str(obj.get("decision", "")).strip().upper()
        if "success_probability" in obj:
            success_prob = _float_in_01(obj.get("success_probability"), default=0.5)
        elif "success_prob" in obj:
            success_prob = _float_in_01(obj.get("success_prob"), default=0.5)
        elif "risk_score" in obj:
            success_prob = 1.0 - _float_in_01(obj.get("risk_score"), default=0.5)
        else:
            conf = _float_in_01(obj.get("confidence", 0.5), default=0.5)

        risk_tags = _normalize_str_list(
            obj.get("risk_tags", obj.get("reason_codes", obj.get("reasons", [])))
        )
        evidence = _normalize_str_list(
            obj.get("evidence", obj.get("evidence_spans", obj.get("cited_log_spans", [])))
        )
    else:
        parse_mode = "regex_fallback"

    if not str(parse_mode).startswith("json"):
        m = re.search(r"(STOP|CONTINUE)", raw_content, flags=re.IGNORECASE)
        if m:
            decision = m.group(1).upper()
        m = re.search(r"success[_ ]?prob(?:ability)?[^0-9]*([0-9]*\.?[0-9]+)", raw_content, flags=re.IGNORECASE)
        if m:
            try:
                success_prob = _float_in_01(float(m.group(1)), default=0.5)
            except Exception:
                success_prob = None
        if success_prob is None:
            m = re.search(r"confidence[^0-9]*([0-9]*\.?[0-9]+)", raw_content, flags=re.IGNORECASE)
            if m:
                try:
                    conf = _float_in_01(float(m.group(1)), default=0.5)
                except Exception:
                    conf = 0.5

    if decision not in ("STOP", "CONTINUE"):
        if success_prob is not None:
            decision = "CONTINUE" if float(success_prob) >= 0.5 else "STOP"
        else:
            decision = "CONTINUE"

    if success_prob is None:
        conf = _float_in_01(conf, default=0.5)
        pred_stop = 1 if decision == "STOP" else 0
        success_prob = conf if pred_stop == 0 else (1.0 - conf)
    else:
        success_prob = _float_in_01(success_prob, default=0.5)
        pred_stop = 1 if decision == "STOP" else 0
        # Backfill confidence if model gave only success probability.
        conf = success_prob if pred_stop == 0 else (1.0 - success_prob)

    details: Dict[str, Any] = {
        "decision": decision,
        "confidence": float(conf),
        "risk_tags": risk_tags,
        "evidence": evidence,
        "parse_mode": parse_mode,
        "raw_content": raw_content,
    }
    return pred_stop, float(success_prob), details


def llm_has_min_rationale(details: Dict[str, Any], min_risk_tags: int, min_evidence: int) -> bool:
    tags = _normalize_str_list(details.get("risk_tags", []))
    evidence = _normalize_str_list(details.get("evidence", []))
    return len(tags) >= max(0, int(min_risk_tags)) and len(evidence) >= max(0, int(min_evidence))


def patch_llm_rationale(
    details: Dict[str, Any],
    row: Dict[str, str],
    min_risk_tags: int,
    min_evidence: int,
) -> Dict[str, Any]:
    tags = _normalize_str_list(details.get("risk_tags", []))
    evidence = _normalize_str_list(details.get("evidence", []))
    while len(tags) < max(0, int(min_risk_tags)):
        tags.append("unspecified_risk")
    cp = str(row.get("checkpoint", ""))
    log_name = os.path.basename(str(row.get("prefix_log", "")))
    while len(evidence) < max(0, int(min_evidence)):
        evidence.append(f"fallback_evidence:{cp}:{log_name}")
    details["risk_tags"] = tags
    details["evidence"] = evidence
    pm = str(details.get("parse_mode", ""))
    details["parse_mode"] = f"{pm}|rationale_patched" if pm else "rationale_patched"
    return details


def parse_llm_decision(content: str) -> Tuple[int, float]:
    pred_stop, success_prob, _ = parse_llm_payload(content)
    return pred_stop, success_prob


def sample_rows_deterministic(rows: List[Dict[str, str]], max_samples: int, seed: int) -> List[Dict[str, str]]:
    sample_rows = rows[:]
    random.Random(seed).shuffle(sample_rows)
    if max_samples > 0:
        sample_rows = sample_rows[:max_samples]
    return sample_rows


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    if not path:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_llm_baseline(
    rows: List[Dict[str, str]],
    api_key: str,
    model_name: str,
    provider: str,
    api_base: str,
    openai_wire_api: str,
    openai_reasoning_effort: str,
    anthropic_version: str,
    max_samples: int,
    seed: int,
    cache_jsonl: str,
    sleep_s: float = 0.0,
    prompt_style: str = "basic",
    log_style: str = "basic",
    two_stage: bool = False,
    use_cache: bool = True,
    refresh_missing_rationale: bool = False,
    require_rationale: bool = False,
    min_risk_tags: int = 1,
    min_evidence: int = 1,
    rationale_retry: int = 1,
    api_max_retries: int = 4,
    api_retry_base_sleep_s: float = 0.8,
    strict_api_errors: bool = False,
    request_status_jsonl: str = "",
    split_name: str = "",
) -> Tuple[List[int], List[float], int, List[Dict[str, str]], List[Dict[str, Any]]]:
    ensure_dir(os.path.dirname(cache_jsonl))
    cache: Dict[str, Dict[str, Any]] = {}
    if use_cache and os.path.isfile(cache_jsonl):
        with open(cache_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sig = f"{obj.get('prefix_id','')}|{obj.get('cfg','default')}"
                    cache[sig] = obj
                except Exception:
                    continue

    sample_rows = sample_rows_deterministic(rows, max_samples=max_samples, seed=seed)

    pred_stop_all: List[int] = []
    prob_all: List[float] = []
    details_all: List[Dict[str, Any]] = []
    hit_cache = 0
    strict_failures: List[Tuple[str, str]] = []

    with open(cache_jsonl, "a", encoding="utf-8") as fw:
        for i, row in enumerate(sample_rows, start=1):
            pid = row.get("prefix_id", "")
            append_jsonl(
                request_status_jsonl,
                {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "event": "row_begin",
                    "split": split_name,
                    "index": i,
                    "total": len(sample_rows),
                    "prefix_id": pid,
                },
            )
            cfg_sig = (
                f"{prompt_style}|{log_style}|{1 if two_stage else 0}"
                f"|rr={1 if require_rationale else 0}|rt={int(min_risk_tags)}|ev={int(min_evidence)}"
            )
            cache_sig = f"{pid}|{cfg_sig}"
            if cache_sig in cache:
                c = cache[cache_sig]
                cached_tags = _normalize_str_list(c.get("risk_tags", []))
                cached_evidence = _normalize_str_list(c.get("evidence", []))
                cached_details: Dict[str, Any] = {
                    "risk_tags": cached_tags,
                    "evidence": cached_evidence,
                }
                refresh_for_missing = refresh_missing_rationale and (not cached_tags and not cached_evidence)
                refresh_for_required = require_rationale and (
                    not llm_has_min_rationale(cached_details, min_risk_tags=min_risk_tags, min_evidence=min_evidence)
                )
                if refresh_for_missing or refresh_for_required:
                    # Refresh stale cache entries that lack rationale fields.
                    pass
                else:
                    pred_stop = int(c.get("pred_stop", 0))
                    success_prob = float(c.get("success_prob", 0.5))
                    pred_stop_all.append(pred_stop)
                    prob_all.append(success_prob)
                    details_all.append(
                        {
                            "prefix_id": pid,
                            "decision": ("STOP" if pred_stop == 1 else "CONTINUE"),
                            "confidence": float(c.get("confidence", (1.0 - success_prob if pred_stop == 1 else success_prob))),
                            "risk_tags": cached_tags,
                            "evidence": cached_evidence,
                            "parse_mode": str(c.get("parse_mode", "cache")),
                            "source": "cache",
                            "raw_content": str(c.get("raw_content", "")),
                        }
                    )
                    hit_cache += 1
                    append_jsonl(
                        request_status_jsonl,
                        {
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "event": "cache_hit_used",
                            "split": split_name,
                            "index": i,
                            "total": len(sample_rows),
                            "prefix_id": pid,
                            "parse_mode": str(c.get("parse_mode", "cache")),
                            "decision": ("STOP" if pred_stop == 1 else "CONTINUE"),
                            "confidence": float(c.get("confidence", (1.0 - success_prob if pred_stop == 1 else success_prob))),
                        },
                    )
                    continue

            stage1_text = ""
            if two_stage:
                try:
                    stage1_text = run_llm_stage1_extraction(
                        row=row,
                        api_key=api_key,
                        model_name=model_name,
                        provider=provider,
                        api_base=api_base,
                        openai_wire_api=openai_wire_api,
                        openai_reasoning_effort=openai_reasoning_effort,
                        anthropic_version=anthropic_version,
                        log_style=log_style,
                    )
                except (
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    TimeoutError,
                    KeyError,
                    ValueError,
                    http.client.HTTPException,
                    OSError,
                ):
                    stage1_text = ""

            prompt = build_llm_prompt(
                row=row,
                prompt_style=prompt_style,
                log_style=log_style,
                stage1_json=stage1_text,
            )

            try:
                append_jsonl(
                    request_status_jsonl,
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "event": "request_start",
                        "split": split_name,
                        "index": i,
                        "total": len(sample_rows),
                        "prefix_id": pid,
                        "provider": provider,
                        "model": model_name,
                    },
                )
                content = llm_chat_completion_with_retry(
                    api_key=api_key,
                    prompt=prompt,
                    model=model_name,
                    provider=provider,
                    api_base=api_base,
                    openai_wire_api=openai_wire_api,
                    openai_reasoning_effort=openai_reasoning_effort,
                    anthropic_version=anthropic_version,
                    max_retries=api_max_retries,
                    retry_base_sleep_s=api_retry_base_sleep_s,
                )
                pred_stop, success_prob, det = parse_llm_payload(content)
                append_jsonl(
                    request_status_jsonl,
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "event": "request_ok",
                        "split": split_name,
                        "index": i,
                        "total": len(sample_rows),
                        "prefix_id": pid,
                        "parse_mode": str(det.get("parse_mode", "unknown")),
                        "decision": str(det.get("decision", "STOP" if pred_stop == 1 else "CONTINUE")),
                        "confidence": _float_in_01(det.get("confidence", (1.0 - success_prob if pred_stop == 1 else success_prob))),
                    },
                )
                if require_rationale and (
                    not llm_has_min_rationale(det, min_risk_tags=min_risk_tags, min_evidence=min_evidence)
                ):
                    retry_prompt = build_rationale_retry_prompt(
                        prompt,
                        min_risk_tags=min_risk_tags,
                        min_evidence=min_evidence,
                    )
                    for _ in range(max(0, int(rationale_retry))):
                        try:
                            content_retry = llm_chat_completion_with_retry(
                                api_key=api_key,
                                prompt=retry_prompt,
                                model=model_name,
                                provider=provider,
                                api_base=api_base,
                                openai_wire_api=openai_wire_api,
                                openai_reasoning_effort=openai_reasoning_effort,
                                anthropic_version=anthropic_version,
                                max_retries=api_max_retries,
                                retry_base_sleep_s=api_retry_base_sleep_s,
                            )
                            pred2, prob2, det2 = parse_llm_payload(content_retry)
                            if llm_has_min_rationale(
                                det2,
                                min_risk_tags=min_risk_tags,
                                min_evidence=min_evidence,
                            ):
                                pred_stop, success_prob, det = pred2, prob2, det2
                                break
                        except (
                            urllib.error.HTTPError,
                            urllib.error.URLError,
                            TimeoutError,
                            KeyError,
                            ValueError,
                            http.client.HTTPException,
                            OSError,
                        ):
                            continue
                    if not llm_has_min_rationale(
                        det,
                        min_risk_tags=min_risk_tags,
                        min_evidence=min_evidence,
                    ):
                        det = patch_llm_rationale(
                            det,
                            row=row,
                            min_risk_tags=min_risk_tags,
                            min_evidence=min_evidence,
                        )
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                ValueError,
                http.client.HTTPException,
                OSError,
            ) as e:
                append_jsonl(
                    request_status_jsonl,
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "event": "request_error",
                        "split": split_name,
                        "index": i,
                        "total": len(sample_rows),
                        "prefix_id": pid,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "strict_api_errors": bool(strict_api_errors),
                    },
                )
                if strict_api_errors:
                    # In strict mode, keep processing remaining rows so successful calls
                    # are cached, then fail once at the end with a concise summary.
                    strict_failures.append((pid, str(e)))
                    continue
                pred_stop, success_prob = 0, 0.5
                det = {
                    "decision": "CONTINUE",
                    "confidence": 0.5,
                    "risk_tags": [],
                    "evidence": [],
                    "parse_mode": "api_error_default",
                    "source": "live",
                    "raw_content": "",
                }
                if require_rationale:
                    det = patch_llm_rationale(
                        det,
                        row=row,
                        min_risk_tags=min_risk_tags,
                        min_evidence=min_evidence,
                    )

            pred_stop_all.append(pred_stop)
            prob_all.append(success_prob)
            details_all.append(
                {
                    "prefix_id": pid,
                    "decision": str(det.get("decision", "STOP" if pred_stop == 1 else "CONTINUE")),
                    "confidence": _float_in_01(det.get("confidence", (1.0 - success_prob if pred_stop == 1 else success_prob))),
                    "risk_tags": _normalize_str_list(det.get("risk_tags", [])),
                    "evidence": _normalize_str_list(det.get("evidence", [])),
                    "parse_mode": str(det.get("parse_mode", "unknown")),
                    "source": "live",
                    "raw_content": str(det.get("raw_content", "")),
                }
            )

            rec = {
                "prefix_id": pid,
                "cfg": cfg_sig,
                "pred_stop": pred_stop,
                "success_prob": success_prob,
                "decision": str(det.get("decision", "STOP" if pred_stop == 1 else "CONTINUE")),
                "confidence": _float_in_01(det.get("confidence", (1.0 - success_prob if pred_stop == 1 else success_prob))),
                "risk_tags": _normalize_str_list(det.get("risk_tags", [])),
                "evidence": _normalize_str_list(det.get("evidence", [])),
                "parse_mode": str(det.get("parse_mode", "unknown")),
                # Truncate raw content to keep cache compact while preserving forensic evidence.
                "raw_content": str(det.get("raw_content", ""))[:4000],
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fw.flush()

            if sleep_s > 0:
                time.sleep(sleep_s)

            if i % 50 == 0:
                print(f"[llm] processed={i}/{len(sample_rows)} cache_hits={hit_cache}", flush=True)

    if strict_api_errors and strict_failures:
        append_jsonl(
            request_status_jsonl,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "event": "strict_fail_summary",
                "split": split_name,
                "failed_count": len(strict_failures),
                "sampled": len(sample_rows),
                "first_failures": strict_failures[:10],
            },
        )
        preview = "; ".join([f"{pid}: {reason}" for pid, reason in strict_failures[:3]])
        raise RuntimeError(
            f"llm_api_failed_count={len(strict_failures)} sampled={len(sample_rows)} "
            f"first_failures={preview}"
        )

    return pred_stop_all, prob_all, len(sample_rows), sample_rows, details_all


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Track-1 baselines.")
    ap.add_argument("--track1-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-field", default="strict_success")
    ap.add_argument("--threshold-metric", default="avg_weighted_net_saved_seconds_with_inference")
    ap.add_argument("--max-success-false-stop-rate", type=float, default=0.10)
    ap.add_argument("--lambda-false-stop", type=float, default=1.0)
    ap.add_argument("--export-threshold-frontier", action="store_true")
    ap.add_argument("--llm-include-inference-cost", action="store_true")
    ap.add_argument("--llm-inference-cost-per-row-s", type=float, default=-1.0)
    ap.add_argument("--disable-threshold-constraint", action="store_true")
    ap.add_argument("--include-missing-indicators", action="store_true")
    ap.add_argument("--supervised-model", choices=["auto", "xgboost", "hgb", "logreg"], default="auto")
    ap.add_argument(
        "--feature-mode",
        choices=["param_plus_prefix", "prefix_only", "param_only"],
        default="param_plus_prefix",
        help="Supervised feature subset mode.",
    )
    ap.add_argument("--disable-cgr", action="store_true")
    ap.add_argument("--cgr-elapsed-bins", type=int, default=4)
    ap.add_argument("--cgr-min-group-total", type=int, default=80)
    ap.add_argument("--cgr-min-group-each", type=int, default=15)
    ap.add_argument("--cgr-lambda-cost", type=float, default=1.0)
    ap.add_argument("--cgr-eta-uncertainty", type=float, default=0.25)
    ap.add_argument("--cgr-alpha", type=float, default=1.0)
    ap.add_argument("--cgr-beta", type=float, default=1.0)
    ap.add_argument("--cgr-risk-mix", type=float, default=0.5)
    ap.add_argument("--cgr-auto-group", action="store_true")
    ap.add_argument("--cgr-target-coverage", type=float, default=0.70)
    ap.add_argument("--run-llm", action="store_true")
    ap.add_argument("--llm-api-key", default="")
    ap.add_argument("--llm-provider", choices=["deepseek", "openai", "anthropic"], default="deepseek")
    ap.add_argument("--llm-api-base", default="")
    ap.add_argument("--llm-openai-wire-api", choices=["chat_completions", "responses"], default="chat_completions")
    ap.add_argument("--llm-openai-reasoning-effort", default="")
    ap.add_argument("--llm-anthropic-version", default="2023-06-01")
    ap.add_argument("--llm-driver-path", default="")
    ap.add_argument("--llm-model", default="deepseek-chat")
    ap.add_argument("--llm-max-samples", type=int, default=500)
    ap.add_argument("--llm-fair-compare", action="store_true")
    ap.add_argument("--llm-fair-val-ratio", type=float, default=0.4)
    ap.add_argument("--llm-seed", type=int, default=42)
    ap.add_argument("--llm-sleep-s", type=float, default=0.0)
    ap.add_argument("--llm-decision-mode", choices=["threshold", "direct"], default="threshold")
    ap.add_argument("--llm-calibration", choices=["none", "discrete"], default="none")
    ap.add_argument("--llm-calibration-prior", type=float, default=2.0)
    ap.add_argument("--llm-prompt-style", choices=["basic", "weak", "strong", "expert"], default="basic")
    ap.add_argument("--llm-log-style", choices=["basic", "enhanced"], default="basic")
    ap.add_argument("--llm-two-stage", action="store_true")
    ap.add_argument("--llm-ignore-cache", action="store_true")
    ap.add_argument("--llm-refresh-missing-rationale", action="store_true")
    ap.add_argument("--llm-require-rationale", action="store_true")
    ap.add_argument("--llm-min-risk-tags", type=int, default=1)
    ap.add_argument("--llm-min-evidence", type=int, default=1)
    ap.add_argument("--llm-rationale-retry", type=int, default=1)
    ap.add_argument("--llm-api-max-retries", type=int, default=4)
    ap.add_argument("--llm-api-retry-base-s", type=float, default=0.8)
    ap.add_argument("--llm-strict-api-errors", action="store_true")
    ap.add_argument("--disable-llm-failure-export", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.output_dir)
    train_csv = os.path.join(args.track1_dir, "splits", "train.csv")
    val_csv = os.path.join(args.track1_dir, "splits", "val.csv")
    test_csv = os.path.join(args.track1_dir, "splits", "test.csv")
    if not (os.path.isfile(train_csv) and os.path.isfile(val_csv) and os.path.isfile(test_csv)):
        raise SystemExit("Missing split CSV files under track1/splits.")

    global TARGET_LABEL_FIELD
    TARGET_LABEL_FIELD = str(args.label_field or "strict_success")

    train_rows = load_rows(train_csv, require_binary_label=True, label_field=TARGET_LABEL_FIELD)
    val_rows = load_rows(val_csv, require_binary_label=True, label_field=TARGET_LABEL_FIELD)
    test_rows = load_rows(test_csv, require_binary_label=True, label_field=TARGET_LABEL_FIELD)
    print(f"[data] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    lambda_fp = max(0.0, float(args.lambda_false_stop))

    summary: Dict[str, Dict[str, float]] = {}
    metrics_table_rows: List[Dict[str, str]] = []

    # 1) Rule baseline
    rule_p_train = [rule_success_prob(r) for r in train_rows]
    rule_p_val = [rule_success_prob(r) for r in val_rows]
    rule_p_test = [rule_success_prob(r) for r in test_rows]
    rule_val_map = {r.get("prefix_id", ""): p for r, p in zip(val_rows, rule_p_val)}
    rule_test_map = {r.get("prefix_id", ""): p for r, p in zip(test_rows, rule_p_test)}
    rule_thr = choose_best_threshold(
        val_rows,
        rule_p_val,
        metric=args.threshold_metric,
        max_success_false_stop_rate=args.max_success_false_stop_rate,
        enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
        lambda_false_stop=lambda_fp,
        inference_cost_per_row=0.0,
    )
    rule_m = evaluate_predictions(
        test_rows,
        rule_p_test,
        rule_thr,
        lambda_false_stop=lambda_fp,
        inference_cost_per_row=0.0,
    )
    rule_m = enrich_with_oracle_headroom(test_rows, rule_m, lambda_false_stop=lambda_fp)
    summary["rule"] = rule_m
    write_predictions_csv(
        os.path.join(args.output_dir, "predictions_rule_test.csv"),
        test_rows,
        rule_p_test,
        rule_thr,
    )
    metrics_table_rows.append({"model": "rule", **{k: str(v) for k, v in rule_m.items()}})
    print(f"[rule] threshold={rule_thr:.2f} weighted_net_with_infer={rule_m['avg_weighted_net_saved_seconds_with_inference']:.3f}")
    if args.export_threshold_frontier:
        rule_frontier = scan_threshold_frontier(
            test_rows,
            rule_p_test,
            lambda_false_stop=lambda_fp,
            inference_cost_per_row=0.0,
            max_success_false_stop_rate=args.max_success_false_stop_rate,
            enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
        )
        write_frontier_csv(os.path.join(args.output_dir, "frontier_rule_test.csv"), rule_frontier)

    # 2) Supervised baseline (xgboost or fallback)
    sup_p_train, sup_p_val, sup_p_test, sup_model_name = train_supervised_model(
        train_rows,
        val_rows,
        test_rows,
        include_missing_indicators=args.include_missing_indicators,
        preferred_model=args.supervised_model,
        feature_mode=args.feature_mode,
    )
    sup_val_map = {r.get("prefix_id", ""): p for r, p in zip(val_rows, sup_p_val)}
    sup_test_map = {r.get("prefix_id", ""): p for r, p in zip(test_rows, sup_p_test)}
    sup_thr = choose_best_threshold(
        val_rows,
        sup_p_val,
        metric=args.threshold_metric,
        max_success_false_stop_rate=args.max_success_false_stop_rate,
        enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
        lambda_false_stop=lambda_fp,
        inference_cost_per_row=0.0,
    )
    sup_m = evaluate_predictions(
        test_rows,
        sup_p_test,
        sup_thr,
        lambda_false_stop=lambda_fp,
        inference_cost_per_row=0.0,
    )
    sup_m = enrich_with_oracle_headroom(test_rows, sup_m, lambda_false_stop=lambda_fp)
    summary[sup_model_name] = sup_m
    write_predictions_csv(
        os.path.join(args.output_dir, f"predictions_{sup_model_name}_test.csv"),
        test_rows,
        sup_p_test,
        sup_thr,
    )
    metrics_table_rows.append({"model": sup_model_name, **{k: str(v) for k, v in sup_m.items()}})
    print(f"[{sup_model_name}] threshold={sup_thr:.2f} weighted_net_with_infer={sup_m['avg_weighted_net_saved_seconds_with_inference']:.3f}")
    if args.export_threshold_frontier:
        sup_frontier = scan_threshold_frontier(
            test_rows,
            sup_p_test,
            lambda_false_stop=lambda_fp,
            inference_cost_per_row=0.0,
            max_success_false_stop_rate=args.max_success_false_stop_rate,
            enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
        )
        write_frontier_csv(os.path.join(args.output_dir, f"frontier_{sup_model_name}_test.csv"), sup_frontier)

    # 3) Constrained Group-Relative (CGR) non-neural policy
    cgr_meta: Dict[str, object] = {}
    cgr_min_total = int(args.cgr_min_group_total)
    cgr_min_each = int(args.cgr_min_group_each)
    if not args.disable_cgr:
        auto_group_meta: Dict[str, float] = {}
        if args.cgr_auto_group:
            cgr_min_total, cgr_min_each, auto_group_meta = recommend_cgr_group_thresholds(
                train_rows=train_rows,
                elapsed_bins=args.cgr_elapsed_bins,
                target_coverage=args.cgr_target_coverage,
            )
            print(
                f"[cgr] auto-group min_total={cgr_min_total} min_each={cgr_min_each} coverage={auto_group_meta.get('coverage', 0.0):.3f}",
                flush=True,
            )

        cgr_p_train, cgr_p_val, cgr_p_test, cgr_model_name, cgr_meta = run_cgr_policy(
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            elapsed_bins=args.cgr_elapsed_bins,
            min_group_total=cgr_min_total,
            min_group_each=cgr_min_each,
            lambda_cost=args.cgr_lambda_cost,
            eta_uncertainty=args.cgr_eta_uncertainty,
            alpha=args.cgr_alpha,
            beta=args.cgr_beta,
            risk_mix=args.cgr_risk_mix,
        )
        cgr_val_map = {r.get("prefix_id", ""): p for r, p in zip(val_rows, cgr_p_val)}
        cgr_test_map = {r.get("prefix_id", ""): p for r, p in zip(test_rows, cgr_p_test)}
        if args.cgr_auto_group:
            cgr_meta["auto_group"] = auto_group_meta
        cgr_thr = choose_best_threshold(
            val_rows,
            cgr_p_val,
            metric=args.threshold_metric,
            max_success_false_stop_rate=args.max_success_false_stop_rate,
            enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
            lambda_false_stop=lambda_fp,
            inference_cost_per_row=0.0,
        )
        cgr_m = evaluate_predictions(
            test_rows,
            cgr_p_test,
            cgr_thr,
            lambda_false_stop=lambda_fp,
            inference_cost_per_row=0.0,
        )
        cgr_m = enrich_with_oracle_headroom(test_rows, cgr_m, lambda_false_stop=lambda_fp)
        summary[cgr_model_name] = cgr_m
        write_predictions_csv(
            os.path.join(args.output_dir, f"predictions_{cgr_model_name}_test.csv"),
            test_rows,
            cgr_p_test,
            cgr_thr,
        )
        metrics_table_rows.append({"model": cgr_model_name, **{k: str(v) for k, v in cgr_m.items()}})
        print(f"[{cgr_model_name}] threshold={cgr_thr:.2f} weighted_net_with_infer={cgr_m['avg_weighted_net_saved_seconds_with_inference']:.3f}")
        if args.export_threshold_frontier:
            cgr_frontier = scan_threshold_frontier(
                test_rows,
                cgr_p_test,
                lambda_false_stop=lambda_fp,
                inference_cost_per_row=0.0,
                max_success_false_stop_rate=args.max_success_false_stop_rate,
                enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
            )
            write_frontier_csv(os.path.join(args.output_dir, f"frontier_{cgr_model_name}_test.csv"), cgr_frontier)

    # 4) LLM baseline (optional)
    if args.run_llm:
        api_key = args.llm_api_key.strip() or load_llm_api_key(args.llm_provider, args.llm_driver_path)
        llm_api_base = str(args.llm_api_base or "").strip()
        if not api_key:
            print(f"[llm] No API key found for provider={args.llm_provider} (env or driver file). Skip LLM baseline.")
        else:
            cache_jsonl = os.path.join(args.output_dir, "llm_cache.jsonl")
            request_status_jsonl = os.path.join(args.output_dir, "llm_request_status.jsonl")

            def subset_probs(rows_subset: List[Dict[str, str]], prob_map: Dict[str, float]) -> List[float]:
                out: List[float] = []
                for r in rows_subset:
                    pid = r.get("prefix_id", "")
                    out.append(float(prob_map.get(pid, 0.5)))
                return out

            if args.llm_fair_compare:
                budget = max(2, int(args.llm_max_samples))
                val_ratio = max(0.1, min(0.9, float(args.llm_fair_val_ratio)))
                n_val = max(1, int(round(budget * val_ratio)))
                n_test = max(1, budget - n_val)
                n_val = min(n_val, len(val_rows))
                n_test = min(n_test, len(test_rows))

                t_llm_val_st = time.time()
                pred_llm_val, p_llm_val, n_llm_val, sample_val_rows, llm_val_details = run_llm_baseline(
                    rows=val_rows,
                    api_key=api_key,
                    model_name=args.llm_model,
                    provider=args.llm_provider,
                    api_base=llm_api_base,
                    openai_wire_api=args.llm_openai_wire_api,
                    openai_reasoning_effort=args.llm_openai_reasoning_effort,
                    anthropic_version=args.llm_anthropic_version,
                    max_samples=n_val,
                    seed=args.llm_seed + 17,
                    cache_jsonl=cache_jsonl,
                    sleep_s=args.llm_sleep_s,
                    prompt_style=args.llm_prompt_style,
                    log_style=args.llm_log_style,
                    two_stage=args.llm_two_stage,
                    use_cache=(not args.llm_ignore_cache),
                    refresh_missing_rationale=args.llm_refresh_missing_rationale,
                    require_rationale=args.llm_require_rationale,
                    min_risk_tags=args.llm_min_risk_tags,
                    min_evidence=args.llm_min_evidence,
                    rationale_retry=args.llm_rationale_retry,
                    api_max_retries=args.llm_api_max_retries,
                    api_retry_base_sleep_s=args.llm_api_retry_base_s,
                    strict_api_errors=args.llm_strict_api_errors,
                    request_status_jsonl=request_status_jsonl,
                    split_name="val",
                )
                t_llm_val_ed = time.time()
                t_llm_test_st = time.time()
                pred_llm_test, p_llm_test, n_llm_test, sample_test_rows, llm_test_details = run_llm_baseline(
                    rows=test_rows,
                    api_key=api_key,
                    model_name=args.llm_model,
                    provider=args.llm_provider,
                    api_base=llm_api_base,
                    openai_wire_api=args.llm_openai_wire_api,
                    openai_reasoning_effort=args.llm_openai_reasoning_effort,
                    anthropic_version=args.llm_anthropic_version,
                    max_samples=n_test,
                    seed=args.llm_seed + 23,
                    cache_jsonl=cache_jsonl,
                    sleep_s=args.llm_sleep_s,
                    prompt_style=args.llm_prompt_style,
                    log_style=args.llm_log_style,
                    two_stage=args.llm_two_stage,
                    use_cache=(not args.llm_ignore_cache),
                    refresh_missing_rationale=args.llm_refresh_missing_rationale,
                    require_rationale=args.llm_require_rationale,
                    min_risk_tags=args.llm_min_risk_tags,
                    min_evidence=args.llm_min_evidence,
                    rationale_retry=args.llm_rationale_retry,
                    api_max_retries=args.llm_api_max_retries,
                    api_retry_base_sleep_s=args.llm_api_retry_base_s,
                    strict_api_errors=args.llm_strict_api_errors,
                    request_status_jsonl=request_status_jsonl,
                    split_name="test",
                )
                t_llm_test_ed = time.time()

                p_llm_val_eval = p_llm_val
                p_llm_test_eval = p_llm_test
                llm_calib_meta: Dict[str, object] = {"mode": args.llm_calibration}
                if args.llm_calibration == "discrete":
                    calib = fit_discrete_success_calibrator(
                        sample_val_rows,
                        p_llm_val,
                        prior_strength=args.llm_calibration_prior,
                    )
                    p_llm_val_eval = apply_discrete_success_calibration(p_llm_val, calib)
                    p_llm_test_eval = apply_discrete_success_calibration(p_llm_test, calib)
                    llm_calib_meta = {
                        "mode": "discrete",
                        "prior_strength": float(args.llm_calibration_prior),
                        "n_buckets": len(calib.get("keys", [])),
                    }

                if args.llm_include_inference_cost:
                    if args.llm_inference_cost_per_row_s >= 0:
                        llm_infer_val = float(args.llm_inference_cost_per_row_s)
                        llm_infer_test = float(args.llm_inference_cost_per_row_s)
                    else:
                        llm_infer_val = (t_llm_val_ed - t_llm_val_st) / max(1, n_llm_val)
                        llm_infer_test = (t_llm_test_ed - t_llm_test_st) / max(1, n_llm_test)
                else:
                    llm_infer_val = 0.0
                    llm_infer_test = 0.0

                if args.llm_decision_mode == "direct":
                    llm_thr = -1.0
                    llm_m = evaluate_with_fixed_stop(
                        sample_test_rows,
                        pred_llm_test,
                        lambda_false_stop=lambda_fp,
                        inference_cost_per_row=llm_infer_test,
                        success_prob=p_llm_test_eval,
                        stop_threshold=llm_thr,
                    )
                else:
                    llm_thr = choose_best_threshold(
                        sample_val_rows,
                        p_llm_val_eval,
                        metric=args.threshold_metric,
                        max_success_false_stop_rate=args.max_success_false_stop_rate,
                        enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
                        lambda_false_stop=lambda_fp,
                        inference_cost_per_row=llm_infer_val,
                    )
                    llm_m = evaluate_predictions(
                        sample_test_rows,
                        p_llm_test_eval,
                        llm_thr,
                        lambda_false_stop=lambda_fp,
                        inference_cost_per_row=llm_infer_test,
                    )
                llm_m = enrich_with_oracle_headroom(sample_test_rows, llm_m, lambda_false_stop=lambda_fp)
                llm_m["n_sampled_val"] = n_llm_val
                llm_m["n_sampled_test"] = n_llm_test
                llm_m["fair_compare"] = 1
                llm_m["threshold_from_val"] = 1
                llm_m["budget_total"] = int(n_llm_val + n_llm_test)
                llm_m["llm_decision_mode"] = args.llm_decision_mode
                llm_m["llm_calibration_mode"] = llm_calib_meta.get("mode", "none")
                llm_m["llm_require_rationale"] = (1 if args.llm_require_rationale else 0)
                llm_m["llm_min_risk_tags"] = int(args.llm_min_risk_tags)
                llm_m["llm_min_evidence"] = int(args.llm_min_evidence)
                if "n_buckets" in llm_calib_meta:
                    llm_m["llm_calibration_buckets"] = llm_calib_meta["n_buckets"]
                summary[f"llm_{args.llm_model}_fair"] = llm_m
                write_predictions_csv(
                    os.path.join(args.output_dir, f"predictions_llm_{args.llm_model}_val_fair.csv"),
                    sample_val_rows,
                    p_llm_val_eval,
                    llm_thr,
                    pred_stop_override=(pred_llm_val if args.llm_decision_mode == "direct" else None),
                )
                write_predictions_csv(
                    os.path.join(args.output_dir, f"predictions_llm_{args.llm_model}_test_fair.csv"),
                    sample_test_rows,
                    p_llm_test_eval,
                    llm_thr,
                    pred_stop_override=(pred_llm_test if args.llm_decision_mode == "direct" else None),
                )
                metrics_table_rows.append({"model": f"llm_{args.llm_model}_fair", **{k: str(v) for k, v in llm_m.items()}})

                # Fair sampled comparisons for non-LLM models on the same sampled rows/protocol.
                rule_val_sub = subset_probs(sample_val_rows, rule_val_map)
                rule_test_sub = subset_probs(sample_test_rows, rule_test_map)
                rule_thr_sub = choose_best_threshold(
                    sample_val_rows,
                    rule_val_sub,
                    metric=args.threshold_metric,
                    max_success_false_stop_rate=args.max_success_false_stop_rate,
                    enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
                    lambda_false_stop=lambda_fp,
                    inference_cost_per_row=0.0,
                )
                rule_m_sub = evaluate_predictions(
                    sample_test_rows,
                    rule_test_sub,
                    rule_thr_sub,
                    lambda_false_stop=lambda_fp,
                    inference_cost_per_row=0.0,
                )
                rule_m_sub = enrich_with_oracle_headroom(sample_test_rows, rule_m_sub, lambda_false_stop=lambda_fp)
                rule_m_sub["n_sampled_val"] = n_llm_val
                rule_m_sub["n_sampled_test"] = n_llm_test
                rule_m_sub["fair_compare"] = 1
                summary["rule_fair"] = rule_m_sub
                metrics_table_rows.append({"model": "rule_fair", **{k: str(v) for k, v in rule_m_sub.items()}})

                sup_val_sub = subset_probs(sample_val_rows, sup_val_map)
                sup_test_sub = subset_probs(sample_test_rows, sup_test_map)
                sup_thr_sub = choose_best_threshold(
                    sample_val_rows,
                    sup_val_sub,
                    metric=args.threshold_metric,
                    max_success_false_stop_rate=args.max_success_false_stop_rate,
                    enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
                    lambda_false_stop=lambda_fp,
                    inference_cost_per_row=0.0,
                )
                sup_m_sub = evaluate_predictions(
                    sample_test_rows,
                    sup_test_sub,
                    sup_thr_sub,
                    lambda_false_stop=lambda_fp,
                    inference_cost_per_row=0.0,
                )
                sup_m_sub = enrich_with_oracle_headroom(sample_test_rows, sup_m_sub, lambda_false_stop=lambda_fp)
                sup_m_sub["n_sampled_val"] = n_llm_val
                sup_m_sub["n_sampled_test"] = n_llm_test
                sup_m_sub["fair_compare"] = 1
                summary[f"{sup_model_name}_fair"] = sup_m_sub
                metrics_table_rows.append({"model": f"{sup_model_name}_fair", **{k: str(v) for k, v in sup_m_sub.items()}})

                if not args.disable_llm_failure_export:
                    failure_artifacts = export_llm_failure_analysis(
                        output_dir=args.output_dir,
                        model_name=args.llm_model,
                        split_name="fair_test",
                        rows=sample_test_rows,
                        llm_success_prob=p_llm_test_eval,
                        llm_pred_stop=pred_llm_test,
                        llm_details=llm_test_details,
                        xgb_success_prob=sup_test_sub,
                        xgb_threshold=sup_thr_sub,
                        lambda_false_stop=lambda_fp,
                    )
                    llm_m["failure_summary_path"] = failure_artifacts["summary_json"]
                    llm_m["failure_false_stop_rate"] = float(
                        failure_artifacts["summary"].get("false_stop_rate", 0.0)  # type: ignore[index]
                    )
                    llm_m["failure_missed_stop_rate"] = float(
                        failure_artifacts["summary"].get("missed_stop_rate", 0.0)  # type: ignore[index]
                    )

                if not args.disable_cgr:
                    cgr_val_sub = subset_probs(sample_val_rows, cgr_val_map)
                    cgr_test_sub = subset_probs(sample_test_rows, cgr_test_map)
                    cgr_thr_sub = choose_best_threshold(
                        sample_val_rows,
                        cgr_val_sub,
                        metric=args.threshold_metric,
                        max_success_false_stop_rate=args.max_success_false_stop_rate,
                        enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
                        lambda_false_stop=lambda_fp,
                        inference_cost_per_row=0.0,
                    )
                    cgr_m_sub = evaluate_predictions(
                        sample_test_rows,
                        cgr_test_sub,
                        cgr_thr_sub,
                        lambda_false_stop=lambda_fp,
                        inference_cost_per_row=0.0,
                    )
                    cgr_m_sub = enrich_with_oracle_headroom(sample_test_rows, cgr_m_sub, lambda_false_stop=lambda_fp)
                    cgr_m_sub["n_sampled_val"] = n_llm_val
                    cgr_m_sub["n_sampled_test"] = n_llm_test
                    cgr_m_sub["fair_compare"] = 1
                    summary[f"{cgr_model_name}_fair"] = cgr_m_sub
                    metrics_table_rows.append({"model": f"{cgr_model_name}_fair", **{k: str(v) for k, v in cgr_m_sub.items()}})

                print(
                    f"[llm-fair] model={args.llm_model} val={n_llm_val} test={n_llm_test} thr={llm_thr:.2f} weighted_net_with_infer={llm_m['avg_weighted_net_saved_seconds_with_inference']:.3f}",
                    flush=True,
                )
                if args.export_threshold_frontier:
                    llm_frontier = scan_threshold_frontier(
                        sample_test_rows,
                        p_llm_test,
                        lambda_false_stop=lambda_fp,
                        inference_cost_per_row=llm_infer_test,
                        max_success_false_stop_rate=args.max_success_false_stop_rate,
                        enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
                    )
                    write_frontier_csv(os.path.join(args.output_dir, f"frontier_llm_{args.llm_model}_test_fair.csv"), llm_frontier)
            else:
                t_llm_st = time.time()
                pred_stop_llm, p_llm, n_llm, sample_rows, llm_sample_details = run_llm_baseline(
                    rows=test_rows,
                    api_key=api_key,
                    model_name=args.llm_model,
                    provider=args.llm_provider,
                    api_base=llm_api_base,
                    openai_wire_api=args.llm_openai_wire_api,
                    openai_reasoning_effort=args.llm_openai_reasoning_effort,
                    anthropic_version=args.llm_anthropic_version,
                    max_samples=args.llm_max_samples,
                    seed=args.llm_seed,
                    cache_jsonl=cache_jsonl,
                    sleep_s=args.llm_sleep_s,
                    prompt_style=args.llm_prompt_style,
                    log_style=args.llm_log_style,
                    two_stage=args.llm_two_stage,
                    use_cache=(not args.llm_ignore_cache),
                    refresh_missing_rationale=args.llm_refresh_missing_rationale,
                    require_rationale=args.llm_require_rationale,
                    min_risk_tags=args.llm_min_risk_tags,
                    min_evidence=args.llm_min_evidence,
                    rationale_retry=args.llm_rationale_retry,
                    api_max_retries=args.llm_api_max_retries,
                    api_retry_base_sleep_s=args.llm_api_retry_base_s,
                    strict_api_errors=args.llm_strict_api_errors,
                    request_status_jsonl=request_status_jsonl,
                    split_name="test_sample",
                )
                t_llm_ed = time.time()
                p_llm_eval = p_llm
                if args.llm_calibration == "discrete":
                    calib = fit_discrete_success_calibrator(sample_rows, p_llm, prior_strength=args.llm_calibration_prior)
                    p_llm_eval = apply_discrete_success_calibration(p_llm, calib)
                if args.llm_include_inference_cost:
                    if args.llm_inference_cost_per_row_s >= 0:
                        llm_infer = float(args.llm_inference_cost_per_row_s)
                    else:
                        llm_infer = (t_llm_ed - t_llm_st) / max(1, n_llm)
                else:
                    llm_infer = 0.0

                if args.llm_decision_mode == "direct":
                    llm_thr = -1.0
                    llm_m = evaluate_with_fixed_stop(
                        sample_rows,
                        pred_stop_llm,
                        lambda_false_stop=lambda_fp,
                        inference_cost_per_row=llm_infer,
                        success_prob=p_llm_eval,
                        stop_threshold=llm_thr,
                    )
                else:
                    llm_thr = 0.5
                    llm_m = evaluate_predictions(
                        sample_rows,
                        p_llm_eval,
                        llm_thr,
                        lambda_false_stop=lambda_fp,
                        inference_cost_per_row=llm_infer,
                    )
                llm_m = enrich_with_oracle_headroom(sample_rows, llm_m, lambda_false_stop=lambda_fp)
                llm_m["n_sampled"] = n_llm
                llm_m["llm_decision_mode"] = args.llm_decision_mode
                llm_m["llm_calibration_mode"] = args.llm_calibration
                llm_m["llm_prompt_style"] = args.llm_prompt_style
                llm_m["llm_log_style"] = args.llm_log_style
                llm_m["llm_two_stage"] = (1 if args.llm_two_stage else 0)
                llm_m["llm_require_rationale"] = (1 if args.llm_require_rationale else 0)
                llm_m["llm_min_risk_tags"] = int(args.llm_min_risk_tags)
                llm_m["llm_min_evidence"] = int(args.llm_min_evidence)
                summary[f"llm_{args.llm_model}"] = llm_m
                write_predictions_csv(
                    os.path.join(args.output_dir, f"predictions_llm_{args.llm_model}_test_sample.csv"),
                    sample_rows,
                    p_llm_eval,
                    llm_thr,
                    pred_stop_override=(pred_stop_llm if args.llm_decision_mode == "direct" else None),
                )
                metrics_table_rows.append({"model": f"llm_{args.llm_model}", **{k: str(v) for k, v in llm_m.items()}})
                print(f"[llm] model={args.llm_model} sampled={n_llm} weighted_net_with_infer={llm_m['avg_weighted_net_saved_seconds_with_inference']:.3f}")

                if not args.disable_llm_failure_export:
                    sup_test_sample = subset_probs(sample_rows, sup_test_map)
                    sup_thr_sample = choose_best_threshold(
                        val_rows,
                        sup_p_val,
                        metric=args.threshold_metric,
                        max_success_false_stop_rate=args.max_success_false_stop_rate,
                        enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
                        lambda_false_stop=lambda_fp,
                        inference_cost_per_row=0.0,
                    )
                    failure_artifacts = export_llm_failure_analysis(
                        output_dir=args.output_dir,
                        model_name=args.llm_model,
                        split_name="test_sample",
                        rows=sample_rows,
                        llm_success_prob=p_llm_eval,
                        llm_pred_stop=pred_stop_llm if args.llm_decision_mode == "direct" else [1 if p < llm_thr else 0 for p in p_llm_eval],
                        llm_details=llm_sample_details,
                        xgb_success_prob=sup_test_sample,
                        xgb_threshold=sup_thr_sample,
                        lambda_false_stop=lambda_fp,
                    )
                    llm_m["failure_summary_path"] = failure_artifacts["summary_json"]
                    llm_m["failure_false_stop_rate"] = float(
                        failure_artifacts["summary"].get("false_stop_rate", 0.0)  # type: ignore[index]
                    )
                    llm_m["failure_missed_stop_rate"] = float(
                        failure_artifacts["summary"].get("missed_stop_rate", 0.0)  # type: ignore[index]
                    )
                if args.export_threshold_frontier:
                    llm_frontier = scan_threshold_frontier(
                        sample_rows,
                        p_llm,
                        lambda_false_stop=lambda_fp,
                        inference_cost_per_row=llm_infer,
                        max_success_false_stop_rate=args.max_success_false_stop_rate,
                        enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
                    )
                    write_frontier_csv(os.path.join(args.output_dir, f"frontier_llm_{args.llm_model}_test_sample.csv"), llm_frontier)

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "track1_dir": args.track1_dir,
        "output_dir": args.output_dir,
        "label_field": TARGET_LABEL_FIELD,
        "threshold_metric": args.threshold_metric,
        "max_success_false_stop_rate": args.max_success_false_stop_rate,
        "threshold_constraint_enabled": (not args.disable_threshold_constraint),
        "reward_config": {
            "lambda_false_stop": lambda_fp,
            "llm_include_inference_cost": args.llm_include_inference_cost,
            "llm_inference_cost_per_row_s": args.llm_inference_cost_per_row_s,
            "export_threshold_frontier": args.export_threshold_frontier,
        },
        "include_missing_indicators": args.include_missing_indicators,
        "supervised_config": {
            "supervised_model": args.supervised_model,
            "selected_model": sup_model_name,
            "feature_mode": args.feature_mode,
        },
        "cgr_enabled": (not args.disable_cgr),
        "llm_config": {
            "run_llm": args.run_llm,
            "llm_provider": args.llm_provider,
            "llm_api_base": "[redacted]" if args.run_llm else "",
            "llm_openai_wire_api": args.llm_openai_wire_api,
            "llm_openai_reasoning_effort": args.llm_openai_reasoning_effort,
            "llm_anthropic_version": args.llm_anthropic_version,
            "llm_model": args.llm_model,
            "llm_max_samples": args.llm_max_samples,
            "llm_fair_compare": args.llm_fair_compare,
            "llm_fair_val_ratio": args.llm_fair_val_ratio,
            "llm_decision_mode": args.llm_decision_mode,
            "llm_calibration": args.llm_calibration,
            "llm_calibration_prior": args.llm_calibration_prior,
            "llm_prompt_style": args.llm_prompt_style,
            "llm_log_style": args.llm_log_style,
            "llm_two_stage": args.llm_two_stage,
            "llm_ignore_cache": args.llm_ignore_cache,
            "llm_refresh_missing_rationale": args.llm_refresh_missing_rationale,
            "llm_require_rationale": args.llm_require_rationale,
            "llm_min_risk_tags": args.llm_min_risk_tags,
            "llm_min_evidence": args.llm_min_evidence,
            "llm_rationale_retry": args.llm_rationale_retry,
            "llm_api_max_retries": args.llm_api_max_retries,
            "llm_api_retry_base_s": args.llm_api_retry_base_s,
            "disable_llm_failure_export": args.disable_llm_failure_export,
        },
        "cgr_config": {
            "elapsed_bins": args.cgr_elapsed_bins,
            "min_group_total": cgr_min_total,
            "min_group_each": cgr_min_each,
            "lambda_cost": args.cgr_lambda_cost,
            "eta_uncertainty": args.cgr_eta_uncertainty,
            "alpha": args.cgr_alpha,
            "beta": args.cgr_beta,
            "risk_mix": args.cgr_risk_mix,
            "auto_group": args.cgr_auto_group,
            "target_coverage": args.cgr_target_coverage,
            "runtime": cgr_meta,
        },
        "data_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "metrics": summary,
    }
    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_metrics_table(os.path.join(args.output_dir, "metrics_table.csv"), metrics_table_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
