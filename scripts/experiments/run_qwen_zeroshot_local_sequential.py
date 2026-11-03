#!/usr/bin/env python3
"""
Run local Qwen zero-shot inference under run-level sequential protocol.

Protocol:
- Decision points: init -> place -> cts -> grt -> drt
- First STOP ends the run; each run contributes utility once.
- Thresholds are checkpoint-specific and selected on val split with FSR constraint.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import run_track1_baselines as b  # noqa: E402
import run_xgb_sequential_track1 as seq  # noqa: E402


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ds_name(ds_dir: str) -> str:
    return os.path.basename(os.path.normpath(ds_dir))


def discover_dataset_dirs(dataset_dir_glob: List[str]) -> List[str]:
    ds_dirs: List[str] = []
    for pat in dataset_dir_glob:
        ds_dirs.extend([p for p in glob.glob(pat) if os.path.isdir(p)])
    ds_dirs = sorted(set(ds_dirs))

    ok: List[str] = []
    for d in ds_dirs:
        tr = os.path.join(d, "track1_strict", "splits", "train.csv")
        va = os.path.join(d, "track1_strict", "splits", "val.csv")
        te = os.path.join(d, "track1_strict", "splits", "test.csv")
        if os.path.isfile(tr) and os.path.isfile(va) and os.path.isfile(te):
            ok.append(d)
    return ok


def _to_device(enc: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in enc.items()}


def _decode_generated_tail(
    tokenizer,
    generated_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> List[str]:
    outs: List[str] = []
    lens = attention_mask.sum(dim=1).tolist()
    for i in range(generated_ids.size(0)):
        start = int(lens[i])
        tail_ids = generated_ids[i, start:]
        txt = tokenizer.decode(tail_ids, skip_special_tokens=True).strip()
        outs.append(txt)
    return outs


def infer_rows_local(
    rows: List[Dict[str, str]],
    tokenizer,
    model,
    device: torch.device,
    prompt_style: str,
    log_style: str,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
) -> Tuple[List[float], List[int], List[Dict[str, object]], float]:
    if not rows:
        return [], [], [], 0.0

    prompts = [b.build_llm_prompt(r, prompt_style=prompt_style, log_style=log_style) for r in rows]
    success_prob: List[float] = []
    pred_stop: List[int] = []
    details: List[Dict[str, object]] = []

    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), max(1, int(batch_size))):
            batch_prompts = prompts[i : i + max(1, int(batch_size))]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(64, int(max_input_length)),
            )
            enc = _to_device(enc, device)
            out = model.generate(
                **enc,
                max_new_tokens=max(8, int(max_new_tokens)),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_texts = _decode_generated_tail(tokenizer, out, enc["attention_mask"])
            for txt in gen_texts:
                ps, sp, det = b.parse_llm_payload(txt)
                pred_stop.append(int(ps))
                success_prob.append(float(sp))
                details.append(det)
            del enc
            del out
            if device.type == "cuda":
                torch.cuda.empty_cache()
    t1 = time.time()

    avg_sec = (t1 - t0) / max(1, len(rows))
    return success_prob, pred_stop, details, float(avg_sec)


def write_predictions_csv(
    path: str,
    rows: List[Dict[str, str]],
    success_prob: List[float],
    thresholds: Dict[str, float],
) -> None:
    fields = [
        "prefix_id",
        "record_key",
        "checkpoint",
        "strict_success",
        "eval_label_field",
        "eval_label_value",
        "success_prob",
        "threshold",
        "pred_success",
        "pred_stop",
        "final_wallclock_seconds",
        "prefix_elapsed_seconds",
        "remaining_seconds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            p = float(success_prob[i]) if i < len(success_prob) else 0.5
            ck = (r.get("checkpoint", "") or "").strip()
            thr = float(thresholds.get(ck, 0.5))
            s = 1 if p < thr else 0
            w.writerow(
                {
                    "prefix_id": r.get("prefix_id", ""),
                    "record_key": r.get("record_key", ""),
                    "checkpoint": ck,
                    "strict_success": r.get("strict_success", ""),
                    "eval_label_field": b.TARGET_LABEL_FIELD,
                    "eval_label_value": r.get(b.TARGET_LABEL_FIELD, ""),
                    "success_prob": f"{p:.6f}",
                    "threshold": f"{thr:.6f}",
                    "pred_success": "1" if p >= 0.5 else "0",
                    "pred_stop": "1" if s == 1 else "0",
                    "final_wallclock_seconds": r.get("final_wallclock_seconds", ""),
                    "prefix_elapsed_seconds": r.get("prefix_elapsed_seconds", ""),
                    "remaining_seconds": r.get("remaining_seconds", ""),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run local Qwen zero-shot with sequential run-level protocol.")
    ap.add_argument("--dataset-dir-glob", action="append", required=True, help="Glob(s) for dataset dirs.")
    ap.add_argument("--target-dataset", action="append", default=[], help="Optional dataset basename(s).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-field", default="strict_success")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model-tag", default="Qwen2.5-0.5B-Instruct_zeroshot_seq")
    ap.add_argument("--prompt-style", choices=["basic", "weak", "strong", "expert"], default="weak")
    ap.add_argument("--log-style", choices=["basic", "enhanced"], default="basic")
    ap.add_argument("--max-success-false-stop-rate", type=float, default=0.10)
    ap.add_argument("--lambda-false-stop", type=float, default=1.0)
    ap.add_argument("--disable-threshold-constraint", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-input-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--max-val-rows", type=int, default=0)
    ap.add_argument("--max-test-rows", type=int, default=0)
    ap.add_argument("--inference-cost-per-row", type=float, default=0.0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    b.TARGET_LABEL_FIELD = str(args.label_field or "strict_success")

    ds_dirs = discover_dataset_dirs(args.dataset_dir_glob)
    if not ds_dirs:
        raise SystemExit("No dataset dirs with track1_strict/splits/{train,val,test}.csv found.")

    target_filter = {str(x).strip() for x in (args.target_dataset or []) if str(x).strip()}
    if target_filter:
        ds_dirs = [d for d in ds_dirs if ds_name(d) in target_filter]
    if not ds_dirs:
        raise SystemExit("No target datasets matched --target-dataset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for local zero-shot run.")

    os.makedirs(args.output_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=bool(args.local_files_only))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=bool(args.local_files_only),
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    fold_rows: List[Dict[str, str]] = []

    for ds in ds_dirs:
        name = ds_name(ds)
        track1_dir = os.path.join(ds, "track1_strict")
        out_ds = os.path.join(args.output_dir, name)
        os.makedirs(out_ds, exist_ok=True)
        metrics_json = os.path.join(out_ds, "metrics_summary.json")
        pred_csv = os.path.join(out_ds, "predictions_qwen_zeroshot_seq_test.csv")

        if args.skip_existing and os.path.isfile(metrics_json) and os.path.isfile(pred_csv):
            with open(metrics_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
            mt = obj.get("metrics_test", {})
            th = obj.get("thresholds", {})
            fold_rows.append(
                {
                    "dataset": name,
                    "status": "ok",
                    "n_runs": str(float(mt.get("n_runs", 0.0))),
                    "accuracy": str(float(mt.get("accuracy", 0.0))),
                    "fail_recall_stop": str(float(mt.get("fail_recall_stop", 0.0))),
                    "success_false_stop_rate": str(float(mt.get("success_false_stop_rate", 0.0))),
                    "stop_rate": str(float(mt.get("stop_rate", 0.0))),
                    "avg_inference_steps": str(float(mt.get("avg_inference_steps", 0.0))),
                    "stop_precision_on_stopped": str(float(mt.get("stop_precision_on_stopped", 0.0))),
                    "net_with_infer": str(float(mt.get("avg_weighted_net_saved_seconds_with_inference", 0.0))),
                    "oracle": str(float(mt.get("oracle_avg_weighted_net_saved_seconds", 0.0))),
                    "headroom_ratio_vs_oracle": str(float(mt.get("headroom_ratio_vs_oracle", 0.0))),
                    "threshold_init": str(float(th.get("init", 0.0))),
                    "threshold_place": str(float(th.get("place", 0.0))),
                    "threshold_cts": str(float(th.get("cts", 0.0))),
                    "threshold_grt": str(float(th.get("grt", 0.0))),
                    "threshold_drt": str(float(th.get("drt", 0.0))),
                }
            )
            continue

        val_rows = b.load_rows(
            os.path.join(track1_dir, "splits", "val.csv"),
            require_binary_label=True,
            label_field=b.TARGET_LABEL_FIELD,
        )
        test_rows = b.load_rows(
            os.path.join(track1_dir, "splits", "test.csv"),
            require_binary_label=True,
            label_field=b.TARGET_LABEL_FIELD,
        )
        if args.max_val_rows and args.max_val_rows > 0:
            val_rows = val_rows[: int(args.max_val_rows)]
        if args.max_test_rows and args.max_test_rows > 0:
            test_rows = test_rows[: int(args.max_test_rows)]
        if not val_rows or not test_rows:
            fold_rows.append({"dataset": name, "status": "skipped_empty_split"})
            continue

        p_val, _pred_stop_val, _det_val, _val_avg = infer_rows_local(
            val_rows,
            tokenizer=tok,
            model=model,
            device=device,
            prompt_style=args.prompt_style,
            log_style=args.log_style,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
        )
        p_test, _pred_stop_test, _det_test, _test_avg = infer_rows_local(
            test_rows,
            tokenizer=tok,
            model=model,
            device=device,
            prompt_style=args.prompt_style,
            log_style=args.log_style,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
        )

        th = seq.choose_thresholds_coordinate_descent(
            val_rows,
            p_val,
            max_success_false_stop_rate=float(args.max_success_false_stop_rate),
            lambda_false_stop=float(args.lambda_false_stop),
            inference_cost_per_row=float(args.inference_cost_per_row),
            enforce_constraint=(not args.disable_threshold_constraint),
            rounds=3,
        )

        m_test, _run_pred_rows = seq.run_level_eval(
            test_rows,
            p_test,
            thresholds=th,
            lambda_false_stop=float(args.lambda_false_stop),
            inference_cost_per_row=float(args.inference_cost_per_row),
        )

        write_predictions_csv(
            pred_csv,
            rows=test_rows,
            success_prob=p_test,
            thresholds=th,
        )

        obj = {
            "generated_at": now_utc_iso(),
            "dataset": name,
            "dataset_dir": ds,
            "label_field": b.TARGET_LABEL_FIELD,
            "model_name": args.model_name,
            "model_tag": args.model_tag,
            "prompt_style": args.prompt_style,
            "log_style": args.log_style,
            "max_success_false_stop_rate": float(args.max_success_false_stop_rate),
            "threshold_constraint_enabled": (not args.disable_threshold_constraint),
            "lambda_false_stop": float(args.lambda_false_stop),
            "inference_cost_per_row": float(args.inference_cost_per_row),
            "thresholds": th,
            "metrics_test": m_test,
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
        }
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

        fold_rows.append(
            {
                "dataset": name,
                "status": "ok",
                "n_runs": str(float(m_test.get("n_runs", 0.0))),
                "accuracy": str(float(m_test.get("accuracy", 0.0))),
                "fail_recall_stop": str(float(m_test.get("fail_recall_stop", 0.0))),
                "success_false_stop_rate": str(float(m_test.get("success_false_stop_rate", 0.0))),
                "stop_rate": str(float(m_test.get("stop_rate", 0.0))),
                "avg_inference_steps": str(float(m_test.get("avg_inference_steps", 0.0))),
                "stop_precision_on_stopped": str(float(m_test.get("stop_precision_on_stopped", 0.0))),
                "net_with_infer": str(float(m_test.get("avg_weighted_net_saved_seconds_with_inference", 0.0))),
                "oracle": str(float(m_test.get("oracle_avg_weighted_net_saved_seconds", 0.0))),
                "headroom_ratio_vs_oracle": str(float(m_test.get("headroom_ratio_vs_oracle", 0.0))),
                "threshold_init": str(float(th.get("init", 0.0))),
                "threshold_place": str(float(th.get("place", 0.0))),
                "threshold_cts": str(float(th.get("cts", 0.0))),
                "threshold_grt": str(float(th.get("grt", 0.0))),
                "threshold_drt": str(float(th.get("drt", 0.0))),
            }
        )

    fields = list(fold_rows[0].keys()) if fold_rows else ["dataset", "status"]
    with open(os.path.join(args.output_dir, "zs_seq_folds.csv"), "w", newline="", encoding="utf-8") as f:
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
        lo, hi = seq.bootstrap_ci_mean(vals, args.bootstrap_iters, args.seed + int.from_bytes(__import__("hashlib").sha256(str(k).encode()).digest()[:8], "little") % 10007)
        ci95[k] = {"lo": lo, "hi": hi}

    summary = {
        "generated_at": now_utc_iso(),
        "dataset_count": len(ds_dirs),
        "folds_total": len(fold_rows),
        "folds_ok": len(ok),
        "label_field": b.TARGET_LABEL_FIELD,
        "model_name": args.model_name,
        "model_tag": args.model_tag,
        "prompt_style": args.prompt_style,
        "log_style": args.log_style,
        "max_success_false_stop_rate": float(args.max_success_false_stop_rate),
        "threshold_constraint_enabled": (not args.disable_threshold_constraint),
        "lambda_false_stop": float(args.lambda_false_stop),
        "inference_cost_per_row": float(args.inference_cost_per_row),
        "macro_mean": macro,
        "macro_mean_ci95_bootstrap": ci95,
    }
    with open(os.path.join(args.output_dir, "zs_seq_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

