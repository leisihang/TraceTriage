#!/usr/bin/env python3
"""
Run local Qwen zero-shot inference for Track-1 datasets and evaluate with
the same threshold protocol used by run_track1_baselines.py.
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
    # generated_ids: [B, T_generated]
    # attention_mask: [B, T_input] (1 for prompt tokens)
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
    pred_stop: List[int],
    label_field: str,
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
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            p = float(success_prob[i]) if i < len(success_prob) else 0.5
            s = int(pred_stop[i]) if i < len(pred_stop) else (1 if p < 0.5 else 0)
            w.writerow(
                {
                    "prefix_id": r.get("prefix_id", ""),
                    "record_key": r.get("record_key", ""),
                    "checkpoint": r.get("checkpoint", ""),
                    "strict_success": r.get("strict_success", ""),
                    "actionable_success": r.get("actionable_success", ""),
                    "eval_label_field": label_field,
                    "eval_label_value": r.get(label_field, ""),
                    "success_prob": f"{p:.6f}",
                    "pred_success": "1" if p >= 0.5 else "0",
                    "pred_stop": "1" if s == 1 else "0",
                    "final_wallclock_seconds": r.get("final_wallclock_seconds", ""),
                    "prefix_elapsed_seconds": r.get("prefix_elapsed_seconds", ""),
                    "remaining_seconds": r.get("remaining_seconds", ""),
                }
            )


def parse_platform_design_from_test(test_rows: List[Dict[str, str]]) -> Tuple[str, str]:
    if not test_rows:
        return "", ""
    return test_rows[0].get("platform", ""), test_rows[0].get("design", "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run local Qwen zero-shot evaluation on Track-1 datasets.")
    ap.add_argument("--dataset-dir-glob", action="append", required=True, help="Glob(s) for dataset dirs.")
    ap.add_argument("--target-dataset", action="append", default=[], help="Optional dataset basename(s).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-field", default="strict_success")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model-tag", default="Qwen2.5-0.5B-Instruct_zeroshot_weak_local")
    ap.add_argument("--prompt-style", choices=["basic", "weak", "strong", "expert"], default="weak")
    ap.add_argument("--log-style", choices=["basic", "enhanced"], default="basic")
    ap.add_argument("--threshold-metric", default="avg_weighted_net_saved_seconds_with_inference")
    ap.add_argument("--max-success-false-stop-rate", type=float, default=0.10)
    ap.add_argument("--lambda-false-stop", type=float, default=1.0)
    ap.add_argument("--disable-threshold-constraint", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-input-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--max-val-rows", type=int, default=0)
    ap.add_argument("--max-test-rows", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
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
        raise SystemExit("CUDA is required for local multi-GPU zero-shot run.")

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

    summary_rows: List[Dict[str, object]] = []

    for ds in ds_dirs:
        name = ds_name(ds)
        track1_dir = os.path.join(ds, "track1_strict")
        out_ds = os.path.join(args.output_dir, name)
        os.makedirs(out_ds, exist_ok=True)
        metrics_json = os.path.join(out_ds, "metrics_summary.json")
        pred_csv = os.path.join(out_ds, "predictions_qwen05b_zeroshot_weak_test.csv")

        if args.skip_existing and os.path.isfile(metrics_json) and os.path.isfile(pred_csv):
            with open(metrics_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
            mt = obj.get("metrics_test", {})
            summary_rows.append(
                {
                    "dataset": name,
                    "threshold": obj.get("threshold", 0.0),
                    "accuracy": mt.get("accuracy", 0.0),
                    "fail_recall_stop": mt.get("fail_recall_stop", 0.0),
                    "success_false_stop_rate": mt.get("success_false_stop_rate", 0.0),
                    "stop_rate": mt.get("stop_rate", 0.0),
                    "net_with_infer": mt.get("avg_weighted_net_saved_seconds_with_inference", 0.0),
                    "oracle": mt.get("oracle_avg_saved_seconds", 0.0),
                    "headroom_ratio_vs_oracle": mt.get("headroom_ratio_vs_oracle", 0.0),
                }
            )
            continue

        train_rows = b.load_rows(os.path.join(track1_dir, "splits", "train.csv"), require_binary_label=True, label_field=b.TARGET_LABEL_FIELD)
        val_rows = b.load_rows(os.path.join(track1_dir, "splits", "val.csv"), require_binary_label=True, label_field=b.TARGET_LABEL_FIELD)
        test_rows = b.load_rows(os.path.join(track1_dir, "splits", "test.csv"), require_binary_label=True, label_field=b.TARGET_LABEL_FIELD)
        if args.max_val_rows and args.max_val_rows > 0:
            val_rows = val_rows[: int(args.max_val_rows)]
        if args.max_test_rows and args.max_test_rows > 0:
            test_rows = test_rows[: int(args.max_test_rows)]

        if not val_rows or not test_rows:
            continue

        p_val, pred_stop_val, _det_val, val_avg = infer_rows_local(
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
        p_test, pred_stop_test, _det_test, test_avg = infer_rows_local(
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

        thr = b.choose_best_threshold(
            val_rows,
            p_val,
            metric=args.threshold_metric,
            max_success_false_stop_rate=float(args.max_success_false_stop_rate),
            enforce_success_false_stop_constraint=(not args.disable_threshold_constraint),
            lambda_false_stop=float(args.lambda_false_stop),
            inference_cost_per_row=0.0,
        )

        m_test = b.evaluate_predictions(
            test_rows,
            p_test,
            float(thr),
            lambda_false_stop=float(args.lambda_false_stop),
            inference_cost_per_row=0.0,
        )
        m_test = b.enrich_with_oracle_headroom(test_rows, m_test, lambda_false_stop=float(args.lambda_false_stop))

        write_predictions_csv(
            pred_csv,
            rows=test_rows,
            success_prob=p_test,
            pred_stop=pred_stop_test,
            label_field=b.TARGET_LABEL_FIELD,
        )

        platform, design = parse_platform_design_from_test(test_rows)
        doc = {
            "generated_at": now_utc_iso(),
            "track1_dir": os.path.abspath(track1_dir),
            "output_dir": os.path.abspath(out_ds),
            "label_field": b.TARGET_LABEL_FIELD,
            "model_name": args.model_tag,
            "hf_model_name": args.model_name,
            "prompt_style": args.prompt_style,
            "log_style": args.log_style,
            "threshold_metric": args.threshold_metric,
            "max_success_false_stop_rate": float(args.max_success_false_stop_rate),
            "lambda_false_stop": float(args.lambda_false_stop),
            "platform": platform,
            "design": design,
            "data_sizes": {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
            },
            "inference": {
                "avg_seconds_per_row_val": float(val_avg),
                "avg_seconds_per_row_test": float(test_avg),
                "batch_size": int(args.batch_size),
                "max_input_length": int(args.max_input_length),
                "max_new_tokens": int(args.max_new_tokens),
                "device": str(device),
                "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
            },
            "threshold": float(thr),
            "metrics_test": m_test,
        }
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

        summary_rows.append(
            {
                "dataset": name,
                "threshold": float(thr),
                "accuracy": float(m_test.get("accuracy", 0.0)),
                "fail_recall_stop": float(m_test.get("fail_recall_stop", 0.0)),
                "success_false_stop_rate": float(m_test.get("success_false_stop_rate", 0.0)),
                "stop_rate": float(m_test.get("stop_rate", 0.0)),
                "net_with_infer": float(m_test.get("avg_weighted_net_saved_seconds_with_inference", 0.0)),
                "oracle": float(m_test.get("oracle_avg_saved_seconds", 0.0)),
                "headroom_ratio_vs_oracle": float(m_test.get("headroom_ratio_vs_oracle", 0.0)),
            }
        )
        print(
            f"[zs-local] dataset={name} n_test={len(test_rows)} thr={float(thr):.2f} "
            f"net={float(m_test.get('avg_weighted_net_saved_seconds_with_inference', 0.0)):.6f}",
            flush=True,
        )

    sum_csv = os.path.join(args.output_dir, "zero_shot_summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "dataset",
            "threshold",
            "accuracy",
            "fail_recall_stop",
            "success_false_stop_rate",
            "stop_rate",
            "net_with_infer",
            "oracle",
            "headroom_ratio_vs_oracle",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    if summary_rows:
        def _mean(key: str) -> float:
            return float(sum(float(x.get(key, 0.0)) for x in summary_rows) / len(summary_rows))

        report = {
            "dataset_count": len(summary_rows),
            "stop_rate_zero_count": int(sum(1 for x in summary_rows if float(x.get("stop_rate", 0.0)) == 0.0)),
            "stop_rate_zero_ratio": float(sum(1 for x in summary_rows if float(x.get("stop_rate", 0.0)) == 0.0) / len(summary_rows)),
            "net_positive_count": int(sum(1 for x in summary_rows if float(x.get("net_with_infer", 0.0)) > 0.0)),
            "macro_mean": {
                "accuracy": _mean("accuracy"),
                "fail_recall_stop": _mean("fail_recall_stop"),
                "success_false_stop_rate": _mean("success_false_stop_rate"),
                "stop_rate": _mean("stop_rate"),
                "net_with_infer": _mean("net_with_infer"),
                "oracle": _mean("oracle"),
                "headroom_ratio_vs_oracle": _mean("headroom_ratio_vs_oracle"),
            },
            "summary_csv": sum_csv,
        }
        with open(os.path.join(args.output_dir, "zero_shot_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
