#!/usr/bin/env python3
"""
API-only LLM runner for Track-1 split CSV.
- No xgboost/sklearn/numpy dependencies
- Only calls LLM APIs and writes predictions/summary artifacts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def maybe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_prefix_metrics(metrics_json: str) -> Dict[str, Optional[float]]:
    if not metrics_json:
        return {}
    try:
        obj = json.loads(metrics_json)
    except Exception:
        return {}
    out: Dict[str, Optional[float]] = {}
    for k, v in obj.items():
        out[k] = maybe_float(v)
    return out


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
        out[k] = None if v is None else float(round(v, 6))
    return out


def compact_run_params_with_missing_for_llm(row: Dict[str, str]) -> Dict[str, Optional[str]]:
    keys = list(RUN_PARAM_NUMERIC_KEYS) + list(RUN_PARAM_BOOL_KEYS) + ["pin_layer", "above_layer"]
    out: Dict[str, Optional[str]] = {}
    for k in keys:
        v = str(row.get(k, "") or "").strip()
        if v:
            out[k] = v
            out[f"{k}_missing"] = "0"
        else:
            out[k] = None
            out[f"{k}_missing"] = "1"
    return out


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def select_rows_deterministic(rows: List[Dict[str, str]], max_samples: int, seed: int) -> List[Dict[str, str]]:
    out = rows[:]
    random.Random(seed).shuffle(out)
    if max_samples > 0:
        out = out[: max_samples]
    return out


def resolve_bundle_path(row: Dict[str, str], path: str) -> str:
    path = str(path or "").strip()
    if not path:
        return ""
    if os.path.isfile(path):
        return path

    package_root = str(row.get("__package_root", "")).strip()
    track1_dir = str(row.get("__track1_dir", "")).strip()
    cands: List[str] = []
    if not os.path.isabs(path):
        if package_root:
            cands.append(os.path.join(package_root, path))
        if track1_dir:
            cands.extend(
                [
                    os.path.join(track1_dir, path),
                    os.path.join(os.path.dirname(track1_dir), path),
                    os.path.join(os.path.dirname(os.path.dirname(track1_dir)), path),
                ]
            )
    for cand in cands:
        if os.path.isfile(cand):
            return cand
    return path


def log_excerpt_for_llm(row: Dict[str, str], enhanced: bool = False, max_chars: int = 1800) -> str:
    excerpt = str(row.get("prefix_log_excerpt", "")).strip()
    if excerpt:
        return excerpt[:max_chars]

    path = str(row.get("prefix_log", "")).strip() or str(row.get("prefix_log_path", "")).strip()
    if not path:
        return ""
    path = resolve_bundle_path(row, path)
    if not os.path.isfile(path):
        return ""

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return ""

    if enhanced:
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
        text = "\n".join(merged) if merged else ("".join(lines[:20]) + "\n...\n" + "".join(lines[-40:])).strip()
        return text[:max_chars]

    pat = re.compile(r"(WARN|ERROR|FATAL|violation|overflow|congestion|slack|WNS|TNS|DRC)", re.IGNORECASE)
    keep = [ln.strip() for ln in lines if pat.search(ln)]
    if not keep:
        text = ("".join(lines[:20]) + "\n...\n" + "".join(lines[-20:])).strip()
        return text[:max_chars]
    return "\n".join(keep[-80:])[:max_chars]


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


def build_llm_prompt(row: Dict[str, str], prompt_style: str, log_style: str, stage1_json: str = "") -> str:
    cp = row.get("checkpoint", "")
    metrics_raw = row.get("prefix_metrics", "{}")
    metrics_small = compact_metrics_for_llm(metrics_raw)
    platform = row.get("platform", "")
    design = row.get("design", "")
    objective = row.get("objective_hint", "")
    psec = row.get("prefix_elapsed_seconds", "")
    run_params = compact_run_params_with_missing_for_llm(row)

    log_excerpt = log_excerpt_for_llm(row=row, enhanced=(log_style == "enhanced"), max_chars=(2800 if log_style == "enhanced" else 1800))
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
        stage1_block = f"Stage-1 Risk Extraction JSON:\n{stage1_json}\n\n" if stage1_json else ""
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
        stage1_block = f"Stage-1 Risk Extraction JSON:\n{stage1_json}\n\n" if stage1_json else ""
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
        f"Constraints: risk_tags length >= {max(0, int(min_risk_tags))}, evidence length >= {max(0, int(min_evidence))}.\n"
        "Each evidence item must be a concise, log-grounded phrase.\n"
    )


def _default_api_base(provider: str) -> str:
    p = str(provider or "deepseek").strip().lower()
    if p == "openai":
        return "https://api.openai.com"
    if p == "anthropic":
        return "https://api.anthropic.com/v1/messages"
    return "https://api.deepseek.com"


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _normalize_usage(provider: str, usage_obj: Any) -> Dict[str, int]:
    p = str(provider or "").strip().lower()
    if not isinstance(usage_obj, dict):
        return {}
    out: Dict[str, int] = {}
    if p == "anthropic":
        # Anthropic style usage keys.
        inp = _safe_int(usage_obj.get("input_tokens", 0))
        outp = _safe_int(usage_obj.get("output_tokens", 0))
        out["prompt_tokens"] = inp
        out["completion_tokens"] = outp
        out["total_tokens"] = inp + outp
        return out
    # OpenAI-compatible usage keys (DeepSeek/OpenAI chat endpoints).
    for k in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "reasoning_tokens",
    ):
        if k in usage_obj:
            out[k] = _safe_int(usage_obj.get(k, 0))
    # Some providers may use input/output token naming.
    if "prompt_tokens" not in out and "input_tokens" in usage_obj:
        out["prompt_tokens"] = _safe_int(usage_obj.get("input_tokens", 0))
    if "completion_tokens" not in out and "output_tokens" in usage_obj:
        out["completion_tokens"] = _safe_int(usage_obj.get("output_tokens", 0))
    if "total_tokens" not in out:
        out["total_tokens"] = out.get("prompt_tokens", 0) + out.get("completion_tokens", 0)
    return out


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
    return_meta: bool = False,
) -> Any:
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
                        "content": [{"type": "input_text", "text": "You are an EDA triage assistant. Reply with strict JSON only."}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
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
            effort = str(openai_reasoning_effort or "").strip()
            if effort:
                # Some OpenAI-compatible gateways read either key.
                payload["reasoning_effort"] = effort
                payload["model_reasoning_effort"] = effort

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                # Some API gateways block urllib's default Python user agent.
                "User-Agent": "python-requests/2.31.0",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw_text = resp.read().decode("utf-8", errors="ignore")
            obj = json.loads(raw_text)

        meta: Dict[str, Any] = {
            "provider": p,
            "endpoint": "[redacted]",
            "usage": _normalize_usage(p, obj.get("usage", {})),
            "response_model": str(obj.get("model", "")),
            "response_id": "[redacted]",
        }
        if p == "openai" and wire_api == "responses":
            out_txt = obj.get("output_text")
            if isinstance(out_txt, str) and out_txt.strip():
                if return_meta:
                    return out_txt, meta
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
                    joined = "\n".join(buf)
                    if return_meta:
                        return joined, meta
                    return joined
            raise KeyError(f"invalid_responses_output_text:{str(obj)[:500]}")

        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            raise KeyError(f"invalid_chat_response_missing_choices:{str(obj)[:500]}")
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        if content is None:
            raise KeyError(f"invalid_chat_response_missing_content:{str(obj)[:500]}")
        content_s = str(content)
        if return_meta:
            return content_s, meta
        return content_s

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
                # Some API gateways block urllib's default Python user agent.
                "User-Agent": "python-requests/2.31.0",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw_text = resp.read().decode("utf-8", errors="ignore")
            obj = json.loads(raw_text)
        meta = {
            "provider": p,
            "endpoint": "[redacted]",
            "usage": _normalize_usage(p, obj.get("usage", {})),
            "response_model": str(obj.get("model", "")),
            "response_id": "[redacted]",
        }
        content = obj.get("content", [])
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    txt = str(blk.get("text", ""))
                    if return_meta:
                        return txt, meta
                    return txt
        raise KeyError("anthropic content.text not found")

    raise ValueError(f"Unsupported llm provider: {provider}")


def llm_chat_completion_with_retry(
    api_key: str,
    prompt: str,
    model: str,
    provider: str,
    api_base: str,
    openai_wire_api: str,
    openai_reasoning_effort: str,
    anthropic_version: str,
    timeout_s: int,
    max_retries: int,
    retry_base_sleep_s: float,
    audit_jsonl: str = "",
    audit_base: Optional[Dict[str, Any]] = None,
) -> str:
    for attempt in range(max(1, int(max_retries))):
        t0 = time.time()
        try:
            content, meta = llm_chat_completion(
                api_key=api_key,
                prompt=prompt,
                model=model,
                provider=provider,
                api_base=api_base,
                openai_wire_api=openai_wire_api,
                openai_reasoning_effort=openai_reasoning_effort,
                anthropic_version=anthropic_version,
                timeout_s=timeout_s,
                return_meta=True,
            )
            if audit_jsonl:
                rec = dict(audit_base or {})
                rec.update(
                    {
                        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "event": "api_call",
                        "status": "ok",
                        "attempt": int(attempt + 1),
                        "elapsed_ms": int((time.time() - t0) * 1000),
                        "prompt_chars": len(prompt or ""),
                        "prompt_sha1": hashlib.sha1((prompt or "").encode("utf-8", errors="ignore")).hexdigest(),
                        "output_chars": len(content or ""),
                        "usage": dict(meta.get("usage", {})) if isinstance(meta, dict) else {},
                        "endpoint": "[redacted]",
                        "response_id": "[redacted]",
                        "response_model": str(meta.get("response_model", "")) if isinstance(meta, dict) else "",
                    }
                )
                append_jsonl(audit_jsonl, rec)
            return str(content)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            ValueError,
            http.client.HTTPException,
            OSError,
        ) as e:
            if audit_jsonl:
                rec = dict(audit_base or {})
                rec.update(
                    {
                        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "event": "api_call",
                        "status": "error",
                        "attempt": int(attempt + 1),
                        "elapsed_ms": int((time.time() - t0) * 1000),
                        "prompt_chars": len(prompt or ""),
                        "prompt_sha1": hashlib.sha1((prompt or "").encode("utf-8", errors="ignore")).hexdigest(),
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                )
                append_jsonl(audit_jsonl, rec)
            if attempt + 1 >= max(1, int(max_retries)):
                raise
            time.sleep(float(retry_base_sleep_s) * (2 ** attempt))
    raise RuntimeError("unreachable")


def summarize_api_audit(audit_jsonl: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "api_call_total": 0,
        "api_call_ok": 0,
        "api_call_error": 0,
        "prompt_chars_total": 0,
        "output_chars_total": 0,
        "elapsed_ms_total": 0,
        "usage_prompt_tokens_total": 0,
        "usage_completion_tokens_total": 0,
        "usage_total_tokens_total": 0,
        "calls_with_usage": 0,
        "call_kind_counts": {},
    }
    if not audit_jsonl or not os.path.isfile(audit_jsonl):
        return out

    kind_counts: Dict[str, int] = {}
    with open(audit_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if obj.get("event") != "api_call":
                continue
            out["api_call_total"] += 1
            st = str(obj.get("status", "")).lower()
            if st == "ok":
                out["api_call_ok"] += 1
            elif st == "error":
                out["api_call_error"] += 1
            out["prompt_chars_total"] += _safe_int(obj.get("prompt_chars", 0))
            out["output_chars_total"] += _safe_int(obj.get("output_chars", 0))
            out["elapsed_ms_total"] += _safe_int(obj.get("elapsed_ms", 0))

            k = str(obj.get("call_kind", "")).strip() or "unknown"
            kind_counts[k] = kind_counts.get(k, 0) + 1

            usage = obj.get("usage", {})
            if isinstance(usage, dict) and usage:
                out["calls_with_usage"] += 1
                out["usage_prompt_tokens_total"] += _safe_int(usage.get("prompt_tokens", 0))
                out["usage_completion_tokens_total"] += _safe_int(usage.get("completion_tokens", 0))
                out["usage_total_tokens_total"] += _safe_int(usage.get("total_tokens", 0))

    out["call_kind_counts"] = kind_counts
    return out


def _json_candidates_from_text(raw_content: str) -> List[Tuple[str, str]]:
    text = (raw_content or "").strip()
    if not text:
        return []
    candidates: List[Tuple[str, str]] = [("raw", text)]

    if text.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text, count=1)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
        stripped = stripped.strip()
        if stripped:
            candidates.append(("fenced", stripped))

    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        obj_slice = text[i : j + 1].strip()
        if obj_slice:
            candidates.append(("slice", obj_slice))

    seen = set()
    out: List[Tuple[str, str]] = []
    for mode, cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        out.append((mode, cand))
    return out


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

        risk_tags = _normalize_str_list(obj.get("risk_tags", obj.get("reason_codes", obj.get("reasons", []))))
        evidence = _normalize_str_list(obj.get("evidence", obj.get("evidence_spans", obj.get("cited_log_spans", []))))

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


def patch_llm_rationale(details: Dict[str, Any], row: Dict[str, str], min_risk_tags: int, min_evidence: int) -> Dict[str, Any]:
    tags = _normalize_str_list(details.get("risk_tags", []))
    evidence = _normalize_str_list(details.get("evidence", []))
    while len(tags) < max(0, int(min_risk_tags)):
        tags.append("unspecified_risk")
    cp = str(row.get("checkpoint", ""))
    pid = str(row.get("prefix_id", ""))
    while len(evidence) < max(0, int(min_evidence)):
        evidence.append(f"fallback_evidence:{cp}:{pid}")
    details["risk_tags"] = tags
    details["evidence"] = evidence
    pm = str(details.get("parse_mode", ""))
    details["parse_mode"] = f"{pm}|rationale_patched" if pm else "rationale_patched"
    return details


def is_fallback_parse_mode(parse_mode: Any) -> bool:
    pm = str(parse_mode or "").strip().lower()
    if not pm:
        return True
    # Synthetic/default outputs that are not purely model-grounded results.
    fallback_markers = (
        "api_error_default",
        "regex_fallback",
        "rationale_patched",
    )
    return any(m in pm for m in fallback_markers)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    if not path:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_cache(path: str) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sig = f"{obj.get('prefix_id','')}|{obj.get('cfg','default')}"
            cache[sig] = obj
    return cache


def row_remaining_seconds(row: Dict[str, str]) -> float:
    p = maybe_float(row.get("prefix_elapsed_seconds", ""))
    f = maybe_float(row.get("final_wallclock_seconds", ""))
    if p is None or f is None:
        return 0.0
    return max(0.0, f - p)


def evaluate_api_predictions(rows: List[Dict[str, str]], pred_stop: List[int], success_prob: List[float], lambda_false_stop: float) -> Dict[str, Any]:
    n = min(len(rows), len(pred_stop), len(success_prob))
    if n <= 0:
        return {
            "n": 0,
            "stop_rate": 0.0,
            "accuracy": 0.0,
            "fail_recall_stop": 0.0,
            "success_false_stop_rate": 0.0,
            "avg_weighted_net_saved_seconds": 0.0,
        }

    tot_stop = 0
    acc = 0
    fail_total = 0
    fail_stop = 0
    succ_total = 0
    succ_false_stop = 0
    net_sum = 0.0

    for i in range(n):
        r = rows[i]
        y = 1 if str(r.get("strict_success", "")).strip() == "1" else 0
        s = 1 if int(pred_stop[i]) == 1 else 0
        pred_success = 0 if s == 1 else 1

        tot_stop += s
        if pred_success == y:
            acc += 1
        if y == 0:
            fail_total += 1
            if s == 1:
                fail_stop += 1
        else:
            succ_total += 1
            if s == 1:
                succ_false_stop += 1

        rem = row_remaining_seconds(r)
        if s == 1 and y == 0:
            net_sum += rem
        elif s == 1 and y == 1:
            net_sum -= (lambda_false_stop * rem)

    return {
        "n": n,
        "stop_rate": float(tot_stop / n),
        "accuracy": float(acc / n),
        "fail_recall_stop": float(fail_stop / fail_total) if fail_total else 0.0,
        "success_false_stop_rate": float(succ_false_stop / succ_total) if succ_total else 0.0,
        "avg_weighted_net_saved_seconds": float(net_sum / n),
    }


def run_api_only(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)

    if args.input_csv:
        csv_path = args.input_csv
    else:
        csv_path = os.path.join(args.track1_dir, "splits", f"{args.split}.csv")

    if not os.path.isfile(csv_path):
        raise SystemExit(f"Missing input CSV: {csv_path}")

    rows = load_rows(csv_path)
    sample_rows = select_rows_deterministic(rows, max_samples=args.max_samples, seed=args.seed)
    for r in sample_rows:
        r["__track1_dir"] = args.track1_dir
        r["__package_root"] = args.package_root

    cache_jsonl = os.path.join(args.output_dir, "llm_cache.jsonl")
    request_status_jsonl = os.path.join(args.output_dir, "llm_request_status.jsonl")
    api_audit_jsonl = os.path.join(args.output_dir, "llm_api_calls.jsonl")
    cache = {} if args.ignore_cache else load_cache(cache_jsonl)

    pred_stop_all: List[int] = []
    success_prob_all: List[float] = []
    details_all: List[Dict[str, Any]] = []
    strict_failures: List[Tuple[str, str]] = []
    cache_hits = 0

    # Keep rationale controls consistent with baseline behavior.
    effective_require_rationale = bool(args.require_rationale)
    effective_min_risk_tags = int(args.min_risk_tags) if effective_require_rationale else 0
    effective_min_evidence = int(args.min_evidence) if effective_require_rationale else 0
    effective_rationale_retry = int(args.rationale_retry) if effective_require_rationale else 0

    cfg_sig = (
        f"{args.prompt_style}|{args.log_style}|{1 if args.two_stage else 0}"
        f"|rr={1 if effective_require_rationale else 0}|rt={effective_min_risk_tags}|ev={effective_min_evidence}"
    )

    with open(cache_jsonl, "a", encoding="utf-8") as fw:
        for i, row in enumerate(sample_rows, start=1):
            pid = str(row.get("prefix_id", ""))
            cache_sig = f"{pid}|{cfg_sig}"

            append_jsonl(
                request_status_jsonl,
                {
                    "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "event": "row_begin",
                    "index": i,
                    "total": len(sample_rows),
                    "split": args.split,
                    "prefix_id": pid,
                },
            )

            if cache_sig in cache:
                c = cache[cache_sig]
                if bool(args.real_results_only) and is_fallback_parse_mode(c.get("parse_mode", "")):
                    # Skip synthetic/default cache rows in real-only mode.
                    pass
                else:
                    pred_stop = int(c.get("pred_stop", 0))
                    success_prob = float(c.get("success_prob", 0.5))
                    detail = {
                        "decision": ("STOP" if pred_stop == 1 else "CONTINUE"),
                        "confidence": float(c.get("confidence", (1.0 - success_prob if pred_stop == 1 else success_prob))),
                        "risk_tags": _normalize_str_list(c.get("risk_tags", [])),
                        "evidence": _normalize_str_list(c.get("evidence", [])),
                        "parse_mode": str(c.get("parse_mode", "cache")),
                        "source": "cache",
                        "raw_content": str(c.get("raw_content", "")),
                    }
                    pred_stop_all.append(pred_stop)
                    success_prob_all.append(success_prob)
                    details_all.append(detail)
                    cache_hits += 1

                    append_jsonl(
                        request_status_jsonl,
                        {
                            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                            "event": "cache_hit_used",
                            "index": i,
                            "total": len(sample_rows),
                            "split": args.split,
                            "prefix_id": pid,
                        },
                    )
                    continue

            stage1_text = ""
            if args.two_stage:
                stage1_prompt = (
                    "Extract structured risk information for this EDA checkpoint.\n"
                    "Return STRICT JSON only with keys:\n"
                    "{\"risk_score\":0.0-1.0,\"timing_risk\":0.0-1.0,\"routing_risk\":0.0-1.0,"
                    "\"hard_fail_signals\":[\"...\"],\"summary\":\"...\"}\n\n"
                    f"Checkpoint: {row.get('checkpoint','')}\n"
                    f"Metrics Compact JSON: {json.dumps(compact_metrics_for_llm(row.get('prefix_metrics','{}')), ensure_ascii=False)}\n"
                    f"Log Excerpt:\n{log_excerpt_for_llm(row, enhanced=(args.log_style=='enhanced'), max_chars=2200)}\n"
                )
                try:
                    stage1_audit_base = {
                        "split": args.split,
                        "prefix_id": pid,
                        "index": i,
                        "call_kind": "stage1",
                        "provider": args.provider,
                        "model": args.model,
                    }
                    stage1_text = llm_chat_completion_with_retry(
                        api_key=args.api_key,
                        prompt=stage1_prompt,
                        model=args.model,
                        provider=args.provider,
                        api_base=args.api_base,
                        openai_wire_api=args.openai_wire_api,
                        openai_reasoning_effort=args.openai_reasoning_effort,
                        anthropic_version=args.anthropic_version,
                        timeout_s=args.timeout_s,
                        max_retries=args.api_max_retries,
                        retry_base_sleep_s=args.api_retry_base_s,
                        audit_jsonl=api_audit_jsonl,
                        audit_base=stage1_audit_base,
                    )
                except Exception:
                    stage1_text = ""

            prompt = build_llm_prompt(row=row, prompt_style=args.prompt_style, log_style=args.log_style, stage1_json=stage1_text)

            try:
                append_jsonl(
                    request_status_jsonl,
                    {
                        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "event": "request_start",
                        "index": i,
                        "total": len(sample_rows),
                        "split": args.split,
                        "prefix_id": pid,
                        "provider": args.provider,
                        "model": args.model,
                    },
                )

                content = llm_chat_completion_with_retry(
                    api_key=args.api_key,
                    prompt=prompt,
                    model=args.model,
                    provider=args.provider,
                    api_base=args.api_base,
                    openai_wire_api=args.openai_wire_api,
                    openai_reasoning_effort=args.openai_reasoning_effort,
                    anthropic_version=args.anthropic_version,
                    timeout_s=args.timeout_s,
                    max_retries=args.api_max_retries,
                    retry_base_sleep_s=args.api_retry_base_s,
                    audit_jsonl=api_audit_jsonl,
                    audit_base={
                        "split": args.split,
                        "prefix_id": pid,
                        "index": i,
                        "call_kind": "main",
                        "provider": args.provider,
                        "model": args.model,
                    },
                )
                pred_stop, success_prob, det = parse_llm_payload(content)
                if bool(args.real_results_only):
                    pm = str(det.get("parse_mode", ""))
                    if not pm.startswith("json"):
                        raise ValueError(f"non_json_parse_mode:{pm}")
                    if is_fallback_parse_mode(pm):
                        raise ValueError(f"fallback_parse_mode:{pm}")

                if effective_require_rationale and not llm_has_min_rationale(
                    det, min_risk_tags=effective_min_risk_tags, min_evidence=effective_min_evidence
                ):
                    retry_prompt = build_rationale_retry_prompt(
                        prompt, min_risk_tags=effective_min_risk_tags, min_evidence=effective_min_evidence
                    )
                    for _ in range(max(0, effective_rationale_retry)):
                        try:
                            content_retry = llm_chat_completion_with_retry(
                                api_key=args.api_key,
                                prompt=retry_prompt,
                                model=args.model,
                                provider=args.provider,
                                api_base=args.api_base,
                                openai_wire_api=args.openai_wire_api,
                                openai_reasoning_effort=args.openai_reasoning_effort,
                                anthropic_version=args.anthropic_version,
                                timeout_s=args.timeout_s,
                                max_retries=args.api_max_retries,
                                retry_base_sleep_s=args.api_retry_base_s,
                                audit_jsonl=api_audit_jsonl,
                                audit_base={
                                    "split": args.split,
                                    "prefix_id": pid,
                                    "index": i,
                                    "call_kind": "rationale_retry",
                                    "provider": args.provider,
                                    "model": args.model,
                                },
                            )
                            pred2, prob2, det2 = parse_llm_payload(content_retry)
                            if llm_has_min_rationale(
                                det2, min_risk_tags=effective_min_risk_tags, min_evidence=effective_min_evidence
                            ):
                                pred_stop, success_prob, det = pred2, prob2, det2
                                break
                        except Exception:
                            continue
                    if not llm_has_min_rationale(
                        det, min_risk_tags=effective_min_risk_tags, min_evidence=effective_min_evidence
                    ):
                        if bool(args.real_results_only):
                            raise ValueError("missing_rationale_after_retry")
                        det = patch_llm_rationale(
                            det, row=row, min_risk_tags=effective_min_risk_tags, min_evidence=effective_min_evidence
                        )

                append_jsonl(
                    request_status_jsonl,
                    {
                        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "event": "request_ok",
                        "index": i,
                        "total": len(sample_rows),
                        "split": args.split,
                        "prefix_id": pid,
                        "decision": str(det.get("decision", "STOP" if pred_stop == 1 else "CONTINUE")),
                        "confidence": _float_in_01(det.get("confidence", (1.0 - success_prob if pred_stop == 1 else success_prob))),
                        "parse_mode": str(det.get("parse_mode", "unknown")),
                    },
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
                        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "event": "request_error",
                        "index": i,
                        "total": len(sample_rows),
                        "split": args.split,
                        "prefix_id": pid,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "strict_api_errors": bool(args.strict_api_errors),
                    },
                )
                if args.strict_api_errors:
                    strict_failures.append((pid, str(e)))
                    continue
                if bool(args.real_results_only):
                    strict_failures.append((pid, str(e)))
                    continue
                pred_stop, success_prob = 0, 0.5
                det = {
                    "decision": "CONTINUE",
                    "confidence": 0.5,
                    "risk_tags": [],
                    "evidence": [],
                    "parse_mode": "api_error_default",
                    "raw_content": "",
                }
                if effective_require_rationale:
                    det = patch_llm_rationale(
                        det, row=row, min_risk_tags=effective_min_risk_tags, min_evidence=effective_min_evidence
                    )

            pred_stop_all.append(pred_stop)
            success_prob_all.append(success_prob)
            detail = {
                "decision": str(det.get("decision", "STOP" if pred_stop == 1 else "CONTINUE")),
                "confidence": _float_in_01(det.get("confidence", (1.0 - success_prob if pred_stop == 1 else success_prob))),
                "risk_tags": _normalize_str_list(det.get("risk_tags", [])),
                "evidence": _normalize_str_list(det.get("evidence", [])),
                "parse_mode": str(det.get("parse_mode", "unknown")),
                "source": "live",
                "raw_content": str(det.get("raw_content", "")),
            }
            details_all.append(detail)

            rec = {
                "prefix_id": pid,
                "cfg": cfg_sig,
                "pred_stop": pred_stop,
                "success_prob": success_prob,
                "decision": detail["decision"],
                "confidence": detail["confidence"],
                "risk_tags": detail["risk_tags"],
                "evidence": detail["evidence"],
                "parse_mode": detail["parse_mode"],
                "raw_content": detail["raw_content"][:4000],
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fw.flush()

            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

            if i % 50 == 0:
                print(f"[llm-api-only] processed={i}/{len(sample_rows)} cache_hits={cache_hits}", flush=True)

    if args.strict_api_errors and strict_failures:
        preview = "; ".join([f"{pid}: {reason}" for pid, reason in strict_failures[:3]])
        raise RuntimeError(f"llm_api_failed_count={len(strict_failures)} sampled={len(sample_rows)} first_failures={preview}")

    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    pred_csv = os.path.join(args.output_dir, "llm_predictions.csv")
    pred_fields = [
        "prefix_id",
        "platform",
        "design",
        "checkpoint",
        "strict_success",
        "pred_stop",
        "pred_success",
        "success_prob",
        "decision",
        "confidence",
        "parse_mode",
        "risk_tags",
        "evidence",
        "raw_content_sha1",
    ]

    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pred_fields)
        w.writeheader()
        for r, s, p, d in zip(sample_rows, pred_stop_all, success_prob_all, details_all):
            raw = str(d.get("raw_content", ""))
            w.writerow(
                {
                    "prefix_id": r.get("prefix_id", ""),
                    "platform": r.get("platform", ""),
                    "design": r.get("design", ""),
                    "checkpoint": r.get("checkpoint", ""),
                    "strict_success": r.get("strict_success", ""),
                    "pred_stop": int(s),
                    "pred_success": (0 if int(s) == 1 else 1),
                    "success_prob": float(p),
                    "decision": d.get("decision", ""),
                    "confidence": d.get("confidence", 0.5),
                    "parse_mode": d.get("parse_mode", ""),
                    "risk_tags": "|".join(_normalize_str_list(d.get("risk_tags", []))),
                    "evidence": " || ".join(_normalize_str_list(d.get("evidence", []))),
                    "raw_content_sha1": hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest() if raw else "",
                }
            )

    parse_counter = Counter([str(d.get("parse_mode", "unknown")) for d in details_all])
    eval_metrics = evaluate_api_predictions(sample_rows, pred_stop_all, success_prob_all, lambda_false_stop=float(args.lambda_false_stop))
    api_audit_summary = summarize_api_audit(api_audit_jsonl)

    result = {
        "generated_at": ts,
        "mode": "api_only",
        "input": {
            "track1_dir": args.track1_dir,
            "input_csv": csv_path,
            "split": args.split,
            "total_rows_in_csv": len(rows),
            "sampled_rows": len(sample_rows),
            "seed": args.seed,
            "max_samples": args.max_samples,
        },
        "api": {
            "provider": args.provider,
            "model": args.model,
            "api_base": "[redacted]",
            "openai_wire_api": args.openai_wire_api,
            "openai_reasoning_effort": args.openai_reasoning_effort,
            "anthropic_version": args.anthropic_version,
            "timeout_s": args.timeout_s,
            "api_max_retries": args.api_max_retries,
            "api_retry_base_s": args.api_retry_base_s,
            "strict_api_errors": bool(args.strict_api_errors),
        },
        "prompt": {
            "prompt_style": args.prompt_style,
            "log_style": args.log_style,
            "two_stage": bool(args.two_stage),
            "require_rationale": bool(effective_require_rationale),
            "min_risk_tags": int(effective_min_risk_tags),
            "min_evidence": int(effective_min_evidence),
            "rationale_retry": int(effective_rationale_retry),
        },
        "cache": {
            "enabled": not bool(args.ignore_cache),
            "cache_hits": int(cache_hits),
        },
        "metrics": eval_metrics,
        "api_audit": api_audit_summary,
        "parse_mode_counts": dict(parse_counter),
        "artifacts": {
            "predictions_csv": pred_csv,
            "cache_jsonl": cache_jsonl,
            "request_status_jsonl": request_status_jsonl,
            "api_calls_jsonl": api_audit_jsonl,
        },
    }

    summary_json = os.path.join(args.output_dir, "llm_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="API-only LLM runner (no xgboost/sklearn/numpy).")
    ap.add_argument("--track1-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--package-root",
        default="",
        help="Dataset package root used to resolve relative logs/openroad/... references.",
    )
    ap.add_argument("--input-csv", default="")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")

    ap.add_argument("--provider", choices=["deepseek", "openai", "anthropic"], default="deepseek")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--api-base", default="")
    ap.add_argument("--openai-wire-api", choices=["chat_completions", "responses"], default="chat_completions")
    ap.add_argument("--openai-reasoning-effort", default="")
    ap.add_argument("--anthropic-version", default="2023-06-01")

    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep-s", type=float, default=0.0)
    ap.add_argument("--timeout-s", type=int, default=60)

    ap.add_argument("--prompt-style", choices=["basic", "weak", "strong", "expert"], default="basic")
    ap.add_argument("--log-style", choices=["basic", "enhanced"], default="basic")
    ap.add_argument("--two-stage", action="store_true")

    ap.add_argument("--require-rationale", action="store_true")
    ap.add_argument("--min-risk-tags", type=int, default=1)
    ap.add_argument("--min-evidence", type=int, default=1)
    ap.add_argument("--rationale-retry", type=int, default=1)

    ap.add_argument("--api-max-retries", type=int, default=4)
    ap.add_argument("--api-retry-base-s", type=float, default=0.8)
    ap.add_argument("--strict-api-errors", action="store_true")
    ap.add_argument("--ignore-cache", action="store_true")
    ap.add_argument(
        "--real-results-only",
        action="store_true",
        help=(
            "Only keep model-grounded outputs: skip fallback cache rows, disallow non-JSON/regex fallback, "
            "disallow api_error_default, and disallow rationale_patched synthesis."
        ),
    )

    ap.add_argument("--lambda-false-stop", type=float, default=1.0)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_api_only(args)


if __name__ == "__main__":
    main()
