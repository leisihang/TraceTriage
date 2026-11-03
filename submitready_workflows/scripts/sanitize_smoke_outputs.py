#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any


def sanitize_value(v: Any, roots: list[str]) -> Any:
    if isinstance(v, dict):
        return {k: sanitize_value(x, roots) for k, x in v.items()}
    if isinstance(v, list):
        return [sanitize_value(x, roots) for x in v]
    if isinstance(v, str):
        s = v
        for r in roots:
            if r and r in s:
                s = s.replace(r, "$ROOT")
        home = str(Path.home())
        if home and home in s:
            s = s.replace(home, "$HOME")
        return s
    return v


def redact_sensitive_fields(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in {
                "api_key",
                "authorization",
                "x-api-key",
                "endpoint",
                "response_id",
                "api_base",
                "token",
                "access_token",
                "refresh_token",
                "secret",
            }:
                out[k] = "[REDACTED]"
                continue
            out[k] = redact_sensitive_fields(v)
        return out
    if isinstance(obj, list):
        return [redact_sensitive_fields(x) for x in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--repo-root", required=True)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    repo_root = str(Path(args.repo_root).resolve())
    roots = [repo_root, os.environ.get("ROOT", "")]

    for p in runs_root.rglob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        new_obj = sanitize_value(obj, roots)
        new_obj = redact_sensitive_fields(new_obj)
        p.write_text(json.dumps(new_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    for p in runs_root.rglob("*.jsonl"):
        lines = []
        changed = False
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                obj2 = sanitize_value(obj, roots)
                obj2 = redact_sensitive_fields(obj2)
                lines.append(json.dumps(obj2, ensure_ascii=False))
                changed = True
            except Exception:
                lines.append(ln)
        if changed:
            p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print("sanitized", str(runs_root))


if __name__ == "__main__":
    main()
