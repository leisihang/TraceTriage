#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_rows(path: Path, max_rows: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def label(row: Dict[str, str], field: str) -> int:
    return 1 if str(row.get(field, "")).strip().lower() in {"1", "true", "yes"} else 0


def metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    correct = sum(1 for y, p in zip(y_true, y_pred) if y == p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": float(n),
        "accuracy": correct / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def iter_dataset_dirs(package_root: Path, target_dataset: str) -> Iterable[Path]:
    data_root = package_root / "data"
    for ds in sorted(data_root.iterdir()):
        if not ds.is_dir():
            continue
        if target_dataset and ds.name != target_dataset:
            continue
        track = ds / "track1_strict"
        if track.is_dir():
            yield ds


def write_predictions(path: Path, rows: List[Dict[str, str]], y_pred: List[int], field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["prefix_id", "split", "label_field", "y_true", "y_pred", "score"],
        )
        w.writeheader()
        for row, pred in zip(rows, y_pred):
            w.writerow(
                {
                    "prefix_id": row.get("prefix_id", ""),
                    "split": row.get("split", ""),
                    "label_field": field,
                    "y_true": label(row, field),
                    "y_pred": pred,
                    "score": float(pred),
                }
            )


def run_one(ds: Path, out_root: Path, label_field: str, max_rows: int) -> Tuple[str, Dict[str, object]]:
    track = ds / "track1_strict"
    train = read_rows(track / "splits" / "train.csv", max_rows)
    val = read_rows(track / "splits" / "val.csv", max_rows)
    test = read_rows(track / "splits" / "test.csv", max_rows)

    train_labels = [label(r, label_field) for r in train]
    majority = 1 if sum(train_labels) >= (len(train_labels) / 2) else 0

    ds_out = out_root / ds.name
    result: Dict[str, object] = {
        "dataset": ds.name,
        "model": "stdlib_majority_class",
        "label_field": label_field,
        "max_rows_per_split": max_rows,
        "train_rows": len(train),
        "majority_class": majority,
    }

    for split, rows in [("val", val), ("test", test)]:
        y_true = [label(r, label_field) for r in rows]
        y_pred = [majority for _ in rows]
        result[f"{split}_metrics"] = metrics(y_true, y_pred)
        write_predictions(ds_out / f"predictions_{split}.csv", rows, y_pred, label_field)

    (ds_out / "metrics_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ds.name, result


def main() -> None:
    ap = argparse.ArgumentParser(description="Standard-library baseline smoke test")
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-dataset", default="asap7_gcd")
    ap.add_argument("--label-field", default="strict_success")
    ap.add_argument("--max-rows-per-split", type=int, default=50)
    args = ap.parse_args()

    package_root = Path(args.package_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, object] = {}
    for ds in iter_dataset_dirs(package_root, args.target_dataset):
        name, result = run_one(ds, out_root, args.label_field, args.max_rows_per_split)
        results[name] = result

    if not results:
        raise SystemExit(f"no datasets matched target_dataset={args.target_dataset!r}")

    summary = {
        "package_root": str(package_root),
        "output_dir": str(out_root),
        "target_dataset": args.target_dataset,
        "datasets": results,
    }
    (out_root / "stdlib_baseline_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
