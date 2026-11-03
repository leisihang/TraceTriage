#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


EXPECTED_DATASETS = {
    "asap7_aes_cipher_top",
    "asap7_gcd",
    "asap7_ibex_core",
    "asap7_jpeg_encoder",
    "asap7_picorv32",
    "sky130hd_aes_cipher_top",
    "sky130hd_gcd",
    "sky130hd_ibex_core",
    "sky130hd_jpeg_encoder",
    "sky130hd_picorv32",
}


def canonical_split(v: str) -> str:
    s = (v or "").strip().lower()
    if s in ("train", "tr"):
        return "train"
    if s in ("val", "valid", "validation", "dev"):
        return "val"
    if s in ("test", "te"):
        return "test"
    return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_concat(paths: List[Path], package_root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x.relative_to(package_root)).replace("\\", "/")):
        h.update(sha256_file(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def sha256_raw_concat(paths: List[Path], package_root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x.relative_to(package_root)).replace("\\", "/")):
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def contains_abs_hint(v: str) -> bool:
    s = (v or "").strip()
    if not s:
        return False
    return s.startswith("/") or s.startswith("file:/") or ":\\\\" in s or "C:\\" in s


def verify_dataset(
    track_dir: Path,
    package_root: Path,
    require_materialized_logs: bool,
    split_source: str,
) -> Tuple[Dict[str, int], List[str], List[Path]]:
    issues: List[str] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    files_for_digest: List[Path] = []

    mani = track_dir / "track1_prefix_manifest.csv"
    if not mani.is_file():
        issues.append(f"missing manifest: {mani}")
        return split_counts, issues, files_for_digest
    files_for_digest.append(mani)

    manifest_rows: List[Dict[str, str]] = []
    with mani.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = list(r.fieldnames or [])
        need_fields = ["prefix_id", "split", "prefix_log", "prefix_log_path", "history_csv", "task_dir"]
        for k in need_fields:
            if k not in fields:
                issues.append(f"{mani}: missing required field `{k}`")
        for row in r:
            manifest_rows.append(row)

    manifest_ids: List[str] = []
    by_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for row in manifest_rows:
        pid = row.get("prefix_id", "")
        manifest_ids.append(pid)
        sp = canonical_split(row.get("split", ""))
        if not sp:
            issues.append(f"{mani}: unknown split label for prefix_id={pid}")
            continue
        by_split[sp].append(pid)

        for k in ("history_csv", "task_dir"):
            if (row.get(k) or "").strip():
                issues.append(f"{mani}: field `{k}` should be blank for prefix_id={pid}")

        for k in ("prefix_log", "prefix_log_path"):
            v = (row.get(k) or "").strip()
            if contains_abs_hint(v):
                issues.append(f"{mani}: absolute path leak in `{k}` for prefix_id={pid}")
            if v and not v.startswith("logs/openroad/"):
                issues.append(f"{mani}: non-canonical `{k}` path for prefix_id={pid}: {v}")
            if v and require_materialized_logs:
                if not (package_root / v).is_file():
                    issues.append(f"{mani}: missing materialized log file referenced by `{k}`: {v}")

    split_dir = track_dir / "splits"
    split_file_ids: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for sp in ("train", "val", "test"):
        spf = split_dir / f"{sp}.csv"
        if not spf.is_file():
            issues.append(f"missing split file: {spf}")
            continue
        files_for_digest.append(spf)

        ids: List[str] = []
        with spf.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                ids.append(row.get("prefix_id", ""))

        split_file_ids[sp] = ids
        if split_source == "split_files":
            split_counts[sp] = len(ids)
        else:
            split_counts[sp] = len(by_split[sp])
            expected = by_split[sp]
            if len(ids) != len(expected):
                issues.append(f"{spf}: row count mismatch file={len(ids)} expected={len(expected)}")
            if set(ids) != set(expected):
                issues.append(f"{spf}: prefix_id set mismatch vs manifest split column")

    if split_source == "split_files":
        manifest_set = set(manifest_ids)
        all_split_ids = split_file_ids["train"] + split_file_ids["val"] + split_file_ids["test"]
        if len(all_split_ids) != len(set(all_split_ids)):
            issues.append(f"{track_dir}: duplicate prefix_id across split files")
        if set(all_split_ids) != manifest_set:
            issues.append(f"{track_dir}: split file prefix_id coverage mismatch vs manifest")

    return split_counts, issues, files_for_digest


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify TraceTriage release package consistency")
    ap.add_argument("--package-root", required=True, help="Path to release_package_full or release_package_smoke")
    ap.add_argument("--expected-total-rows", type=int, default=81041)
    ap.add_argument("--expected-train", type=int, default=-1)
    ap.add_argument("--expected-val", type=int, default=-1)
    ap.add_argument("--expected-test", type=int, default=-1)
    ap.add_argument(
        "--split-source",
        choices=["split_files", "manifest"],
        default="split_files",
        help="Use split CSV files, or the manifest split column, as authoritative for split counts.",
    )
    ap.add_argument(
        "--require-materialized-logs",
        action="store_true",
        help="Require every logs/openroad/... reference in CSV files to exist under package root.",
    )
    ap.add_argument("--write-report", default="", help="Optional JSON output path")
    ap.add_argument(
        "--report-package-root",
        default="",
        help="Optional redacted/display package root for JSON output",
    )
    args = ap.parse_args()

    root = Path(args.package_root)
    data_root = root / "data"
    if not data_root.is_dir():
        raise SystemExit(f"missing data dir: {data_root}")

    dataset_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    dataset_names = {p.name for p in dataset_dirs}

    issues: List[str] = []
    if dataset_names != EXPECTED_DATASETS:
        missing = sorted(EXPECTED_DATASETS - dataset_names)
        extra = sorted(dataset_names - EXPECTED_DATASETS)
        if missing:
            issues.append(f"missing datasets: {missing}")
        if extra:
            issues.append(f"unexpected datasets: {extra}")

    total = {"train": 0, "val": 0, "test": 0}
    all_files_for_digest: List[Path] = []
    per_dataset: Dict[str, Dict[str, int]] = {}

    for ds in sorted(dataset_dirs):
        track = ds / "track1_strict"
        sc, ds_issues, files_digest = verify_dataset(
            track,
            root,
            require_materialized_logs=args.require_materialized_logs,
            split_source=args.split_source,
        )
        per_dataset[ds.name] = dict(sc)
        for k in total:
            total[k] += sc[k]
        issues.extend(ds_issues)
        all_files_for_digest.extend(files_digest)

    summary_json = data_root / "summary_all.json"
    if summary_json.is_file():
        all_files_for_digest.append(summary_json)
    else:
        issues.append(f"missing {summary_json}")

    expected = {}
    if args.expected_train >= 0:
        expected["train"] = args.expected_train
    if args.expected_val >= 0:
        expected["val"] = args.expected_val
    if args.expected_test >= 0:
        expected["test"] = args.expected_test
    if expected:
        actual_subset = {k: total[k] for k in expected}
        if actual_subset != expected:
            issues.append(f"global split counts mismatch actual={actual_subset} expected={expected}")
    if sum(total.values()) != args.expected_total_rows:
        issues.append(
            f"total rows mismatch actual={sum(total.values())} expected={args.expected_total_rows}"
        )

    aggregate_digest = sha256_concat(all_files_for_digest, root)
    aggregate_raw_concat = sha256_raw_concat(all_files_for_digest, root)

    result = {
        "package_root": args.report_package_root or str(root),
        "dataset_count": len(dataset_dirs),
        "split_source": args.split_source,
        "split_counts_from_manifest": total,
        "total_rows": sum(total.values()),
        "per_dataset_split_counts": per_dataset,
        "aggregate_digest_manifest_and_splits_sha256_of_file_sha256": aggregate_digest,
        "aggregate_digest_manifest_and_splits_sha256_raw_concat": aggregate_raw_concat,
        "issues_count": len(issues),
        "issues": issues[:2000],
    }

    if args.write_report:
        out = Path(args.write_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
