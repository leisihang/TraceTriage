#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rel_in_package(abs_path: str, source_root: str, package_subdir: str) -> str:
    p = Path(abs_path)
    sr = Path(source_root)
    rel = p.relative_to(sr)
    return str(Path(package_subdir) / rel).replace('\\', '/')


def safe_copy(abs_path: str, dst_abs: Path) -> bool:
    src = Path(abs_path)
    if not src.is_file():
        return False
    ensure_dir(dst_abs.parent)
    if not dst_abs.exists():
        shutil.copy2(src, dst_abs)
    return True


def iter_dataset_manifests(data_root: Path) -> Iterable[Path]:
    for p in sorted(data_root.glob('*/track1_strict/track1_prefix_manifest.csv')):
        yield p


def copy_tree_min(src: Path, dst: Path, names: List[str]) -> None:
    ensure_dir(dst)
    for n in names:
        s = src / n
        d = dst / n
        if s.is_file():
            ensure_dir(d.parent)
            shutil.copy2(s, d)
        elif s.is_dir():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_release_readme(out_root: Path) -> None:
    txt = """# TraceTriage Sanitized Release Package

This package is sanitized for public/review release:
- dataset manifests/splits are included under `data/`
- referenced prefix logs are included under `logs/openroad/`
- matched report files are included under `reports/openroad/`
- all manifest path fields are rewritten to relative paths
- sensitive absolute path fields are blanked
"""
    (out_root / "README.md").write_text(txt, encoding="utf-8")


def rebuild_checksums(out_root: Path, out_data_root: Path) -> None:
    cdir = out_root / "checksums"
    ensure_dir(cdir)
    file_list = cdir / "data_file_list.txt"
    sha_file = cdir / "data_sha256.txt"

    files: List[Path] = []
    for p in sorted(out_data_root.rglob("*")):
        if p.is_file():
            files.append(p)
    summary_json = out_data_root.parent / "summary_all.json"
    if summary_json.is_file():
        files.append(summary_json)

    rels = [str(p.relative_to(out_root)).replace("\\", "/") for p in files]
    file_list.write_text("\n".join(rels) + ("\n" if rels else ""), encoding="utf-8")

    lines: List[str] = []
    for rel, p in zip(rels, files):
        lines.append(f"{sha256_file(p)}  {rel}")
    sha_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def sanitize_row_paths(
    row: Dict[str, str],
    out_root: Path,
    log_root: Path,
    rpt_root: Path,
    copied_logs: Set[str],
    copied_rpts: Set[str],
    copied_report_runs: Set[str],
) -> Dict[str, str]:
    r = dict(row)

    for k in ("prefix_log", "prefix_log_path"):
        v = (r.get(k) or "").strip()
        if not v:
            continue
        if v.startswith(str(log_root)):
            rel = rel_in_package(v, str(log_root), "logs/openroad")
            dst = out_root / rel
            if safe_copy(v, dst):
                copied_logs.add(rel)
            try:
                rel_parts = Path(v).relative_to(log_root).parts
                if len(rel_parts) >= 4:
                    platform, design, run_key = rel_parts[0], rel_parts[1], rel_parts[2]
                    run_sig = f"{platform}/{design}/{run_key}"
                    if run_sig not in copied_report_runs:
                        report_run_dir = rpt_root / platform / design / run_key
                        if report_run_dir.is_dir():
                            for rpt in report_run_dir.glob("*.rpt"):
                                rpt_rel = str(Path("reports/openroad") / rpt.relative_to(rpt_root)).replace("\\", "/")
                                rpt_dst = out_root / rpt_rel
                                if safe_copy(str(rpt), rpt_dst):
                                    copied_rpts.add(rpt_rel)
                        copied_report_runs.add(run_sig)
            except Exception:
                pass
            r[k] = rel
        else:
            r[k] = ""

    for k in ("history_csv", "task_dir"):
        if k in r:
            r[k] = ""

    return r


def canonical_split(v: str) -> str:
    s = (v or "").strip().lower()
    if s in ("train", "tr"):
        return "train"
    if s in ("val", "valid", "validation", "dev"):
        return "val"
    if s in ("test", "te"):
        return "test"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description='Sanitize release package with relative paths and in-package logs/reports')
    ap.add_argument('--source-data-root', required=True, help='private/source data directory')
    ap.add_argument('--source-package-root', required=True, help='private/source package directory')
    ap.add_argument('--source-log-root', required=True, help='OpenROAD flow logs root')
    ap.add_argument('--source-report-root', required=True, help='OpenROAD flow reports root')
    ap.add_argument('--output-root', required=True, help='release_candidate output root')
    ap.add_argument('--max-rows-per-dataset', type=int, default=0, help='for smoke test; 0 means full')
    args = ap.parse_args()

    source_data_root = Path(args.source_data_root)
    source_pkg_root = Path(args.source_package_root)
    log_root = Path(args.source_log_root)
    rpt_root = Path(args.source_report_root)
    out_root = Path(args.output_root)

    out_data_root = out_root / 'data'
    out_logs_root = out_root / 'logs' / 'openroad'
    out_reports_root = out_root / 'reports' / 'openroad'
    ensure_dir(out_data_root)
    ensure_dir(out_logs_root)
    ensure_dir(out_reports_root)

    # summary_all.json is regenerated/sanitized after data copy.

    copied_logs: Set[str] = set()
    copied_rpts: Set[str] = set()
    copied_report_runs: Set[str] = set()
    dataset_stats: Dict[str, Dict[str, int]] = {}

    for mani in iter_dataset_manifests(source_data_root):
        ds_dir = mani.parent.parent
        ds_name = ds_dir.name
        out_ds_track = out_data_root / ds_name / 'track1_strict'
        ensure_dir(out_ds_track)

        with mani.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            rows = list(reader)

        if args.max_rows_per_dataset > 0:
            rows = rows[: args.max_rows_per_dataset]

        sanitized_rows: List[Dict[str, str]] = []
        for r in rows:
            r2 = sanitize_row_paths(
                r,
                out_root=out_root,
                log_root=log_root,
                rpt_root=rpt_root,
                copied_logs=copied_logs,
                copied_rpts=copied_rpts,
                copied_report_runs=copied_report_runs,
            )
            sanitized_rows.append(r2)

        # Rebuild split files from manifest split labels to guarantee consistency.
        split_rows_map: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}
        for r in sanitized_rows:
            k = canonical_split(r.get("split", ""))
            if k:
                split_rows_map[k].append(r)

        out_mani = out_ds_track / 'track1_prefix_manifest.csv'
        with out_mani.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sanitized_rows:
                w.writerow(r)

        out_splits = out_ds_track / "splits"
        ensure_dir(out_splits)
        for split_name, srows in split_rows_map.items():
            out_sp = out_splits / f"{split_name}.csv"
            with out_sp.open("w", newline="", encoding="utf-8") as sf:
                sw = csv.DictWriter(sf, fieldnames=fields)
                sw.writeheader()
                for sr in srows:
                    sw.writerow(sr)

        dataset_stats[ds_name] = {
            'rows': len(rows),
            'splits_rebuilt': 1,
            'split_counts': {
                'train': len(split_rows_map.get('train', [])),
                'val': len(split_rows_map.get('val', [])),
                'test': len(split_rows_map.get('test', [])),
            },
        }

    # Build sanitized summary_all.json with relative dataset dirs.
    summary_all = {"datasets": []}
    for ds_name, st in dataset_stats.items():
        summary_all["datasets"].append(
            {
                "dataset": ds_name,
                "dir": f"data/{ds_name}/track1_strict",
                "rows": st.get("rows", 0),
            }
        )
    with (out_data_root / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2, ensure_ascii=False)

    summary = {
        'output_root': '.',
        'datasets': dataset_stats,
        'copied_log_files': len(copied_logs),
        'copied_rpt_files': len(copied_rpts),
        'path_policy': {
            'prefix_log_fields': 'relative paths under logs/openroad',
            'history_csv': 'blanked',
            'task_dir': 'blanked',
        },
    }

    write_release_readme(out_root)
    rebuild_checksums(out_root, out_data_root)

    with (out_root / 'sanitization_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
