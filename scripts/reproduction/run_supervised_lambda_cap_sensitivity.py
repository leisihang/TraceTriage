#!/usr/bin/env python3
"""Supervised lambda x FSR-cap sensitivity evaluation.

This runner intentionally imports the published Track-1 baseline harness.  It
does not implement a second threshold selector or metric: every cell calls
``run_track1_baselines.choose_best_threshold`` and
``run_track1_baselines.evaluate_predictions`` on the same cross-node splits and
model configuration that generated the frozen paper rows.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO / "scripts/experiments"
sys.path.insert(0, str(SCRIPT_DIR))
import run_track1_baselines as baseline
import os
OUTPUT_ROOT = REPO / "reproduction_outputs"
DATA_ROOT = Path(os.environ.get("PACKAGE_ROOT", ".")) / "data"
QWEN_GRID = REPO / "reference_results/qwen_lambda_cap_grid.csv"
SEEDS = (42, 123, 3407)
MODELS = ("hgb", "xgboost", "logreg")
LAMBDAS = (0.5, 1.0, 2.0, 5.0)
CAPS = (0.01, 0.05, 0.10, 0.20)
PUBLISHED_NET = {"qwen": 67.12, "hgb": 61.22, "xgboost": 24.24, "logreg": 20.40}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise RuntimeError("Cannot aggregate an empty collection")
    return float(statistics.mean(values)), float(statistics.stdev(values)) if len(values) > 1 else 0.0


def dataset_rows(dataset: Path, split: str) -> list[dict[str, str]]:
    rows = baseline.load_rows(
        str(dataset / "track1_strict" / "splits" / f"{split}.csv"),
        require_binary_label=True,
        label_field="strict_success",
    )
    if not rows:
        raise RuntimeError(f"Empty {split} split for {dataset}")
    return rows


def platform(dataset: Path) -> str:
    rows = dataset_rows(dataset, "test")
    value = str(rows[0].get("platform", ""))
    if value not in {"asap7", "sky130hd"}:
        raise RuntimeError(f"Unexpected platform for {dataset}: {value!r}")
    return value


def all_datasets() -> list[Path]:
    found = sorted(
        path for path in DATA_ROOT.iterdir()
        if (path / "track1_strict" / "splits" / "test.csv").is_file()
    )
    if len(found) != 10:
        raise RuntimeError(f"Expected 10 canonical datasets, found {len(found)} under {DATA_ROOT}")
    return found


def train_and_score(model: str, seed: int, source_datasets: list[Path], target_datasets: list[Path]) -> list[dict[str, Any]]:
    """Train once using the canonical harness, retaining val/test probabilities.

    Lambda and FSR-cap affect only threshold selection, so keeping the frozen
    model's probabilities permits all 16 operating points without changing a
    fitted model.  This is the same separation used by the Qwen sensitivity
    evaluator.
    """
    source_rows = [
        row
        for dataset in source_datasets
        for split in ("train", "val")
        for row in dataset_rows(dataset, split)
    ]
    # The canonical function fits only on ``source_rows``. Concatenating target
    # rows therefore leaves the trained model untouched while avoiding five
    # identical fits per direction; the score vectors are split back on the
    # original dataset boundaries before each target-specific threshold search.
    target_splits = [(dataset, dataset_rows(dataset, "val"), dataset_rows(dataset, "test")) for dataset in target_datasets]
    all_val = [row for _dataset, val_rows, _test_rows in target_splits for row in val_rows]
    all_test = [row for _dataset, _val_rows, test_rows in target_splits for row in test_rows]
    _p_train, all_p_val, all_p_test, selected = baseline.train_supervised_model(
        source_rows,
        all_val,
        all_test,
        include_missing_indicators=False,
        preferred_model=model,
        feature_mode="param_plus_prefix",
        seed=seed,
    )
    if selected != model and not (model == "logreg" and selected == "logreg_sgd"):
        raise RuntimeError(f"Requested {model}, selected {selected}; canonical environment changed")

    outputs: list[dict[str, Any]] = []
    val_start = 0
    test_start = 0
    for target_dataset, val_rows, test_rows in target_splits:
        p_val = all_p_val[val_start:val_start + len(val_rows)]
        p_test = all_p_test[test_start:test_start + len(test_rows)]
        val_start += len(val_rows)
        test_start += len(test_rows)
        if len(p_val) != len(val_rows) or len(p_test) != len(test_rows):
            raise RuntimeError(f"Score alignment failure for {target_dataset}")
        outputs.append(
            {
                "model": model,
                "seed": seed,
                "source_platform": platform(source_datasets[0]),
                "target_platform": platform(target_dataset),
                "dataset": target_dataset.name,
                "selected_model": selected,
                "source_rows": len(source_rows),
                "val_rows": val_rows,
                "test_rows": test_rows,
                "p_val": p_val,
                "p_test": p_test,
            }
        )
    if val_start != len(all_p_val) or test_start != len(all_p_test):
        raise RuntimeError("Unused target score rows after split reconstruction")
    return outputs


def evaluate_cell(fold: dict[str, Any], lam: float, cap: float) -> dict[str, Any]:
    threshold = baseline.choose_best_threshold(
        fold["val_rows"],
        fold["p_val"],
        metric="avg_weighted_net_saved_seconds_with_inference",
        max_success_false_stop_rate=cap,
        enforce_success_false_stop_constraint=True,
        lambda_false_stop=lam,
        inference_cost_per_row=0.0,
    )
    metrics = baseline.evaluate_predictions(
        fold["test_rows"],
        fold["p_test"],
        threshold,
        lambda_false_stop=lam,
        inference_cost_per_row=0.0,
    )
    metrics = baseline.enrich_with_oracle_headroom(fold["test_rows"], metrics, lambda_false_stop=lam)
    return {
        "model": fold["model"],
        "lambda": lam,
        "cap": cap,
        "seed": fold["seed"],
        "source_platform": fold["source_platform"],
        "target_platform": fold["target_platform"],
        "dataset": fold["dataset"],
        "selected_model": fold["selected_model"],
        "source_rows": fold["source_rows"],
        "val_n": len(fold["val_rows"]),
        "test_n": len(fold["test_rows"]),
        "threshold": threshold,
        "accuracy": float(metrics["accuracy"]),
        "frs": float(metrics["fail_recall_stop"]),
        "fsr": float(metrics["success_false_stop_rate"]),
        "net": float(metrics["avg_weighted_net_saved_seconds_with_inference"]),
        "headroom": float(metrics["headroom_ratio_vs_oracle"]),
    }


def aggregate(
    rows: list[dict[str, Any]],
    models: tuple[str, ...] = MODELS,
    lambdas: tuple[float, ...] = LAMBDAS,
    caps: tuple[float, ...] = CAPS,
) -> list[dict[str, Any]]:
    by_cell_seed: dict[tuple[str, float, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell_seed[(str(row["model"]), float(row["lambda"]), float(row["cap"]), int(row["seed"]))].append(row)

    out: list[dict[str, Any]] = []
    for model in models:
        for lam in lambdas:
            for cap in caps:
                per_seed: dict[str, list[float]] = defaultdict(list)
                for seed in SEEDS:
                    folds = by_cell_seed[(model, lam, cap, seed)]
                    if len(folds) != 10:
                        raise RuntimeError(f"{model}, lambda={lam}, cap={cap}, seed={seed}: expected 10 folds, got {len(folds)}")
                    for metric in ("net", "frs", "fsr"):
                        per_seed[metric].append(float(statistics.mean(float(fold[metric]) for fold in folds)))
                net_mean, net_std = mean_std(per_seed["net"])
                frs_mean, frs_std = mean_std(per_seed["frs"])
                fsr_mean, fsr_std = mean_std(per_seed["fsr"])
                out.append(
                    {
                        "model": model,
                        "lambda": lam,
                        "cap": cap,
                        "net_mean": net_mean,
                        "net_std": net_std,
                        "frs_mean": frs_mean,
                        "frs_std": frs_std,
                        "fsr_mean": fsr_mean,
                        "fsr_std": fsr_std,
                    }
                )
    return out


def qwen_rows() -> list[dict[str, Any]]:
    if not QWEN_GRID.is_file():
        raise FileNotFoundError(f"Missing canonical Qwen grid: {QWEN_GRID}")
    rows: list[dict[str, Any]] = []
    for row in read_csv(QWEN_GRID):
        rows.append(
            {
                "model": "qwen",
                "lambda": float(row["lambda"]),
                "cap": float(row["cap"]),
                "net_mean": float(row["net_mean"]),
                "net_std": float(row["net_std"]),
                "frs_mean": float(row["frs_mean"]),
                "frs_std": float(row["frs_std"]),
                "fsr_mean": float(row["fsr_mean"]),
                "fsr_std": float(row["fsr_std"]),
            }
        )
    if len(rows) != 16:
        raise RuntimeError(f"Expected 16 canonical Qwen cells, found {len(rows)}")
    return rows


def gate_rows(tabular_grid: list[dict[str, Any]], qwen_grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(str(row["model"]), float(row["lambda"]), float(row["cap"])): row for row in tabular_grid + qwen_grid}
    out: list[dict[str, Any]] = []
    for model, published in PUBLISHED_NET.items():
        row = lookup.get((model, 1.0, 0.10))
        if row is None:
            raise RuntimeError(f"Missing headline-point row for {model}")
        recomputed = float(row["net_mean"])
        diff = recomputed - published
        out.append(
            {
                "model": model,
                "lambda": 1.0,
                "cap": 0.10,
                "published_net": published,
                "recomputed_net": recomputed,
                "difference": diff,
                "tolerance": 1.0,
                "status": "PASS" if abs(diff) <= 1.0 else "FAIL",
                "source": "canonical_qwen_r1_grid" if model == "qwen" else "run_track1_baselines.py canonical cross-node harness",
            }
        )
    return out


def inversion_rows(grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(str(row["model"]), float(row["lambda"]), float(row["cap"])): row for row in grid}
    out: list[dict[str, Any]] = []
    for lam in LAMBDAS:
        for cap in CAPS:
            qwen = float(lookup[("qwen", lam, cap)]["net_mean"])
            others = [(model, float(lookup[(model, lam, cap)]["net_mean"])) for model in MODELS]
            best_model, best_net = max(others, key=lambda item: item[1])
            out.append(
                {
                    "lambda": lam,
                    "cap": cap,
                    "qwen_net": qwen,
                    "max_non_qwen_model": best_model,
                    "max_non_qwen_net": best_net,
                    "max_non_qwen_ge_qwen": str(best_net >= qwen).upper(),
                }
            )
    return out


def write_final_report(output_dir: Path, gates: list[dict[str, Any]], grid: list[dict[str, Any]], inversions: list[dict[str, Any]]) -> None:
    lines = ["# Supervised lambda x FSR-cap sensitivity", "",
             "Cells report equal-dataset macro Net, averaged across three seeds.", "",
             "| Model | Reference Net | Computed Net | Match |",
             "| --- | ---: | ---: | --- |"]
    lines.extend(f"| {row['model']} | {float(row['published_net']):.2f} | {float(row['recomputed_net']):.3f} | {row['status']} |" for row in gates)
    lines.extend(["", "## Risk-tolerance dependence", ""])
    for row in inversions:
        if row["max_non_qwen_ge_qwen"] == "TRUE":
            lines.append(f"At lambda={row['lambda']}, cap={row['cap']}, {row['max_non_qwen_model']} has Net {float(row['max_non_qwen_net']):.3f} versus Qwen {float(row['qwen_net']):.3f}.")
    (output_dir / "sensitivity_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    lookup = {(str(row["model"]), float(row["lambda"]), float(row["cap"])): row for row in grid}
    tex = [
        "% Auto-generated by run_supervised_lambda_cap_sensitivity.py; do not hand-edit.",
        "\\begin{tabular}{rrcccc}",
        "\\toprule",
        r"$\lambda$ & FSR cap & Qwen & HGB & XGBoost & LogReg \\",
        "\\midrule",
    ]
    for lam in LAMBDAS:
        for cap in CAPS:
            values = []
            for model in ("qwen", "hgb", "xgboost", "logreg"):
                row = lookup[(model, lam, cap)]
                values.append(f"{float(row['net_mean']):.2f} $\\pm$ {float(row['net_std']):.2f}")
            tex.append(f"{lam:g} & {cap:.2f} & " + " & ".join(values) + " " + r"\\")
    tex.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output_dir / "supervised_lambda_cap_grid.tex").write_text("\n".join(tex), encoding="utf-8")


def parse_grid(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    required = ("model", "lambda", "cap", "net_mean", "net_std", "frs_mean", "frs_std", "fsr_mean", "fsr_std")
    rows = read_csv(path)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in {path}, found {len(rows)}")
    return [{key: (str(row[key]) if key == "model" else float(row[key])) for key in required} for row in rows]


def lambda_tag(lam: float) -> str:
    return f"{lam:g}"


def write_score_export(path: Path, scored: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for fold in scored:
        for split, source_rows, probs in (("val", fold["val_rows"], fold["p_val"]), ("test", fold["test_rows"], fold["p_test"])):
            for row, prob in zip(source_rows, probs):
                rows.append(
                    {
                        "model": fold["model"], "seed": fold["seed"],
                        "source_platform": fold["source_platform"], "target_platform": fold["target_platform"],
                        "dataset": fold["dataset"], "selected_model": fold["selected_model"],
                        "source_rows": fold["source_rows"], "split": split,
                        "prefix_id": row["prefix_id"], "success_prob": float(prob),
                    }
                )
    write_csv(path, rows)


def load_score_export(model: str, output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / f"partial_{model}_scores.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model score export: {path}; run --model {model} first")
    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        if row.get("model") != model:
            raise RuntimeError(f"Unexpected model in {path}: {row.get('model')}")
        grouped[(int(row["seed"]), row["dataset"], row["split"])].append(row)
    datasets_by_name = {dataset.name: dataset for dataset in all_datasets()}
    folds: list[dict[str, Any]] = []
    for seed in SEEDS:
        for dataset_name, dataset in sorted(datasets_by_name.items()):
            val_score_rows = grouped[(seed, dataset_name, "val")]
            test_score_rows = grouped[(seed, dataset_name, "test")]
            if not val_score_rows or not test_score_rows:
                raise RuntimeError(f"Missing val/test scores for {model}, seed={seed}, dataset={dataset_name}")
            val_prob = {row["prefix_id"]: float(row["success_prob"]) for row in val_score_rows}
            test_prob = {row["prefix_id"]: float(row["success_prob"]) for row in test_score_rows}
            val_rows = dataset_rows(dataset, "val")
            test_rows = dataset_rows(dataset, "test")
            if set(val_prob) != {row["prefix_id"] for row in val_rows} or set(test_prob) != {row["prefix_id"] for row in test_rows}:
                raise RuntimeError(f"Prefix-score join mismatch for {model}, seed={seed}, dataset={dataset_name}")
            folds.append(
                {
                    "model": model, "seed": seed, "dataset": dataset_name,
                    "source_platform": val_score_rows[0]["source_platform"],
                    "target_platform": val_score_rows[0]["target_platform"],
                    "selected_model": val_score_rows[0]["selected_model"],
                    "source_rows": int(val_score_rows[0]["source_rows"]),
                    "val_rows": val_rows, "test_rows": test_rows,
                    "p_val": [val_prob[row["prefix_id"]] for row in val_rows],
                    "p_test": [test_prob[row["prefix_id"]] for row in test_rows],
                }
            )
    if len(folds) != 30:
        raise RuntimeError(f"Expected 30 scored folds for {model}, found {len(folds)}")
    return folds


def run_model_chunk(model: str, output_dir: Path) -> None:
    baseline.TARGET_LABEL_FIELD = "strict_success"
    datasets = all_datasets()
    by_platform = {
        name: [dataset for dataset in datasets if platform(dataset) == name]
        for name in ("asap7", "sky130hd")
    }
    if any(len(value) != 5 for value in by_platform.values()):
        raise RuntimeError(f"Expected five datasets per platform, got { {key: len(value) for key, value in by_platform.items()} }")

    scored: list[dict[str, Any]] = []
    for seed in SEEDS:
        for source_name, target_name in (("asap7", "sky130hd"), ("sky130hd", "asap7")):
            print(f"[fit] model={model} seed={seed} {source_name}->{target_name}", flush=True)
            scored.extend(train_and_score(model, seed, by_platform[source_name], by_platform[target_name]))

    headline_rows = [evaluate_cell(fold, 1.0, 0.10) for fold in scored]
    headline_grid = aggregate(headline_rows, (model,), (1.0,), (0.10,))
    headline = headline_grid[0]
    published = PUBLISHED_NET[model]
    gate = {
        "model": model,
        "lambda": 1.0,
        "cap": 0.10,
        "published_net": published,
        "recomputed_net": float(headline["net_mean"]),
        "difference": float(headline["net_mean"]) - published,
        "tolerance": 1.0,
        "status": "PASS" if abs(float(headline["net_mean"]) - published) <= 1.0 else "FAIL",
        "source": "run_track1_baselines.py canonical cross-node harness",
    }
    write_csv(output_dir / f"partial_{model}_gate_check.csv", [gate])
    if gate["status"] != "PASS":
        raise SystemExit(f"protocol mismatch — not comparable for {model}; partial gate written and grid was not emitted")
    write_score_export(output_dir / f"partial_{model}_scores.csv", scored)
    print(f"{output_dir / f'partial_{model}_scores.csv'}", flush=True)


def run_model_lambda_chunk(model: str, lam: float, cap: float, output_dir: Path) -> None:
    gate_path = output_dir / f"partial_{model}_gate_check.csv"
    if not gate_path.is_file() or read_csv(gate_path)[0].get("status") != "PASS":
        raise RuntimeError(f"Model gate is absent or failed for {model}; run --model {model} first")
    scored = load_score_export(model, output_dir)
    per_fold = [evaluate_cell(fold, lam, cap) for fold in scored]
    tabular_grid = aggregate(per_fold, (model,), (lam,), (cap,))
    write_csv(
        output_dir / f"partial_{model}_lambda_{lambda_tag(lam)}_cap_{lambda_tag(cap)}_grid.csv",
        tabular_grid,
        ["model", "lambda", "cap", "net_mean", "net_std", "frs_mean", "frs_std", "fsr_mean", "fsr_std"],
    )
    write_csv(output_dir / f"partial_{model}_lambda_{lambda_tag(lam)}_cap_{lambda_tag(cap)}_per_fold.csv", per_fold)


def finalize(output_dir: Path) -> None:
    tabular_grid = [
        row
        for model in MODELS
        for lam in LAMBDAS
        for cap in CAPS
        for row in parse_grid(output_dir / f"partial_{model}_lambda_{lambda_tag(lam)}_cap_{lambda_tag(cap)}_grid.csv", 1)
    ]
    qwen_grid = qwen_rows()
    gates = gate_rows(tabular_grid, qwen_grid)
    write_csv(output_dir / "gate_check.csv", gates)
    if any(row["status"] != "PASS" for row in gates):
        raise SystemExit("protocol mismatch — not comparable; final grid was not emitted")
    combined = sorted(qwen_grid + tabular_grid, key=lambda row: (float(row["cap"]), float(row["lambda"]), str(row["model"])))
    write_csv(
        output_dir / "supervised_lambda_cap_grid.csv",
        combined,
        ["model", "lambda", "cap", "net_mean", "net_std", "frs_mean", "frs_std", "fsr_mean", "fsr_std"],
    )
    per_fold_paths = [
        output_dir / f"partial_{model}_lambda_{lambda_tag(lam)}_cap_{lambda_tag(cap)}_per_fold.csv"
        for model in MODELS for lam in LAMBDAS for cap in CAPS
    ]
    per_fold = [row for path in per_fold_paths for row in read_csv(path)]
    write_csv(output_dir / "tabular_lambda_cap_per_fold.csv", per_fold)
    inversions = inversion_rows(combined)
    write_csv(output_dir / "inversion_check.csv", inversions)
    inverted = [row for row in inversions if row["max_non_qwen_ge_qwen"] == "TRUE"]
    (output_dir / "inversion_summary.txt").write_text(
        "ranking inverts anywhere: " + ("yes" if inverted else "no") + "\n"
        + ("\n".join(f"lambda={row['lambda']}, cap={row['cap']}, model={row['max_non_qwen_model']}" for row in inverted) if inverted else ""),
        encoding="utf-8",
    )
    write_final_report(output_dir, gates, combined, inversions)
    print(output_dir, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "supervised_lambda_cap_sensitivity")
    parser.add_argument("--model", choices=MODELS, help="Run and gate one deterministic tabular model chunk")
    parser.add_argument("--model-lambda", choices=MODELS, help="Evaluate one lambda/cap cell from an already-gated score export")
    parser.add_argument("--lambda", dest="lam", type=float, choices=LAMBDAS, help="Lambda for --model-lambda")
    parser.add_argument("--cap", type=float, choices=CAPS, help="FSR cap for --model-lambda")
    parser.add_argument("--finalize", action="store_true", help="Gate and combine all completed model chunks with the existing Qwen grid")
    args = parser.parse_args()
    selected_modes = int(bool(args.model)) + int(bool(args.model_lambda)) + int(bool(args.finalize))
    if selected_modes != 1 or (args.model_lambda and (args.lam is None or args.cap is None)) or (not args.model_lambda and (args.lam is not None or args.cap is not None)):
        raise SystemExit("Pass one of --model MODEL, --model-lambda MODEL --lambda LAMBDA --cap CAP, or --finalize")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.model:
        run_model_chunk(str(args.model), args.output_dir)
    elif args.model_lambda:
        run_model_lambda_chunk(str(args.model_lambda), float(args.lam), float(args.cap), args.output_dir)
    else:
        finalize(args.output_dir)


if __name__ == "__main__":
    main()
