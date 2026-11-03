# TraceTriage

Code and offline evaluation materials for **TraceTriage: A Benchmark for Cost-Aware Stop-or-Continue Decisions in Delayed-Outcome Workflows**, NeurIPS 2026 Evaluations and Datasets Track.

Sihang Lei, Yihang Qiu, Xueyan Zhao, Yuwei Wang, Weiqiang Wang.

## Data

- Full logs/reports and complete CSV data: https://www.kaggle.com/datasets/leisihang/tracetriage
- Compact package with complete CSV data and representative logs/reports: https://www.kaggle.com/datasets/leisihang/tracetriage-light

Both packages contain the same 81,041 prefix rows and the authoritative 64,645 / 8,132 / 8,264 train / validation / test rows. Set `PACKAGE_ROOT` to the extracted directory containing `data/`. Load train, validation and test membership from the explicit split files.

Code license: MIT. Derived TraceTriage data license: CC BY-SA 4.0, subject to the upstream assets' own licenses.

## Quick start

```sh
export PACKAGE_ROOT=/path/to/TraceTriage
python scripts/verify_release_package.py --package-root "$PACKAGE_ROOT"
cd submitready_workflows
bash scripts/run_all_submitready.sh --dry-run
```

The standard-library smoke workflow checks installation and data access. The training and inference runners are under `scripts/experiments/`; full datasets, local model weights, ML dependencies, or provider credentials are required as appropriate. No external API is called by the offline replay commands.

## Reproduce final evaluations

See [REPRODUCING.md](REPRODUCING.md) for commands to compute the hosted table, validation-fitted Platt/isotonic controls, Qwen/shared-run bootstrap, frozen continue-cost sweep, and sensitivity-grid aggregation. Compact frozen inputs are included in `reproduction_inputs/`. `reference_results/` identifies the saved comparison targets.

The experiment runners report variation across folds. Run-level confidence intervals use `scripts/reproduction/recompute_bootstrap.py`, which resamples complete test runs.

LODFO per-fold results and summaries are in `reference_results/`. Qwen sensitivity tables use the included frozen grid cells. Experiment coverage and input requirements are listed in the reproduction guide.

## Release metadata

`metadata/dataset_release_info.json` records dataset URLs, authors, license and the SHA-256 digest of the uploaded `TraceTriage.zip`. Dataset limitations, provenance and intended uses are described in `metadata/responsible_ai_metadata.md`.

The code and frozen inputs contain no provider keys or raw API request logs. Keep credentials outside the repository. See `correction_note.md` for the reporting corrections described in the paper.
