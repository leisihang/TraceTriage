# Quick-start workflows

This folder provides dataset verification and experiment entry points for TraceTriage.

The default workflow:
- reads the selected dataset package
- verifies the dataset selected by `PACKAGE_ROOT`
- runs a small standard-library baseline smoke test on `asap7_gcd`
- redacts local paths and API metadata from generated JSON/JSONL outputs
- does not call external APIs
- does not require Qwen weights, CUDA, or network access


## Dataset Placement

Set `PACKAGE_ROOT` to the restored Kaggle dataset directory that directly
contains `data/`:

```bash
export PACKAGE_ROOT=/path/to/TraceTriage
# or, for the compact package:
export PACKAGE_ROOT=/path/to/TraceTriage_2026_light
```

The default workflow works with either package. Add `REQUIRE_MATERIALIZED_LOGS=1`
only when checking the full package.

## Layout
- `scripts/run_all_submitready.sh`: quick-start default entry point
- `scripts/run_dataset_pipeline.sh`: verifies the existing package
- `scripts/run_baselines_pipeline.sh`: standard-library smoke baseline by default; optional full XGB baselines
- `scripts/run_qwen_pipeline.sh`: optional Qwen experiment using package-local scripts
- `scripts/run_llm_api_pipeline.sh`: optional API experiment using package-local script
- `scripts/sanitize_smoke_outputs.py`: output redaction helper
- `runs/`: generated outputs

## Quick Start
```bash
cd submitready_workflows
PACKAGE_ROOT=<PACKAGE_ROOT> bash scripts/run_all_submitready.sh
```

Dry run:
```bash
PACKAGE_ROOT=<PACKAGE_ROOT> bash scripts/run_all_submitready.sh --dry-run
```

The default command should complete with only Python standard-library dependencies.

`PACKAGE_ROOT` can point to either the full TraceTriage package or the compact
package. The expected restored layout is:

```text
<dataset-package>/data/
<dataset-package>/logs/openroad/
<dataset-package>/reports/openroad/
<dataset-package>/checksums/
```

Use `REQUIRE_MATERIALIZED_LOGS=1` only for the full package, where every
`logs/openroad/...` CSV reference is materialized. Do not set it for the compact
sample, which intentionally includes only a small log/RPT subset.

## Optional Experiments
Full XGB, Qwen and API experiments are available through the following commands. They use the sanitized package data selected by `PACKAGE_ROOT`; Qwen requires model dependencies, and API experiments require a provider key.

```bash
BASELINE_MODE=single_xgb bash scripts/run_baselines_pipeline.sh
BASELINE_MODE=lodo_xgb bash scripts/run_baselines_pipeline.sh
BASELINE_MODE=crossnode_xgb bash scripts/run_baselines_pipeline.sh

RUN_QWEN=1 MODE=zeroshot bash scripts/run_all_submitready.sh
RUN_LLM_API=1 PROVIDER_KEY="<provider_key>" MAX_SAMPLES=5 bash scripts/run_all_submitready.sh
```

Do not include private API keys, endpoint identifiers, personal paths, or unsanitized logs in public outputs.
