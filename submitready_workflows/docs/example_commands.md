# Example Commands


Expected local placement after Kaggle download:
```text
/path/to/TraceTriage/data
/path/to/TraceTriage_2026_light/data
```

Set `PACKAGE_ROOT` to the parent directory of `data/`.

Run the quick-start default workflow:
```bash
cd submitready_workflows
PACKAGE_ROOT=<PACKAGE_ROOT> bash scripts/run_all_submitready.sh
```

Inspect commands without executing them:
```bash
PACKAGE_ROOT=<PACKAGE_ROOT> bash scripts/run_all_submitready.sh --dry-run
```

Run only package verification:
```bash
PACKAGE_ROOT=<PACKAGE_ROOT> bash scripts/run_dataset_pipeline.sh
```

Full package verification with all materialized logs required:
```bash
PACKAGE_ROOT=/path/to/TraceTriage \
  REQUIRE_MATERIALIZED_LOGS=1 \
  bash scripts/run_dataset_pipeline.sh
```

Compact compact package verification:
```bash
PACKAGE_ROOT=/path/to/TraceTriage_2026_light \
  bash scripts/run_dataset_pipeline.sh
```

Run only the standard-library smoke baseline:
```bash
PACKAGE_ROOT=<PACKAGE_ROOT> TARGET_DATASET=asap7_gcd MAX_ROWS_PER_SPLIT=50 bash scripts/run_baselines_pipeline.sh
```

Optional full-source experiments:
```bash
BASELINE_MODE=single_xgb bash scripts/run_baselines_pipeline.sh
BASELINE_MODE=lodo_xgb bash scripts/run_baselines_pipeline.sh
BASELINE_MODE=crossnode_xgb bash scripts/run_baselines_pipeline.sh

RUN_QWEN=1 MODE=zeroshot bash scripts/run_all_submitready.sh
PACKAGE_ROOT=<PACKAGE_ROOT> RUN_LLM_API=1 PROVIDER_KEY="<provider_key>" MAX_SAMPLES=5 bash scripts/run_all_submitready.sh
```

Full log/report package:
```bash
PACKAGE_ROOT=/path/to/TraceTriage \
  bash scripts/run_all_submitready.sh
```

Verify the full package and require all materialized log files:
```bash
REQUIRE_MATERIALIZED_LOGS=1 \
PACKAGE_ROOT=/path/to/TraceTriage \
  bash scripts/run_dataset_pipeline.sh
```
