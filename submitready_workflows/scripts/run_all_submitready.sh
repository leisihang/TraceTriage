#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WF_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RC_ROOT="$(cd "$WF_ROOT/.." && pwd)"
ROOT="${ROOT:-$RC_ROOT}"
DRY_RUN_FLAG=""
DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN_FLAG="--dry-run"
  DRY_RUN=1
fi

bash "$SCRIPT_DIR/run_dataset_pipeline.sh" $DRY_RUN_FLAG
bash "$SCRIPT_DIR/run_baselines_pipeline.sh" $DRY_RUN_FLAG

if [ "${RUN_QWEN:-0}" = "1" ]; then
  bash "$SCRIPT_DIR/run_qwen_pipeline.sh" $DRY_RUN_FLAG
else
  echo "[all] skip qwen; set RUN_QWEN=1 to run optional local Qwen experiments"
fi

if [ "${RUN_LLM_API:-0}" = "1" ]; then
  bash "$SCRIPT_DIR/run_llm_api_pipeline.sh" $DRY_RUN_FLAG
else
  echo "[all] skip llm api; set RUN_LLM_API=1 and PROVIDER_KEY to run optional API experiments"
fi

SANITIZE_PY="$SCRIPT_DIR/sanitize_smoke_outputs.py"
RUNS_ROOT="$WF_ROOT/runs"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] python3 \"$SANITIZE_PY\" --runs-root \"$RUNS_ROOT\" --repo-root \"$ROOT\""
else
  python3 "$SANITIZE_PY" --runs-root "$RUNS_ROOT" --repo-root "$ROOT"
fi

echo "[all] done"
