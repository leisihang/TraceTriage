#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WF_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RC_ROOT="$(cd "$WF_ROOT/.." && pwd)"
PY_BIN="${PY_BIN:-python3}"
DRY_RUN=0

if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

PACKAGE_ROOT="${PACKAGE_ROOT:-$RC_ROOT}"
TARGET_DATASET="${TARGET_DATASET:-asap7_gcd}"
LABEL_FIELD="${LABEL_FIELD:-strict_success}"
MAX_ROWS_PER_SPLIT="${MAX_ROWS_PER_SPLIT:-50}"
BASELINE_MODE="${BASELINE_MODE:-stdlib_smoke}" # stdlib_smoke|single_xgb|lodo_xgb|crossnode_xgb
EXPERIMENTS_DIR="$RC_ROOT/scripts/experiments"
DATASET_GLOB="${DATASET_GLOB:-$PACKAGE_ROOT/data/*}"
SUPERVISED_MODEL="${SUPERVISED_MODEL:-auto}"
SOURCE_PLATFORM="${SOURCE_PLATFORM:-asap7}"
TARGET_PLATFORM="${TARGET_PLATFORM:-sky130hd}"

if [ -z "${OUT_ROOT+x}" ]; then
  case "$BASELINE_MODE" in
    stdlib_smoke) OUT_ROOT="$WF_ROOT/runs/stdlib_baseline_smoke" ;;
    single_xgb) OUT_ROOT="$WF_ROOT/runs/baseline_single_xgb" ;;
    lodo_xgb) OUT_ROOT="$WF_ROOT/runs/baseline_lodo_xgb" ;;
    crossnode_xgb) OUT_ROOT="$WF_ROOT/runs/baseline_crossnode_xgb" ;;
    *) OUT_ROOT="$WF_ROOT/runs/baseline_$BASELINE_MODE" ;;
  esac
fi

run_cmd() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run] %s\n' "$*"
  else
    eval "$@"
  fi
}

mkdir -p "$OUT_ROOT"

echo "[baseline] package_root=$PACKAGE_ROOT"
echo "[baseline] output_root=$OUT_ROOT"
echo "[baseline] mode=$BASELINE_MODE target_dataset=$TARGET_DATASET max_rows_per_split=$MAX_ROWS_PER_SPLIT"

case "$BASELINE_MODE" in
  stdlib_smoke)
    run_cmd "$PY_BIN \"$SCRIPT_DIR/run_stdlib_baseline_smoke.py\" \
      --package-root \"$PACKAGE_ROOT\" \
      --output-dir \"$OUT_ROOT\" \
      --target-dataset \"$TARGET_DATASET\" \
      --label-field \"$LABEL_FIELD\" \
      --max-rows-per-split \"$MAX_ROWS_PER_SPLIT\""
    ;;
  single_xgb)
    run_cmd "$PY_BIN \"$EXPERIMENTS_DIR/run_xgb_sequential_track1.py\" \
      --mode single \
      --dataset-dir-glob \"$DATASET_GLOB\" \
      --output-dir \"$OUT_ROOT\" \
      --label-field \"$LABEL_FIELD\" \
      --supervised-model \"$SUPERVISED_MODEL\""
    ;;
  lodo_xgb)
    run_cmd "$PY_BIN \"$EXPERIMENTS_DIR/run_lodo_xgb_track1.py\" \
      --dataset-dir-glob \"$DATASET_GLOB\" \
      --output-dir \"$OUT_ROOT\" \
      --label-field \"$LABEL_FIELD\" \
      --supervised-model \"$SUPERVISED_MODEL\""
    ;;
  crossnode_xgb)
    run_cmd "$PY_BIN \"$EXPERIMENTS_DIR/run_cross_node_xgb_track1.py\" \
      --dataset-dir-glob \"$DATASET_GLOB\" \
      --output-dir \"$OUT_ROOT\" \
      --source-platform \"$SOURCE_PLATFORM\" \
      --target-platform \"$TARGET_PLATFORM\" \
      --label-field \"$LABEL_FIELD\" \
      --supervised-model \"$SUPERVISED_MODEL\""
    ;;
  *)
    echo "[baseline] unsupported BASELINE_MODE=$BASELINE_MODE" >&2
    exit 2
    ;;
esac

echo "[baseline] done"
