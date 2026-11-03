#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WF_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RC_ROOT="$(cd "$WF_ROOT/.." && pwd)"
ROOT_DEFAULT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
ROOT="${ROOT:-$RC_ROOT}"
PACKAGE_ROOT="${PACKAGE_ROOT:-$RC_ROOT}"
PY_BIN="${PY_BIN:-python3}"
DRY_RUN=0

if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

MODE="${MODE:-all}" # zeroshot|single|lodo|crossnode|all
DATASET_GLOB="${DATASET_GLOB:-$PACKAGE_ROOT/data/*}"
OUT_ROOT="${OUT_ROOT:-$WF_ROOT/runs/qwen}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$RC_ROOT/scripts/experiments}"
LABEL_FIELD="${LABEL_FIELD:-strict_success}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
MAX_LEN="${MAX_LEN:-1536}"
EPOCHS="${EPOCHS:-1.0}"
LR="${LR:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
SEED="${SEED:-42}"
MAX_FSR="${MAX_FSR:-0.10}"
LAMBDA_FALSE_STOP="${LAMBDA_FALSE_STOP:-1.0}"

run_cmd() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run] %s\n' "$*"
  else
    eval "$@"
  fi
}

mkdir -p "$OUT_ROOT"

echo "[qwen] mode=$MODE dataset_glob=$DATASET_GLOB"

for REQUIRED in \
  "$EXPERIMENTS_DIR/run_qwen_zeroshot_local_sequential.py" \
  "$EXPERIMENTS_DIR/run_single_qwen_track1.py" \
  "$EXPERIMENTS_DIR/run_lodo_qwen_track1.py" \
  "$EXPERIMENTS_DIR/run_cross_node_qwen_track1.py" \
  "$EXPERIMENTS_DIR/train_small_llm_track1.py" \
  "$EXPERIMENTS_DIR/eval_finetuned_track1.py"; do
  if [ ! -f "$REQUIRED" ]; then
    echo "[qwen] missing optional experiment script: $REQUIRED" >&2
    echo "[qwen] quick-start default does not require Qwen; optional runs need model dependencies." >&2
    exit 2
  fi
done

if [ "$MODE" = "zeroshot" ] || [ "$MODE" = "all" ]; then
  run_cmd "$PY_BIN \"$EXPERIMENTS_DIR/run_qwen_zeroshot_local_sequential.py\" \
    --dataset-dir-glob \"$DATASET_GLOB\" \
    --output-dir \"$OUT_ROOT/zeroshot_seq\" \
    --label-field \"$LABEL_FIELD\" \
    --model-name \"$MODEL_NAME\" \
    --prompt-style weak --log-style basic \
    --max-success-false-stop-rate \"$MAX_FSR\" \
    --lambda-false-stop \"$LAMBDA_FALSE_STOP\" \
    --seed \"$SEED\""
fi

if [ "$MODE" = "single" ] || [ "$MODE" = "all" ]; then
  run_cmd "$PY_BIN \"$EXPERIMENTS_DIR/run_single_qwen_track1.py\" \
    --dataset-dir-glob \"$DATASET_GLOB\" \
    --output-dir \"$OUT_ROOT/single\" \
    --label-field \"$LABEL_FIELD\" \
    --model-name \"$MODEL_NAME\" \
    --max-length \"$MAX_LEN\" \
    --epochs \"$EPOCHS\" --lr \"$LR\" \
    --batch-size \"$BATCH_SIZE\" --grad-accum \"$GRAD_ACCUM\" \
    --seed \"$SEED\" \
    --max-success-false-stop-rate \"$MAX_FSR\" \
    --lambda-false-stop \"$LAMBDA_FALSE_STOP\""
fi

if [ "$MODE" = "lodo" ] || [ "$MODE" = "all" ]; then
  run_cmd "$PY_BIN \"$EXPERIMENTS_DIR/run_lodo_qwen_track1.py\" \
    --dataset-dir-glob \"$DATASET_GLOB\" \
    --output-dir \"$OUT_ROOT/lodo\" \
    --label-field \"$LABEL_FIELD\" \
    --model-name \"$MODEL_NAME\" \
    --max-length \"$MAX_LEN\" \
    --epochs \"$EPOCHS\" --lr \"$LR\" \
    --batch-size \"$BATCH_SIZE\" --grad-accum \"$GRAD_ACCUM\" \
    --seed \"$SEED\" \
    --max-success-false-stop-rate \"$MAX_FSR\" \
    --lambda-false-stop \"$LAMBDA_FALSE_STOP\""
fi

if [ "$MODE" = "crossnode" ] || [ "$MODE" = "all" ]; then
  run_cmd "$PY_BIN \"$EXPERIMENTS_DIR/run_cross_node_qwen_track1.py\" \
    --dataset-dir-glob \"$DATASET_GLOB\" \
    --output-dir \"$OUT_ROOT/crossnode_asap7_to_sky130hd\" \
    --source-platform asap7 --target-platform sky130hd \
    --label-field \"$LABEL_FIELD\" \
    --model-name \"$MODEL_NAME\" \
    --max-length \"$MAX_LEN\" \
    --epochs \"$EPOCHS\" --lr \"$LR\" \
    --batch-size \"$BATCH_SIZE\" --grad-accum \"$GRAD_ACCUM\" \
    --seed \"$SEED\" \
    --max-success-false-stop-rate \"$MAX_FSR\" \
    --lambda-false-stop \"$LAMBDA_FALSE_STOP\""

  run_cmd "$PY_BIN \"$EXPERIMENTS_DIR/run_cross_node_qwen_track1.py\" \
    --dataset-dir-glob \"$DATASET_GLOB\" \
    --output-dir \"$OUT_ROOT/crossnode_sky130hd_to_asap7\" \
    --source-platform sky130hd --target-platform asap7 \
    --label-field \"$LABEL_FIELD\" \
    --model-name \"$MODEL_NAME\" \
    --max-length \"$MAX_LEN\" \
    --epochs \"$EPOCHS\" --lr \"$LR\" \
    --batch-size \"$BATCH_SIZE\" --grad-accum \"$GRAD_ACCUM\" \
    --seed \"$SEED\" \
    --max-success-false-stop-rate \"$MAX_FSR\" \
    --lambda-false-stop \"$LAMBDA_FALSE_STOP\""
fi

echo "[qwen] done"
