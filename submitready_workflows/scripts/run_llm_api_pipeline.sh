#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WF_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RC_ROOT="$(cd "$WF_ROOT/.." && pwd)"
ROOT_DEFAULT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
ROOT="${ROOT:-$RC_ROOT}"
PY_BIN="${PY_BIN:-python3}"
DRY_RUN=0

if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

PACKAGE_ROOT="${PACKAGE_ROOT:-$RC_ROOT}"
DATASET_GLOB="${DATASET_GLOB:-$PACKAGE_ROOT/data/sky130hd_jpeg_encoder}"
OUT_ROOT="${OUT_ROOT:-$WF_ROOT/runs/llm_api}"
SPLIT="${SPLIT:-test}"

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-5.5}"          # user can override
PROVIDER_KEY="${PROVIDER_KEY:-${API_KEY:-}}" # must be set for real API runs
API_BASE="${API_BASE:-}"            # optional
PROMPT_STYLE="${PROMPT_STYLE:-strong}"
LOG_STYLE="${LOG_STYLE:-enhanced}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"     # 0 means full split; default to a small API example
SEED="${SEED:-42}"
LAMBDA_FALSE_STOP="${LAMBDA_FALSE_STOP:-1.0}"

API_SCRIPT="${API_SCRIPT:-$RC_ROOT/scripts/experiments/run_llm_api_only.py}"

run_cmd() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run] %s\n' "$*"
  else
    eval "$@"
  fi
}

if [ "$DRY_RUN" -ne 1 ] && [ -z "$PROVIDER_KEY" ]; then
  echo "PROVIDER_KEY is required for API runs." >&2
  exit 2
fi

if [ -z "$API_SCRIPT" ]; then
  echo "[llm_api] API_SCRIPT is required for optional API experiments." >&2
  echo "[llm_api] quick-start default does not require API calls." >&2
  exit 2
fi

if [ ! -f "$API_SCRIPT" ]; then
  echo "[llm_api] missing optional API experiment script: $API_SCRIPT" >&2
  echo "[llm_api] quick-start default does not require API calls." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"

for DS in $DATASET_GLOB; do
  [ -d "$DS" ] || continue
  NAME="$(basename "$DS")"
  TRACK1="$DS/track1_strict"
  OUT_DS="$OUT_ROOT/$NAME"
  mkdir -p "$OUT_DS"

  EXTRA_MAX=""
  if [ "$MAX_SAMPLES" != "0" ]; then
    EXTRA_MAX="--max-samples $MAX_SAMPLES"
  fi

  run_cmd "$PY_BIN \"$API_SCRIPT\" \
    --track1-dir \"$TRACK1\" \
    --output-dir \"$OUT_DS\" \
    --package-root \"$PACKAGE_ROOT\" \
    --split \"$SPLIT\" \
    --provider \"$PROVIDER\" \
    --model \"$MODEL\" \
    --api-key \"$PROVIDER_KEY\" \
    --api-base \"$API_BASE\" \
    --prompt-style \"$PROMPT_STYLE\" \
    --log-style \"$LOG_STYLE\" \
    --seed \"$SEED\" \
    --lambda-false-stop \"$LAMBDA_FALSE_STOP\" \
    $EXTRA_MAX"
done

echo "[llm_api] done"
