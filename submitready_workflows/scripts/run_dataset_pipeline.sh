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

PACKAGE_ROOT="${PACKAGE_ROOT:-${OUT_PACKAGE:-$RC_ROOT}}"
VERIFY_JSON="${VERIFY_JSON:-$RC_ROOT/reports/release_package_full_verification.json}"
REQUIRE_MATERIALIZED_LOGS="${REQUIRE_MATERIALIZED_LOGS:-0}"

run_cmd() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run] %s\n' "$*"
  else
    eval "$@"
  fi
}

echo "[dataset] package_root=$PACKAGE_ROOT"

EXTRA_VERIFY=""
if [ "$REQUIRE_MATERIALIZED_LOGS" = "1" ]; then
  EXTRA_VERIFY="--require-materialized-logs"
fi

run_cmd "$PY_BIN \"$RC_ROOT/scripts/verify_release_package.py\" --package-root \"$PACKAGE_ROOT\" --write-report \"$VERIFY_JSON\" --report-package-root \"$PACKAGE_ROOT\" $EXTRA_VERIFY"

echo "[dataset] done"
