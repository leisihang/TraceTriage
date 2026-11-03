#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${SOURCE_DATA_ROOT:?set SOURCE_DATA_ROOT to the private/source data directory}"
: "${SOURCE_PACKAGE_ROOT:?set SOURCE_PACKAGE_ROOT to the private/source package directory}"
: "${SOURCE_LOG_ROOT:?set SOURCE_LOG_ROOT to the source OpenROAD logs directory}"
: "${SOURCE_REPORT_ROOT:?set SOURCE_REPORT_ROOT to the source OpenROAD reports directory}"

OUT="${OUT:-$RC_ROOT/release_package_smoke}"

rm -rf "$OUT"
mkdir -p "$OUT"

python3 "$SCRIPT_DIR/sanitize_and_materialize_release.py" \
  --source-data-root "$SOURCE_DATA_ROOT" \
  --source-package-root "$SOURCE_PACKAGE_ROOT" \
  --source-log-root "$SOURCE_LOG_ROOT" \
  --source-report-root "$SOURCE_REPORT_ROOT" \
  --output-root "$OUT" \
  --max-rows-per-dataset 200

echo "[done] smoke release package built: $OUT"
