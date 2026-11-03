#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-$RC_ROOT/release_package_full}"
OUT_DIR="$RC_ROOT/indexes"

mkdir -p "$OUT_DIR"

{
  printf 'rel_path\n'
  find "$PACKAGE_ROOT" -type f | sed "s#^$PACKAGE_ROOT/##" | sort
} > "$OUT_DIR/release_full_file_index.csv"

echo "[done] regenerated quick-start indexes under $OUT_DIR"
