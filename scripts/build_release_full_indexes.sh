#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG="$BASE/release_package_full"
OUT_DIR="$BASE/indexes"

mkdir -p "$OUT_DIR"

if [ ! -d "$PKG" ]; then
  echo "missing package: $PKG" >&2
  exit 1
fi

OUT_CSV="$OUT_DIR/release_full_file_index.csv"
OUT_SHA="$OUT_DIR/release_full_sha256.txt"

printf 'relative_path,size_bytes\n' > "$OUT_CSV"
find "$PKG" -type f | sort | while IFS= read -r f; do
  rel="${f#$PKG/}"
  sz="$(wc -c < "$f" | tr -d ' ')"
  printf '%s,%s\n' "$rel" "$sz" >> "$OUT_CSV"
done

if [ "${WITH_SHA256:-0}" = "1" ]; then
  find "$PKG" -type f | sort | while IFS= read -r f; do
    rel="${f#$PKG/}"
    sha="$(sha256sum "$f" | awk '{print $1}')"
    printf '%s  %s\n' "$sha" "$rel"
  done > "$OUT_SHA"
  echo "[done] wrote:"
  echo "  $OUT_CSV"
  echo "  $OUT_SHA"
  wc -l "$OUT_CSV" "$OUT_SHA"
else
  echo "[done] wrote:"
  echo "  $OUT_CSV"
  echo "[skip] release_full_sha256.txt (set WITH_SHA256=1 to enable full-file hashing)"
  wc -l "$OUT_CSV"
fi
