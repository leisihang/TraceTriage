#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARTIFACT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="${SRC_DIR:-$ARTIFACT_ROOT/dataverse_upload}"
OUT_DIR="${OUT_DIR:-$ARTIFACT_ROOT/kaggle_upload}"
KAGGLE_OWNER="${KAGGLE_OWNER:-leisihang}"
KAGGLE_SLUG="${KAGGLE_SLUG:-tracetriage}"

mkdir -p "$OUT_DIR"

cp "$ARTIFACT_ROOT/DATAVERSE_FILE_GUIDE.md" "$OUT_DIR/"
cp "$SRC_DIR/tracetriage_docs_scripts_metadata_v2.tar.gz" "$OUT_DIR/"
cp "$SRC_DIR/tracetriage_release_package_full_data.tar.gz" "$OUT_DIR/"
cp "$SRC_DIR/tracetriage_release_package_full_logs.tar.gz.part000" "$OUT_DIR/"

shopt -s nullglob
for f in "$SRC_DIR"/tracetriage_release_package_full_reports.tar.gz.part*; do
  cp "$f" "$OUT_DIR/"
done
shopt -u nullglob

(
  cd "$OUT_DIR"
  sha256sum tracetriage_*.tar.gz tracetriage_*.tar.gz.part* > checksums_sha256.txt
)

cat > "$OUT_DIR/dataset-metadata.json" <<JSON
{
  "title": "TraceTriage",
  "id": "${KAGGLE_OWNER}/${KAGGLE_SLUG}",
  "licenses": [
    {
      "name": "CC-BY-SA-4.0"
    }
  ],
  "subtitle": "Sanitized EDA flow triage dataset",
  "description": "TraceTriage is a sanitized EDA flow triage dataset for early stop/continue prediction over OpenROAD-based digital design runs. The Kaggle upload uses archives because the restored artifact contains more than 300,000 files. See DATAVERSE_FILE_GUIDE.md for archive contents, restore commands, checksums, and verification instructions."
}
JSON

echo "[kaggle] prepared $OUT_DIR"
find "$OUT_DIR" -maxdepth 1 -type f -printf '%f %s\n' | sort
