#!/usr/bin/env bash
# На машине С интернетом: скачать колёса для установки без интернета.
# Колёса должны соответствовать целевой ОС и версии Python.
#   ./scripts/download_wheels.sh                 — для этой же ОС и Python
#   PLATFORM=win_amd64 PY=3.12 ./scripts/download_wheels.sh   — для Windows x64, Python 3.12
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-wheels}
mkdir -p "$OUT"
if [[ -n "${PLATFORM:-}" ]]; then
  pip download -d "$OUT" -r requirements-dev.txt --only-binary=:all: --platform "$PLATFORM" --python-version "${PY:-3.12}"
  pip download -d "$OUT" --only-binary=:all: --platform "$PLATFORM" --python-version "${PY:-3.12}" "setuptools>=68" wheel
else
  pip download -d "$OUT" -r requirements-dev.txt
  pip download -d "$OUT" "setuptools>=68" wheel
fi
echo "Колёса в $OUT/. Перенесите папку проекта вместе с $OUT/ на целевую машину."
