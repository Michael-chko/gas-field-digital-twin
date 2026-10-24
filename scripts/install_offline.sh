#!/usr/bin/env bash
# На машине БЕЗ интернета (Linux, в том числе РЕД ОС): установка из папки wheels/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
$PY -m venv .venv
. .venv/bin/activate
pip install --no-index --find-links wheels setuptools wheel
pip install --no-index --find-links wheels -r requirements-dev.txt
pip install --no-index --find-links wheels --no-build-isolation --no-deps -e .
echo "Готово. Активируйте окружение: . .venv/bin/activate; проверка: pytest"
