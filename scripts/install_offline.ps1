# На машине БЕЗ интернета (Windows): установка из папки wheels\.
# Запуск: powershell -ExecutionPolicy Bypass -File scripts\install_offline.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --no-index --find-links wheels setuptools wheel
& .\.venv\Scripts\python.exe -m pip install --no-index --find-links wheels -r requirements-dev.txt
& .\.venv\Scripts\python.exe -m pip install --no-index --find-links wheels --no-build-isolation --no-deps -e .
Write-Host "Готово. Активируйте окружение: .\.venv\Scripts\Activate.ps1; проверка: pytest"
