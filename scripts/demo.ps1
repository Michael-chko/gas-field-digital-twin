# Демонстрация (Windows): синтетика → обучение моделей → проигрывание архива с экранами.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
field-twin-generate --out data\train --days 8 --hydrate-episodes 4
field-twin-ingest data\train\absorber.csv data\train\methanol.csv --db data\train.sqlite | Out-Null
field-twin-hydrates --db data\train.sqlite --truth data\train\truth.json --trials 10
field-twin-chudin --synthetic-days 400
field-twin-generate --out data\demo --days 2 --seed 5
Remove-Item data\playback.sqlite* -ErrorAction SilentlyContinue
$files = Get-ChildItem data\demo\*.csv | ForEach-Object { $_.FullName }
field-twin-playback @files --db data\playback.sqlite --speed 600 --cycle-minutes 30 --serve
