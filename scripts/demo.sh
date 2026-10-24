#!/usr/bin/env bash
# Демонстрация от начала до конца: синтетика → обучение моделей → проигрывание архива с экранами.
set -euo pipefail
cd "$(dirname "$0")/.."
field-twin-generate --out data/train --days 8 --hydrate-episodes 4
field-twin-ingest data/train/absorber.csv data/train/methanol.csv --db data/train.sqlite > /dev/null
field-twin-hydrates --db data/train.sqlite --truth data/train/truth.json --trials 10
field-twin-chudin --synthetic-days 400
field-twin-generate --out data/demo --days 2 --seed 5
rm -f data/playback.sqlite*
field-twin-playback data/demo/*.csv --db data/playback.sqlite --speed 600 --cycle-minutes 30 --serve
