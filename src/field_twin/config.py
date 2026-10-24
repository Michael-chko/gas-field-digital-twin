"""Загрузка конфигурации из папки config/ (ТЗ, «Технические решения прототипа»)."""

from __future__ import annotations

import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = Path(os.environ.get("FIELD_TWIN_CONFIG", ROOT / "config"))

_INTERVAL_RE = re.compile(r"^\s*(\d+)\s*(s|min|h|d)\s*$")
_UNITS = {"s": "seconds", "min": "minutes", "h": "hours", "d": "days"}


def load_yaml(name: str, config_dir: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_dir or DEFAULT_CONFIG_DIR) / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_interval(text: str | None) -> timedelta | None:
    """'2min', '5min', '1h', '1d' -> timedelta; None -> None (интервал не указан)."""
    if text is None:
        return None
    m = _INTERVAL_RE.match(str(text))
    if not m:
        raise ValueError(f"Неверный интервал: {text!r}; ожидается вида 2min, 1h, 1d")
    return timedelta(**{_UNITS[m.group(2)]: int(m.group(1))})


def thresholds(config_dir: Path | str | None = None) -> dict[str, Any]:
    return load_yaml("thresholds.yaml", config_dir)


def quality(config_dir: Path | str | None = None) -> dict[str, Any]:
    return load_yaml("quality.yaml", config_dir)


def simulator(config_dir: Path | str | None = None) -> dict[str, Any]:
    return load_yaml("simulator.yaml", config_dir)


def anomaly_reasons(config_dir: Path | str | None = None) -> list[dict[str, Any]]:
    return load_yaml("anomaly_reasons.yaml", config_dir).get("reasons", [])
