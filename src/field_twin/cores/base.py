"""М9. Единый интерфейс адаптера внешнего расчётного ядра (Войтенков; Кулагин и соавторы; ИРМ).

Адаптер получает текущие очищенные замеры и возвращает расчётные параметры и варианты режима.
Рекомендации ядер проходят тот же путь, что сигналы моделей: правило П12, уведомление,
решение главного инженера, журнал. Ядро ничего не пишет в АСУ ТП.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class CoreInput:
    now: pd.Timestamp
    measurements: dict[str, float]            # "поток.объект.параметр" -> последнее очищенное значение
    timestamps: dict[str, pd.Timestamp] = field(default_factory=dict)

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.measurements.get(key, default)


@dataclass
class Change:
    parameter: str          # что менять (управляющее воздействие или распределение потока)
    current: float | None
    proposed: float | None
    unit: str = ""


@dataclass
class Recommendation:
    key: str                # устойчивый идентификатор рекомендации (для журнала и повторов)
    title: str
    message: str
    object_id: str | None = None
    changes: list[Change] = field(default_factory=list)
    expected_effect: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreResult:
    core: str
    ts: pd.Timestamp
    status: str = "ok"      # ok | not_connected | error
    calculated: dict[str, float] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)
    error: str | None = None


class CoreAdapter(ABC):
    """Адаптер ядра. Обязателен только calculate; connect/close — по необходимости."""

    name = "core"

    def __init__(self, options: dict[str, Any] | None = None):
        self.options = options or {}

    def connect(self) -> None:
        """Подключиться к ядру (лицензия, сеанс, открытие расчётного файла)."""

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def calculate(self, inp: CoreInput) -> CoreResult:
        """Передать ядру замеры, получить расчётные параметры и варианты режима."""

    def close(self) -> None:
        """Закрыть сеанс ядра."""


def load_class(path: str):
    module, cls = path.split(":")
    return getattr(importlib.import_module(module), cls)


def load_cores(cfg: dict[str, Any], only_enabled: bool = True) -> list[CoreAdapter]:
    """Ядра из config/cores.yaml: класс задаётся строкой — замена без правки кода."""
    out = []
    for c in cfg.get("cores", []):
        if only_enabled and not c.get("enabled", True):
            continue
        core = load_class(c["class"])(c.get("options") or {})
        core.name = c.get("name", core.name)
        out.append(core)
    return out


def run_cores(cores: list[CoreAdapter], inp: CoreInput) -> list[CoreResult]:
    """Вызов ядер; ошибка одного ядра не останавливает цикл двойника."""
    results = []
    for core in cores:
        try:
            if not core.is_available():
                results.append(CoreResult(core.name, inp.now, status="not_connected"))
                continue
            results.append(core.calculate(inp))
        except NotImplementedError as e:
            results.append(CoreResult(core.name, inp.now, status="not_connected", error=str(e)))
        except Exception as e:   # noqa: BLE001 — ядро внешнее, его сбой не должен ронять двойник
            results.append(CoreResult(core.name, inp.now, status="error", error=f"{type(e).__name__}: {e}"))
    return results


def core_input(data: pd.DataFrame, now: pd.Timestamp) -> CoreInput:
    """Последние очищенные значения по каждому сигналу."""
    d = data[data["value_clean"].notna()].sort_values("ts").groupby(["stream", "object_id", "parameter"]).tail(1)
    keys = d["stream"] + "." + d["object_id"] + "." + d["parameter"]
    return CoreInput(now, dict(zip(keys, d["value_clean"].astype(float))), dict(zip(keys, d["ts"])))
