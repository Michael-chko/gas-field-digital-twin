"""Технологическая сигнализация имитатора АСУ ТП по уставкам из словаря параметров.

Уровни: alarm_low (LL), warn_low (L), warn_high (H), alarm_high (HH). Событие формируется при
смене состояния (вход в нарушение и возврат в норму), как в SCADA.
"""

from __future__ import annotations

from dataclasses import dataclass

LEVELS = [("alarm_low", "LL", 800), ("warn_low", "L", 500), ("warn_high", "H", 500), ("alarm_high", "HH", 800)]


@dataclass
class Alarm:
    tag: str
    level: str          # LL | L | H | HH | NORMAL
    value: float
    setpoint: float | None
    severity: int
    message: str


class AlarmEngine:
    def __init__(self, setpoints: dict[str, dict]):
        self.setpoints = {k: v for k, v in setpoints.items() if v}
        self.state: dict[str, str] = {}

    def level(self, tag: str, value: float) -> tuple[str, float | None, int]:
        sp = self.setpoints.get(tag) or {}
        if "alarm_low" in sp and value <= sp["alarm_low"]:
            return "LL", sp["alarm_low"], 800
        if "alarm_high" in sp and value >= sp["alarm_high"]:
            return "HH", sp["alarm_high"], 800
        if "warn_low" in sp and value <= sp["warn_low"]:
            return "L", sp["warn_low"], 500
        if "warn_high" in sp and value >= sp["warn_high"]:
            return "H", sp["warn_high"], 500
        return "NORMAL", None, 100

    def evaluate(self, tag: str, value: float | None) -> Alarm | None:
        if value is None or tag not in self.setpoints:
            return None
        lvl, sp, sev = self.level(tag, value)
        prev = self.state.get(tag, "NORMAL")
        if lvl == prev:
            return None
        self.state[tag] = lvl
        if lvl == "NORMAL":
            return Alarm(tag, lvl, value, None, sev, f"{tag}: возврат в норму ({value:g})")
        return Alarm(tag, lvl, value, sp, sev, f"{tag}: {lvl} — значение {value:g}, уставка {sp:g}")
