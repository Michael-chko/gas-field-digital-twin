"""М11. Стык с заводом (продольный разбор технологий «месторождение + завод»).

Формат обмена по пяти потокам стыка:
1. состав газа и остаток C5+ (пласт и УКПГ → завод);
2. объём газа (пласт и сбор → завод);
3. раздельные потоки — с этаном и сухой (сбор → газохимия; СПГ и аммиак);
4. доступная мощность приводов (ГПА → оба двойника);
5. окна снабжения и ремонтов (логистика → оба двойника).

Обратный поток «завод → промысел» (например, снижение приёма из-за ремонта линии, чтобы промысел
заранее менял режим кустов): в материалах такой связи нет — здесь формат и заглушка.

Остаток C5+ выше допуска завода — правило П16 (тяжёлые компоненты вымерзают в теплообменниках
сжижения). Порог задаёт пользователь.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from field_twin import config

FORMAT_VERSION = "1.0"


@dataclass
class Composition:
    c5plus_g_m3: float | None
    methane_mol_pct: float | None
    measured_at: str | None
    source: str = "gas_composition"     # измерения на потоке в материалах не описаны («Открытые вопросы»)


@dataclass
class GasVolume:
    flow_thousand_m3_h: float | None
    measured_at: str | None


@dataclass
class SeparateStreams:
    ethane_rich_thousand_m3_h: float | None = None   # с этаном → газохимия (Долгов, Пожидаев)
    dry_thousand_m3_h: float | None = None           # сухой → СПГ и аммиак
    note: str = "раздельный сбор в данных прототипа не измеряется"


@dataclass
class DrivePower:
    available_mw: dict[str, float] = field(default_factory=dict)   # агрегат → доступная мощность


@dataclass
class Window:
    kind: str            # supply | repair
    start: str
    end: str
    note: str = ""


@dataclass
class FieldToPlant:
    """Сообщение «промысел → завод»."""
    ts: str
    composition: Composition
    volume: GasVolume
    separate: SeparateStreams
    drive_power: DrivePower
    windows: list[Window]
    format_version: str = FORMAT_VERSION
    direction: str = "field_to_plant"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class PlantToField:
    """Сообщение «завод → промысел» (обратный поток; в материалах нет — формат прототипа)."""
    ts: str
    intake_limit_pct: float          # допустимый приём газа заводом, % от нормального
    start: str
    end: str
    reason: str
    format_version: str = FORMAT_VERSION
    direction: str = "plant_to_field"

    @classmethod
    def from_json(cls, text: str) -> "PlantToField":
        d = json.loads(text)
        if d.get("direction") != "plant_to_field":
            raise ValueError("Ожидается сообщение «завод → промысел»")
        if d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Версия формата {d.get('format_version')!r}, ожидается {FORMAT_VERSION}")
        if not 0 <= float(d["intake_limit_pct"]) <= 100:
            raise ValueError("intake_limit_pct должен быть от 0 до 100")
        for k in ("ts", "start", "end"):
            if pd.Timestamp(d[k]).tzinfo is None:
                raise ValueError(f"{k}: время без часового пояса")
        return cls(**{k: d[k] for k in ("ts", "intake_limit_pct", "start", "end", "reason")})


def _last(data: pd.DataFrame, stream: str, obj: str | None, param: str) -> tuple[float | None, str | None]:
    d = data[(data["stream"] == stream) & (data["parameter"] == param) & data["value_clean"].notna()]
    if obj:
        d = d[d["object_id"] == obj]
    if d.empty:
        return None, None
    r = d.loc[d["ts"].idxmax()]
    return float(r["value_clean"]), r["ts"].isoformat()


def build_message(data: pd.DataFrame, now: pd.Timestamp, config_dir=None) -> FieldToPlant:
    s = config.load_yaml("plant_link.yaml", config_dir)
    c5, c5_ts = _last(data, "gas_composition", None, "C5plus")
    ch4, _ = _last(data, "gas_composition", None, "methane")
    vol, vol_ts = _last(data, *s["volume_signal"])
    power = {}
    for unit, (stream, obj, param) in (s.get("drive_power_signals") or {}).items():
        v, _ = _last(data, stream, obj, param)
        if v is not None:
            power[unit] = v
    windows = [Window(**w) for w in s.get("windows", [])]
    return FieldToPlant(now.isoformat(), Composition(c5, ch4, c5_ts), GasVolume(vol, vol_ts), SeparateStreams(),
                        DrivePower(power), windows)


def handle_plant_message(msg: PlantToField) -> dict[str, Any]:
    """Заглушка обратного потока: уведомление главному инженеру заранее изменить режим кустов."""
    return {
        "code": "М11", "recipient": "chief_engineer",
        "message": (f"Завод ограничивает приём газа до {msg.intake_limit_pct:g} % с {msg.start} по {msg.end} "
                    f"({msg.reason}) — заранее изменить режим кустов"),
        "payload": asdict(msg),
    }
