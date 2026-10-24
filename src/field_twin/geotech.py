"""М5. Геотехнический мониторинг оснований (Затырко и соавторы).

- класс каждого параметра каждого пункта контроля относительно предельного значения:
  ниже 2/3 предела — зелёный «ниже предельного»; от 2/3 до предела — жёлтый «ниже предельного,
  внимание!»; на пределе и выше — красный «выше предельного» (правило П2);
- оценка по каждому параметру и по объекту в целом: худший класс среди параметров
  (в источнике правило сведения не приведено — решение прототипа);
- тренд осадки и скорость вертикального смещения против 15 мм/год (П3);
- неравномерность осадки фундаментов обвязки против 2 мм/м (П4);
- дата последнего радарного замера: зимой радара нет, оценка держится на контактных датчиках.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from field_twin.gateway.tag_dictionary import TagDictionary

GREEN, YELLOW, RED = "green", "yellow", "red"
TITLES = {GREEN: "ниже предельного", YELLOW: "ниже предельного, внимание!", RED: "выше предельного"}
ORDER = {GREEN: 0, YELLOW: 1, RED: 2}


def classify(value: float, limit: float, yellow_from=(2, 3), red_from=(1, 1)) -> str:
    """Класс значения относительно предела. Граница 2/3 относится к жёлтому, предел — к красному."""
    if limit is None or limit <= 0:
        raise ValueError("Предельное значение должно быть положительным")
    v = abs(float(value))
    eps = 1e-9 * limit
    # Сравнение в целых долях, чтобы граница 2/3 не терялась на округлении.
    if v * red_from[1] >= red_from[0] * limit - eps:
        return RED
    if v * yellow_from[1] >= yellow_from[0] * limit - eps:
        return YELLOW
    return GREEN


@dataclass
class ParameterState:
    object_id: str
    parameter: str
    ts: pd.Timestamp
    value: float
    limit: float
    ratio: float
    color: str


@dataclass
class ObjectState:
    object_id: str
    color: str
    rate_mm_per_year: float | None
    trend_span_days: float
    unevenness: float | None
    last_radar: pd.Timestamp | None
    last_sensor: pd.Timestamp | None


@dataclass
class GeotechAssessment:
    parameters: list[ParameterState] = field(default_factory=list)
    objects: list[ObjectState] = field(default_factory=list)
    last_radar: pd.Timestamp | None = None

    def to_text(self) -> str:
        lines = ["Геотехнический мониторинг оснований"]
        for o in self.objects:
            rate = f"{o.rate_mm_per_year:.1f} мм/год" if o.rate_mm_per_year is not None else \
                f"нет (ряд {o.trend_span_days:.0f} сут)"
            radar = o.last_radar.strftime("%Y-%m-%d") if o.last_radar is not None else "нет"
            lines.append(f"  {o.object_id}: {TITLES[o.color]}; скорость осадки {rate}; "
                         f"неравномерность {o.unevenness if o.unevenness is not None else '—'} мм/м; "
                         f"последний радарный замер {radar}")
            for p in self.parameters:
                if p.object_id == o.object_id:
                    lines.append(f"    {p.parameter}: {p.value:.3g} из {p.limit:g} ({p.ratio:.0%}) — {TITLES[p.color]}")
        return "\n".join(lines)


def assess(data: pd.DataFrame, tags: TagDictionary, thresholds: dict[str, Any], now: pd.Timestamp) -> GeotechAssessment:
    p2, p3 = thresholds["П2"], thresholds["П3"]
    yf, rf = tuple(p2["yellow_from"]), tuple(p2["red_from"])
    good = data[data["value_clean"].notna()]
    sensors = good[good["stream"] == "geotech"]
    radar = good[good["stream"] == "geotech_radar"]
    out = GeotechAssessment()
    objects = sorted({s.object_id for s in tags.stream_signals("geotech")})
    for obj in objects:
        colors = []
        for sig in tags.stream_signals("geotech"):
            if sig.object_id != obj:
                continue
            limit = (sig.object_attrs.get("limits") or {}).get(sig.parameter)
            sub = sensors[(sensors.object_id == obj) & (sensors.parameter == sig.parameter)]
            if limit is None or sub.empty:
                continue
            last = sub.loc[sub["ts"].idxmax()]
            color = classify(last["value_clean"], limit, yf, rf)
            colors.append(color)
            out.parameters.append(ParameterState(obj, sig.parameter, last["ts"], float(last["value_clean"]),
                                                 float(limit), abs(float(last["value_clean"])) / limit, color))
        settle = pd.concat([sensors[(sensors.object_id == obj) & (sensors.parameter == "settlement")],
                            radar[(radar.object_id == obj) & (radar.parameter == "settlement")]])
        rate, span = None, 0.0
        if len(settle) >= 3:
            span = (settle["ts"].max() - settle["ts"].min()).total_seconds() / 86400
            if span >= p3.get("trend_min_days", 30):
                years = (settle["ts"] - settle["ts"].min()).dt.total_seconds().to_numpy() / (365.25 * 86400)
                rate = float(np.polyfit(years, settle["value_clean"].to_numpy(dtype=float), 1)[0])
        unev = sensors[(sensors.object_id == obj) & (sensors.parameter == "settlement_unevenness")]
        r = radar[radar.object_id == obj]
        s_obj = sensors[sensors.object_id == obj]
        out.objects.append(ObjectState(
            obj, max(colors, key=ORDER.get) if colors else GREEN, rate, span,
            float(unev.loc[unev["ts"].idxmax(), "value_clean"]) if len(unev) else None,
            r["ts"].max() if len(r) else None, s_obj["ts"].max() if len(s_obj) else None,
        ))
    out.last_radar = radar["ts"].max() if len(radar) else None
    return out
