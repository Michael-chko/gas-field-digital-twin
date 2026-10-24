"""М10. Технологические остановки скважин (Савченко и соавторы).

- остановка: в статье — по нулевому току насоса, для газовых скважин — по нулевому дебиту;
- проверяется минимальная длительность (в статье 4 ч для высокопроницаемых и 12 ч для
  низкопроницаемых объектов) и монотонный рост давления за время остановки;
- найденные остановки сохраняются как кандидаты для оценки пластового давления.

Регрессии из статьи (y = a·ln(x) + b и y = a·ln(c·x / (c + x)) + b) выведены для нефтяных
скважин с насосами. Здесь только поиск и разметка остановок; оценка давления — после проверки
применимости к газовым скважинам («Открытые вопросы»).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from field_twin import config
from field_twin.gateway.tag_dictionary import TagDictionary


@dataclass
class WellStop:
    object_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    hours: float
    min_hours: float
    pressure_rise: float | None
    monotonic: bool

    @property
    def long_enough(self) -> bool:
        return self.hours >= self.min_hours

    @property
    def candidate(self) -> bool:
        """Кандидат для оценки пластового давления."""
        return self.long_enough and self.monotonic


def settings(config_dir=None) -> dict[str, Any]:
    return config.load_yaml("models.yaml", config_dir)["stops"]


def find_stops(data: pd.DataFrame, tags: TagDictionary, s: dict[str, Any]) -> list[WellStop]:
    wells = data[data["stream"] == s["stream"]]
    out = []
    for obj, w in wells.groupby("object_id"):
        rate = w[w["parameter"] == s["rate_parameter"]].sort_values("ts").set_index("ts")["value"]
        if rate.empty:
            continue
        sig = tags.get(s["stream"], obj, s["rate_parameter"])
        perm = (sig.object_attrs.get("permeability") if sig else None) or "high"
        min_h = float(s["min_hours"][perm])
        step = rate.index.to_series().diff().median()
        zero = (rate == 0)
        # пропуск между двумя нулями считается частью остановки
        zero = zero | (rate.isna() & zero.shift(1, fill_value=False) & zero.shift(-1, fill_value=False))
        p = w[w["parameter"] == s["pressure_parameter"]].sort_values("ts").set_index("ts")
        pcol = "value_smooth" if s.get("use_smoothed") and "value_smooth" in p and p["value_smooth"].notna().any() else "value_clean"
        runs, start = [], None
        idx = list(rate.index)
        for i, z in enumerate(zero.to_numpy()):
            if z and start is None:
                start = i
            if (not z or i == len(idx) - 1) and start is not None:
                end = i if z else i - 1
                runs.append((start, end))
                start = None
        for a, b in runs:
            t0, t1 = idx[a], idx[b]
            hours = (b - a + 1) * step.total_seconds() / 3600
            pw = p.loc[t0:t1, pcol].dropna().to_numpy()
            mono = bool(len(pw) >= 2 and np.all(np.diff(pw) > 0))
            out.append(WellStop(obj, t0, t1, hours, min_h, float(pw[-1] - pw[0]) if len(pw) >= 2 else None, mono))
    return out


def save_stops(conn: sqlite3.Connection, stops: list[WellStop]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.executemany(
        """INSERT INTO well_stops(object_id, start_utc, end_utc, hours, min_hours, pressure_rise, monotonic,
               candidate, found_at) VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(object_id, start_utc) DO UPDATE SET end_utc = excluded.end_utc, hours = excluded.hours,
               pressure_rise = excluded.pressure_rise, monotonic = excluded.monotonic, candidate = excluded.candidate""",
        [(s.object_id, s.start.isoformat(), s.end.isoformat(), s.hours, s.min_hours, s.pressure_rise,
          int(s.monotonic), int(s.candidate), now) for s in stops])
    conn.commit()
    return cur.rowcount


def candidates(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM well_stops WHERE candidate = 1 ORDER BY start_utc")]
