"""М6. Пороговые правила (ТЗ, «Правила уведомлений»). Пороги — в config/thresholds.yaml.

Этап 2: П13 «Устаревшие данные» и П14 «Расхождение связанных датчиков».
П2–П11 — этап 3, П12 — этап 7, П15 — этап 6, П16 — этап 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from field_twin.config import parse_interval
from field_twin.quality import QualityReport, pearson_pair


@dataclass
class RuleEvent:
    code: str
    ts: pd.Timestamp
    stream: str | None
    object_id: str | None
    message: str
    recipient: str
    payload: dict[str, Any] = field(default_factory=dict)


def check_p13(report: QualityReport, thresholds: dict[str, Any]) -> list[RuleEvent]:
    """П13: поток не обновлялся дольше двух своих интервалов (оценка целостности данных)."""
    rule = thresholds["П13"]
    events = []
    for f in report.freshness:
        if not f.stale:
            continue
        events.append(RuleEvent(
            code="П13", ts=report.now, stream=f.stream, object_id=None, recipient=rule["recipient"],
            message=(f"Поток «{f.stream}» не обновлялся {_td(f.age)} при интервале {_td(f.interval)}"
                     + (" (интервал оценён по данным)" if f.interval_estimated else "")),
            payload={"last_ts": f.last_ts.isoformat(), "age_s": f.age.total_seconds(),
                     "interval_s": f.interval.total_seconds(), "interval_estimated": f.interval_estimated,
                     "stale_after_intervals": rule["stale_after_intervals"]},
        ))
    return events


def p14_windows(data: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Корреляция Пирсона связанных датчиков по окнам (Шевченко, Рычков).

    Используются только годные значения: без простоев, выбросов и нефизичных значений.
    """
    win = parse_interval((settings.get("pearson") or {}).get("window", "2h"))
    min_points = int((settings.get("pearson") or {}).get("min_points", 12))
    good = data[data["value_clean"].notna() & ~data["f_downtime"] & ~data["f_outlier"]]
    rows = []
    for pair in settings.get("correlated_pairs") or []:
        s = good[good["stream"] == pair["stream"]]
        if s.empty:
            continue
        t0 = s["ts"].min().floor("h")
        t = t0
        while t <= s["ts"].max():
            w = s[(s["ts"] >= t) & (s["ts"] < t + win)]
            n = _common_points(w, pair["a"], pair["b"])
            r = pearson_pair(w, pair["stream"], pair["a"], pair["b"]) if n >= min_points else float("nan")
            rows.append({"stream": pair["stream"], "a": pair["a"], "b": pair["b"], "start": t, "end": t + win,
                         "n": n, "pearson": r})
            t += win
    return pd.DataFrame(rows, columns=["stream", "a", "b", "start", "end", "n", "pearson"])


def check_p14(data: pd.DataFrame, settings: dict[str, Any], thresholds: dict[str, Any]) -> list[RuleEvent]:
    """П14: корреляция связанных параметров ниже порога. Без порога правило выключено."""
    rule = thresholds["П14"]
    thr = rule.get("min_pearson")
    if thr is None:
        return []
    events = []
    for w in p14_windows(data, settings).itertuples():
        if pd.isna(w.pearson) or w.pearson >= thr:
            continue
        obj = w.a.split(".", 1)[0]
        events.append(RuleEvent(
            code="П14", ts=w.end, stream=w.stream, object_id=obj, recipient=rule["recipient"],
            message=f"Расхождение датчиков {w.a} и {w.b}: корреляция {w.pearson:.2f} < {thr} "
                    f"за {w.start:%Y-%m-%d %H:%M}–{w.end:%H:%M} UTC",
            payload={"a": w.a, "b": w.b, "pearson": w.pearson, "threshold": thr, "n": int(w.n),
                     "start": w.start.isoformat(), "end": w.end.isoformat()},
        ))
    return events


def log_events(journal, events: list[RuleEvent], shadow: bool = False) -> list[int]:
    """Записать срабатывания в журнал (М8)."""
    return [
        journal.log_event("rule", code=e.code, message=e.message, ts=e.ts.to_pydatetime(), stream=e.stream,
                          object_id=e.object_id, payload={**e.payload, "recipient": e.recipient}, shadow=shadow)
        for e in events
    ]


def _common_points(w: pd.DataFrame, a: str, b: str) -> int:
    ta = set(w[(w["object_id"] + "." + w["parameter"]) == a]["ts"])
    tb = set(w[(w["object_id"] + "." + w["parameter"]) == b]["ts"])
    return len(ta & tb)


def _td(td: timedelta | None) -> str:
    if td is None:
        return "—"
    m = int(td.total_seconds() // 60)
    return f"{m // 60} ч {m % 60} мин" if m >= 60 else f"{m} мин"
