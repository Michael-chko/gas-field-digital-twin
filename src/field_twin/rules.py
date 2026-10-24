"""М6. Пороговые правила (ТЗ, «Правила уведомлений»). Пороги — в config/thresholds.yaml.

П2–П4 — по оценке геотехники (М5), П5–П11 — по рядам очищенных значений (М2),
П13 и П14 — по отчёту о качестве. П1 — модуль notify (М7), П12 — внешние ядра (М9),
П15 — модели М4, П16 — стык с заводом (М11).

Правила по рядам срабатывают в начале эпизода нарушения, а не на каждой точке.
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


def recipient(code: str, thresholds: dict[str, Any]) -> str:
    """Адресат правила; «configurable» — настраиваемый (ТЗ: для П2–П11 получатель настраивается)."""
    r = thresholds[code].get("recipient", "configurable")
    return thresholds.get("configurable_recipient", "chief_engineer") if r == "configurable" else r


def check_p13(report: QualityReport, thresholds: dict[str, Any]) -> list[RuleEvent]:
    """П13: поток не обновлялся дольше двух своих интервалов (оценка целостности данных)."""
    rule = thresholds["П13"]
    events = []
    for f in report.freshness:
        if not f.stale:
            continue
        events.append(RuleEvent(
            code="П13", ts=report.now, stream=f.stream, object_id=None, recipient=recipient("П13", thresholds),
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
            code="П14", ts=w.end, stream=w.stream, object_id=obj, recipient=recipient("П14", thresholds),
            message=f"Расхождение датчиков {w.a} и {w.b}: корреляция {w.pearson:.2f} < {thr} "
                    f"за {w.start:%Y-%m-%d %H:%M}–{w.end:%H:%M} UTC",
            payload={"a": w.a, "b": w.b, "pearson": w.pearson, "threshold": thr, "n": int(w.n),
                     "start": w.start.isoformat(), "end": w.end.isoformat()},
        ))
    return events


# ---------- П2–П4: геотехника (М5) ----------

def check_geotech(assessment, thresholds: dict[str, Any], now: pd.Timestamp) -> list[RuleEvent]:
    from field_twin.geotech import RED, TITLES, YELLOW

    events = []
    for p in assessment.parameters:
        if p.color in (YELLOW, RED):
            events.append(RuleEvent(
                "П2", now, "geotech", p.object_id,
                f"Основание {p.object_id}: {p.parameter} = {p.value:.3g} ({p.ratio:.0%} предела {p.limit:g}) — {TITLES[p.color]}",
                recipient("П2", thresholds),
                {"parameter": p.parameter, "value": p.value, "limit": p.limit, "ratio": p.ratio, "color": p.color,
                 "measured_at": p.ts.isoformat()}))
    max_rate = thresholds["П3"]["max_mm_per_year"]
    max_unev = thresholds["П4"]["max_mm_per_m"]
    for o in assessment.objects:
        if o.rate_mm_per_year is not None and o.rate_mm_per_year > max_rate:
            events.append(RuleEvent(
                "П3", now, "geotech", o.object_id,
                f"Основание {o.object_id}: скорость вертикального смещения {o.rate_mm_per_year:.1f} мм/год > {max_rate}",
                recipient("П3", thresholds), {"rate_mm_per_year": o.rate_mm_per_year, "span_days": o.trend_span_days}))
        if o.unevenness is not None and o.unevenness > max_unev:
            events.append(RuleEvent(
                "П4", now, "geotech", o.object_id,
                f"Основание {o.object_id}: неравномерность осадки {o.unevenness:.2f} мм/м > {max_unev}",
                recipient("П4", thresholds), {"unevenness": o.unevenness}))
    return events


# ---------- П5–П11: ряды очищенных значений ----------

def _good(data: pd.DataFrame, stream: str, parameter: str | None = None) -> pd.DataFrame:
    m = (data["stream"] == stream) & data["value_clean"].notna() & ~data["f_downtime"]
    if parameter:
        m &= data["parameter"] == parameter
    return data[m].sort_values("ts")


def episodes(mask: pd.Series) -> list[tuple[int, int]]:
    """Непрерывные участки True: список (первая, последняя позиция)."""
    out, start = [], None
    arr = mask.to_numpy()
    for i, v in enumerate(arr):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(arr) - 1))
    return out


def _series_rule(data, thresholds, code, stream, parameter, violated, describe) -> list[RuleEvent]:
    events = []
    sub = _good(data, stream, parameter)
    for obj, s in sub.groupby("object_id"):
        s = s.reset_index(drop=True)
        mask = violated(s["value_clean"])
        for a, b in episodes(mask):
            seg = s.iloc[a : b + 1]
            v = float(seg["value_clean"].iloc[0])
            events.append(RuleEvent(
                code, seg["ts"].iloc[0], stream, obj, describe(obj, v), recipient(code, thresholds),
                {"parameter": parameter, "value": v, "episode_start": seg["ts"].iloc[0].isoformat(),
                 "episode_end": seg["ts"].iloc[-1].isoformat(), "points": len(seg)}))
    return events


def check_p5(data, th):
    lim = th["П5"]["max_m_per_s"]
    return _series_rule(data, th, "П5", "pipelines", "gas_velocity", lambda v: v > lim,
                        lambda o, v: f"Газопровод {o}: скорость газа {v:.1f} м/с > {lim} м/с")


def check_p6(data, th):
    lim = th["П6"]["max_c"]
    return _series_rule(data, th, "П6", "gtu", "T_exhaust", lambda v: v > lim,
                        lambda o, v: f"ГТУ {o}: температура за турбиной {v:.0f} °C > {lim} °C")


def check_p8(data, th):
    lo, hi = th["П8"]["surge_margin_min_pct"], th["П8"]["surge_margin_max_pct"]
    return _series_rule(data, th, "П8", "gtu", "surge_margin", lambda v: (v < lo) | (v > hi),
                        lambda o, v: f"ГТУ {o}: запас газодинамической устойчивости {v:.1f} % вне {lo}–{hi} %")


def check_p9(data, th):
    lim = th["П9"]["min_years"]
    return _series_rule(data, th, "П9", "pipelines", "safe_life_years", lambda v: v < lim,
                        lambda o, v: f"Газопровод {o}: срок безопасной эксплуатации {v:g} лет < {lim}")


def check_p10(data, th):
    """Ток или средний защитный потенциал близок или равен нулю (Запевалов и соавторы).

    «Близок» в источнике не определено: без near_zero_abs срабатывает только равенство нулю.
    """
    near = th["П10"].get("near_zero_abs") or 0.0
    out = []
    for p in ("current", "potential"):
        out += _series_rule(data, th, "П10", "ecp", p, lambda v: v.abs() <= near,
                            lambda o, v, p=p: f"ЭХЗ {o}: {'ток' if p == 'current' else 'средний защитный потенциал'} "
                                              f"{v:g} — установка не работает")
    return out


def check_p7(data, th):
    """Ухудшение ГТУ за межремонтный период: мощность −4 %, КПД −2 % (Невский завод)."""
    rule = th["П7"]
    win = pd.Timedelta(hours=rule.get("baseline_hours", 24))
    events = []
    for param, lim, title in (("N_power", rule["max_power_drop_pct"], "мощность"),
                              ("efficiency", rule["max_efficiency_drop_pct"], "КПД")):
        sub = _good(data, "gtu", param)
        for obj, s in sub.groupby("object_id"):
            s = s.set_index("ts")["value_clean"]
            step = s.index.to_series().diff().median() if len(s) > 1 else pd.Timedelta(0)
            # текущее окно не должно перекрываться с базовым
            first_cur = s.index.min() + 2 * win - step
            if s.index.max() < first_cur:
                continue
            base = s[s.index < s.index.min() + win].mean()
            cur = s.rolling(win).mean()
            cur = cur[cur.index >= first_cur]
            drop = (base - cur) / base * 100
            for a, b in episodes(drop > lim):
                t = drop.index[a]
                events.append(RuleEvent(
                    "П7", t, "gtu", obj, f"ГТУ {obj}: {title} снизилась на {drop.iloc[a]:.1f} % > {lim} % от начала периода",
                    recipient("П7", th), {"parameter": param, "drop_pct": float(drop.iloc[a]), "baseline": float(base),
                                          "current": float(cur.iloc[a])}))
    return events


def check_p11(data, th, tags, now: pd.Timestamp) -> list[RuleEvent]:
    """Пропуск обслуживания ЭХЗ: замеры реже нормы для типа установки (Запевалов и соавторы).

    Норма «N раз за период» проверяется как «с последнего замера прошло больше period/N».
    """
    norms = th["П11"]["min_measurements"]
    ecp = data[data["stream"] == "ecp"]
    events = []
    for sig in tags.stream_signals("ecp"):
        if sig.parameter != "current":
            continue
        kind = sig.object_attrs.get("install_type")
        norm = norms.get(kind)
        if not norm:
            continue
        allowed = pd.Timedelta(days=norm["period_days"] / norm["count"])
        last = ecp[ecp["object_id"] == sig.object_id]["ts"].max()
        if pd.isna(last) or now - last > allowed:
            age = "замеров нет" if pd.isna(last) else f"последний замер {last:%Y-%m-%d}, {(now - last).days} сут назад"
            events.append(RuleEvent(
                "П11", now, "ecp", sig.object_id,
                f"ЭХЗ {sig.object_id} ({kind}): {age}; норма — {norm['count']} раз за {norm['period_days']} сут",
                recipient("П11", th), {"install_type": kind, "last": None if pd.isna(last) else last.isoformat(),
                                       "allowed_days": allowed.total_seconds() / 86400}))
    return events


def check_p16(data, th):
    """Остаток C5+ выше допуска завода (продольный разбор). Порог задаёт пользователь."""
    lim = th["П16"].get("max_c5plus_g_m3")
    if lim is None:
        return []
    return _series_rule(data, th, "П16", "gas_composition", "C5plus", lambda v: v > lim,
                        lambda o, v: f"Остаток C5+ на выходе {o}: {v:.2f} г/м³ > допуска завода {lim} г/м³ — "
                                     f"тяжёлые компоненты вымерзают в теплообменниках сжижения")


def check_series_rules(data, th, tags, now) -> list[RuleEvent]:
    return check_p5(data, th) + check_p6(data, th) + check_p7(data, th) + check_p8(data, th) + \
        check_p9(data, th) + check_p10(data, th) + check_p11(data, th, tags, now) + check_p16(data, th)


# ---------- П15: модели М4 ----------

def check_p15(errors: list[dict[str, Any]], th: dict[str, Any], now: pd.Timestamp) -> list[RuleEvent]:
    """Ошибка модели относительно замеров выше порога — сигнал к перекалибровке
    (Григорьев и соавторы: калибровка по промысловым замерам). Без порога правило выключено."""
    lim = th["П15"].get("max_error_pct")
    if lim is None:
        return []
    return [RuleEvent(
        "П15", now, "dks" if e["model"] == "ДКС" else "cluster", e["output"].split(".", 1)[0],
        f"Модель «{e['model']}» расходится с замерами по {e['output']}: {e['error_pct']:.1f} % > {lim} % "
        f"за {e['days']} сут — перекалибровать модель или проверить датчик",
        recipient("П15", th), {"parameter": e["output"], "error_pct": e["error_pct"], "threshold": lim, "days": e["days"]})
        for e in errors if e["error_pct"] > lim]


# ---------- П12: рекомендации внешних ядер (М9) ----------

def check_p12(core_results, th: dict[str, Any], now: pd.Timestamp) -> list[RuleEvent]:
    """Ядро предложило изменение управляющих воздействий или перераспределение потоков
    (Войтенков): система предлагает вариант, решение за человеком."""
    events = []
    for r in core_results:
        for rec in r.recommendations:
            events.append(RuleEvent(
                "П12", now, None, rec.object_id, f"[{r.core}] {rec.title}. {rec.message}", recipient("П12", th),
                {"parameter": f"{r.core}:{rec.key}", "core": r.core, "title": rec.title,
                 "changes": [c.__dict__ for c in rec.changes], "expected_effect": rec.expected_effect}))
    return events


def log_events(journal, events: list[RuleEvent], shadow: bool = False) -> list[int]:
    """Записать срабатывания в журнал (М8). Рекомендации ядер — вид recommendation."""
    return [
        journal.log_event("recommendation" if e.code == "П12" else "rule", code=e.code, message=e.message, ts=e.ts.to_pydatetime(), stream=e.stream,
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
