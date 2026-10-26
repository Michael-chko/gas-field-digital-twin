"""М2. Контроль качества и очистка данных (Шевченко, Рычков; Печко и соавторы; Савченко и соавторы).

Конвейер Печко и соавторов: разбор и приведение к единому формату (М1) → фильтрация нефизичных
значений → выбросы → пропуски. Генерация признаков и нормализация — только при подготовке
выборок для М3 и М4, здесь их нет.

Исходные записи не меняются: результат — очищенное значение и флаги по каждой записи.

Флаги:
    gap                 пустое значение (пропуск)
    non_numeric         нечисловое значение (Шевченко, Рычков)
    state_derived       состояние оборудования выведено по нулевому расходу (Шевченко, Рычков)
    downtime            остановка или ремонт: значения не участвуют в статистике
    nonphysical         нуль давления или значение вне допустимого диапазона (Савченко и соавторы)
    zero_flow_pressure  аномальное давление при нулевом расходе (Шевченко, Рычков)
    outlier             вне трёх стандартных отклонений; помечается, не удаляется (Шевченко, Рычков)
    smooth_removed      отклоняется от сглаживающего полинома больше порога (Савченко и соавторы)
    restored            пропуск восстановлен интерполяцией (Савченко и соавторы; Печко и соавторы)
    gap_multi           в строке не хватает больше одного параметра — только помечено (Печко и соавторы)
    gap_unrestorable    одиночный пропуск без соседних значений с обеих сторон или во время простоя
    excluded            выброс, исключение которого подтвердил пользователь

«Строка» — один момент времени по одному объекту потока (широкая запись).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from field_twin import config, storage
from field_twin.gateway.tag_dictionary import TagDictionary

FLAGS = {
    "gap": "Пропуск",
    "non_numeric": "Нечисловое значение",
    "state_derived": "Состояние выведено по нулевому расходу",
    "downtime": "Простой (остановка или ремонт)",
    "nonphysical": "Нефизичное значение",
    "zero_flow_pressure": "Аномальное давление при нулевом расходе",
    "outlier": "Выброс (помечен, не удалён)",
    "smooth_removed": "Отклонение от сглаживающего полинома",
    "restored": "Пропуск восстановлен",
    "gap_multi": "Несколько пропусков в строке — только помечено",
    "gap_unrestorable": "Пропуск не восстановить",
    "excluded": "Исключено пользователем",
}
ROW_KEY = ["stream", "object_id", "ts"]
SIGNAL_KEY = ["stream", "object_id", "parameter"]


@dataclass
class StreamFreshness:
    stream: str
    last_ts: pd.Timestamp
    interval: timedelta | None
    interval_estimated: bool       # интервала нет в словаре — оценён по данным (решение прототипа)
    age: timedelta
    stale: bool


@dataclass
class QualityReport:
    rows_total: int = 0                       # записей (сигнал × момент)
    wide_rows: int = 0                        # строк (объект × момент)
    complete_rows_share: float = 0.0
    complete_rows_share_by_stream: dict[str, float] = field(default_factory=dict)
    gap_share_by_parameter: dict[str, float] = field(default_factory=dict)
    flag_counts: dict[str, int] = field(default_factory=dict)
    duplicates_dropped: int = 0
    criteria: list[dict[str, Any]] = field(default_factory=list)
    freshness: list[StreamFreshness] = field(default_factory=list)
    smoothing_enabled: bool = False
    now: pd.Timestamp | None = None

    @property
    def stale_streams(self) -> list[str]:
        return [f.stream for f in self.freshness if f.stale]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_total": self.rows_total,
            "wide_rows": self.wide_rows,
            "complete_rows_share": self.complete_rows_share,
            "complete_rows_share_by_stream": self.complete_rows_share_by_stream,
            "gap_share_by_parameter": self.gap_share_by_parameter,
            "flag_counts": self.flag_counts,
            "duplicates_dropped": self.duplicates_dropped,
            "criteria": self.criteria,
            "freshness": [
                {"stream": f.stream, "last_ts": f.last_ts.isoformat(), "age_s": f.age.total_seconds(),
                 "interval_s": f.interval.total_seconds() if f.interval else None,
                 "interval_estimated": f.interval_estimated, "stale": f.stale}
                for f in self.freshness
            ],
            "stale_streams": self.stale_streams,
            "smoothing_enabled": self.smoothing_enabled,
            "now": self.now.isoformat() if self.now is not None else None,
        }

    def to_text(self) -> str:
        lines = [
            "Отчёт о качестве данных",
            f"  записей: {self.rows_total}; строк (объект × момент): {self.wide_rows}",
            f"  доля строк без пропусков: {self.complete_rows_share:.1%}",
        ]
        for s, v in sorted(self.complete_rows_share_by_stream.items()):
            lines.append(f"    {s}: {v:.1%}")
        gaps = {k: v for k, v in self.gap_share_by_parameter.items() if v > 0}
        if gaps:
            lines.append("  доля пропусков по параметрам:")
            for k, v in sorted(gaps.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {k}: {v:.1%}")
        lines.append("  помечено по критериям:")
        for code, title in FLAGS.items():
            lines.append(f"    {title} ({code}): {self.flag_counts.get(code, 0)}")
        if self.duplicates_dropped:
            lines.append(f"  повторных записей отброшено: {self.duplicates_dropped}")
        if not self.smoothing_enabled:
            lines.append("  сглаживание полиномом выключено: степень, порог и окно не заданы (config/quality.yaml)")
        lines.append(f"  свежесть потоков на {self.now.isoformat() if self.now is not None else '—'}:")
        for f in self.freshness:
            iv = _fmt_td(f.interval) + (" (оценён по данным)" if f.interval_estimated else "") if f.interval else "не известен"
            mark = "УСТАРЕЛ" if f.stale else "ok"
            lines.append(f"    {f.stream}: {mark}; последнее значение {_fmt_td(f.age)} назад; интервал {iv}")
        return "\n".join(lines)


@dataclass
class QualityResult:
    data: pd.DataFrame
    report: QualityReport
    monthly: pd.DataFrame


def _fmt_td(td: timedelta | None) -> str:
    if td is None:
        return "—"
    s = int(td.total_seconds())
    if s % 86400 == 0 and s:
        return f"{s // 86400} сут"
    if s % 3600 == 0 and s:
        return f"{s // 3600} ч"
    return f"{s // 60} мин" if s >= 60 else f"{s} с"


def load_measurements(conn: sqlite3.Connection) -> pd.DataFrame:
    """Принятые записи без ошибок формата."""
    df = pd.read_sql_query(
        """SELECT id, ts_utc, stream, object_id, parameter, value, value_raw, unit, quality,
                  equipment_state, equipment_state_source, origin
           FROM measurements WHERE ingest_status != 'error' AND ts_utc IS NOT NULL""",
        conn,
    )
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    return df


def load_exclusions(conn: sqlite3.Connection) -> set[int]:
    return {r[0] for r in conn.execute("SELECT measurement_id FROM quality_exclusions")}


def confirm_exclusion(conn: sqlite3.Connection, measurement_ids: list[int], user_name: str, comment: str | None = None) -> None:
    """Пользователь подтверждает исключение выбросов (Шевченко, Рычков)."""
    if not user_name:
        raise ValueError("Не указан пользователь")
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """INSERT INTO quality_exclusions(measurement_id, user_name, comment, created_at) VALUES (?,?,?,?)
           ON CONFLICT(measurement_id) DO NOTHING""",
        [(int(i), user_name, comment, now) for i in measurement_ids],
    )
    conn.commit()


class QualityPipeline:
    def __init__(
        self,
        tags: TagDictionary | None = None,
        settings: dict[str, Any] | None = None,
        thresholds: dict[str, Any] | None = None,
        config_dir: str | Path | None = None,
    ):
        self.tags = tags or TagDictionary.load(config_dir)
        self.s = settings if settings is not None else config.quality(config_dir)
        th = thresholds if thresholds is not None else config.thresholds(config_dir)
        self.stale_factor = th["П13"]["stale_after_intervals"]
        self.sigma = float(self.s.get("outliers", {}).get("sigma", 3))

    # ---------- конвейер ----------

    def run(self, df: pd.DataFrame, now: datetime | pd.Timestamp | None = None,
            exclusions: set[int] | None = None) -> QualityResult:
        report = QualityReport()
        n0 = len(df)
        df = (df.sort_values([*SIGNAL_KEY, "ts", "id"])
                .drop_duplicates([*SIGNAL_KEY, "ts"], keep="last")
                .reset_index(drop=True))
        report.duplicates_dropped = n0 - len(df)
        df["kind"] = [self._kind(s, o, p) for s, o, p in zip(df.stream, df.object_id, df.parameter)]

        raw = df["value_raw"].fillna("").astype(str).str.strip()
        df["f_gap"] = raw == ""
        df["f_non_numeric"] = ~df["f_gap"] & df["value"].isna()
        numeric = df["value"].notna()

        self._equipment_state(df)
        self._nonphysical(df, numeric)
        df["f_excluded"] = df["id"].isin(exclusions or set())

        base = numeric & ~df["f_nonphysical"] & ~df["f_downtime"] & ~df["f_excluded"]
        stats = self._outliers(df, numeric, base)
        zfp = self.s.get("zero_flow_pressure") or {}
        zfp_excluded = set(zfp.get("exclude_streams") or [])
        zfp_sigma = float(zfp.get("sigma", self.sigma))
        df["f_zero_flow_pressure"] = (
            (df["kind"] == "pressure") & df["f_downtime"] & numeric & ~df["stream"].isin(zfp_excluded)
            & (df["f_nonphysical"] | (df["_z"].abs() > zfp_sigma))
        )
        report.smoothing_enabled = self._smoothing(df, base)
        good = base & ~df["f_outlier"] & ~df["f_smooth_removed"]
        self._gaps(df, good)

        df["value_clean"] = df["value"].where(numeric & ~df["f_nonphysical"] & ~df["f_excluded"])
        df.loc[df["f_restored"], "value_clean"] = df.loc[df["f_restored"], "_restored"]
        df["flags"] = self._flag_strings(df)

        self._fill_report(df, report, stats, now)
        monthly = self._monthly(df)
        out = df.drop(columns=[c for c in df.columns if c.startswith("_")])
        return QualityResult(out, report, monthly)

    def _kind(self, stream: str, obj: str, param: str) -> str:
        sig = self.tags.get(stream, obj, param)
        return sig.kind if sig else "other"

    def _equipment_state(self, df: pd.DataFrame) -> None:
        """Состояние из файла; если его нет — по нулевому расходу (Шевченко, Рычков)."""
        keys = [df[k] for k in ROW_KEY]
        given = df["equipment_state"].where(df["equipment_state"].fillna("").astype(str).str.strip() != "")
        given = given.groupby(keys).transform("first")
        is_flow = df["kind"] == "flow"
        any_zero = (is_flow & (df["value"] == 0)).groupby(keys).transform("any")
        any_pos = (is_flow & (df["value"] > 0)).groupby(keys).transform("any")
        derived = pd.Series(np.where(any_zero, "stop", np.where(any_pos, "run", None)), index=df.index, dtype=object)
        df["state"] = given.astype(object).where(given.notna(), derived)
        df["f_state_derived"] = given.isna() & df["state"].notna()
        df["state_source"] = np.where(df["f_state_derived"], "derived", np.where(given.notna(), "input", None))
        df["f_downtime"] = df["state"].isin(["stop", "repair"])

    def _nonphysical(self, df: pd.DataFrame, numeric: pd.Series) -> None:
        phys = self.s.get("physical_ranges") or {}
        ranges = self.s.get("ranges") or {}
        lo = df["kind"].map(lambda k: (phys.get(k) or {}).get("min", -np.inf)).astype(float)
        hi = df["kind"].map(lambda k: (phys.get(k) or {}).get("max", np.inf)).astype(float)
        excl = df["kind"].map(lambda k: bool((phys.get(k) or {}).get("min_exclusive", False)))
        if ranges:
            key = df["stream"] + "." + df["parameter"]
            r_lo = key.map(lambda k: (ranges.get(k) or {}).get("min", -np.inf)).astype(float)
            r_hi = key.map(lambda k: (ranges.get(k) or {}).get("max", np.inf)).astype(float)
            lo, hi = np.maximum(lo, r_lo), np.minimum(hi, r_hi)
        v = df["value"]
        below = (v < lo) | (excl & (v <= lo))
        df["f_nonphysical"] = numeric & (below | (v > hi))

    def _outliers(self, df: pd.DataFrame, numeric: pd.Series, base: pd.Series) -> pd.DataFrame:
        """Три критерия по каждому параметру (Шевченко, Рычков) и пометка выбросов."""
        vb = df["value"].where(base)
        g = vb.groupby([df[k] for k in SIGNAL_KEY])
        mean, std = g.transform("mean"), g.transform("std")
        z = (df["value"] - mean) / std.where(std > 0)
        df["_z"] = z
        df["f_outlier"] = base & (z.abs() > self.sigma)
        stats = (
            pd.DataFrame({**{k: df[k] for k in SIGNAL_KEY}, "v": vb, "within": (z.abs() <= self.sigma).astype(float).where(base & z.notna())})
            .groupby(SIGNAL_KEY, sort=True)
            .agg(n=("v", "count"), mean=("v", "mean"), median=("v", "median"), std=("v", "std"),
                 share_within_3sigma=("within", "mean"))
            .reset_index()
        )
        stats["mean_median_diff"] = stats["mean"] - stats["median"]
        stats["pearson"] = np.nan
        stats["pearson_with"] = None
        for pair in self.s.get("correlated_pairs") or []:
            r = pearson_pair(df.assign(value_clean=vb), pair["stream"], pair["a"], pair["b"])
            for me, other in ((pair["a"], pair["b"]), (pair["b"], pair["a"])):
                obj, par = me.split(".", 1)
                m = (stats.stream == pair["stream"]) & (stats.object_id == obj) & (stats.parameter == par)
                stats.loc[m, "pearson"] = r
                stats.loc[m, "pearson_with"] = other
        return stats

    def _smoothing(self, df: pd.DataFrame, usable: pd.Series) -> bool:
        """Полином по окнам (или по всему ряду); точки дальше порога удаляются, полином перестраивается
        (Савченко и соавторы, 2024, с. 149). Порог абсолютный (в единицах параметра) или в σ остатков."""
        sm = self.s.get("smoothing") or {}
        df["f_smooth_removed"] = False
        df["value_smooth"] = np.nan
        deg, thr, win = sm.get("degree"), sm.get("threshold"), sm.get("window")
        absolute = sm.get("threshold_mode", "sigma") == "absolute"
        if not sm.get("enabled", True) or deg is None or thr is None:
            return False
        key = df["stream"] + "." + df["parameter"]
        for series in sm.get("series") or []:
            for _, idx in df[(key == series) & usable].groupby("object_id").groups.items():
                sub = df.loc[idx].sort_values("ts")
                size = int(win) if win else len(sub)
                for start in range(0, len(sub), size):
                    w = sub.iloc[start : start + size]
                    if len(w) <= deg + 1:
                        continue
                    t = (w["ts"] - w["ts"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
                    y = w["value"].to_numpy(dtype=float)
                    keep = np.ones(len(w), dtype=bool)
                    for _ in range(20):
                        coef = np.polyfit(t[keep], y[keep], int(deg))
                        resid = y - np.polyval(coef, t)
                        sd = resid[keep].std()
                        if absolute:
                            new = keep & (np.abs(resid) <= thr)
                        else:
                            new = keep & (np.abs(resid) <= thr * sd) if sd > 0 else keep
                        if new.sum() == keep.sum() or new.sum() <= deg + 1:
                            break
                        keep = new
                    df.loc[w.index, "value_smooth"] = np.polyval(coef, t)
                    df.loc[w.index[~keep], "f_smooth_removed"] = True
        return True

    def _gaps(self, df: pd.DataFrame, good: pd.Series) -> None:
        """Одиночный пропуск в строке — интерполяция по соседним значениям; несколько — только пометка."""
        n_gap = df["f_gap"].groupby([df[k] for k in ROW_KEY]).transform("sum")
        single = df["f_gap"] & (n_gap == 1)
        df["f_gap_multi"] = df["f_gap"] & (n_gap > 1)
        df["f_restored"] = False
        df["f_gap_unrestorable"] = False
        df["_restored"] = np.nan
        targets = single & ~df["f_downtime"]
        df.loc[single & df["f_downtime"], "f_gap_unrestorable"] = True
        if not targets.any():
            return
        tsec = df["ts"].astype("int64").to_numpy() / 1e9
        groups = df.groupby(SIGNAL_KEY).groups
        tgt_by_sig = df[targets].groupby(SIGNAL_KEY).groups
        for sig, tidx in tgt_by_sig.items():
            gidx = groups[sig]
            known = [i for i in gidx if good[i]]
            if not known:
                df.loc[tidx, "f_gap_unrestorable"] = True
                continue
            kt = tsec[known]
            order = np.argsort(kt)
            kt, kv = kt[order], df.loc[known, "value"].to_numpy(dtype=float)[order]
            tt = tsec[list(tidx)]
            inside = (tt > kt[0]) & (tt < kt[-1])
            ok = np.array(list(tidx))[inside]
            df.loc[ok, "_restored"] = np.interp(tt[inside], kt, kv)
            df.loc[ok, "f_restored"] = True
            df.loc[np.array(list(tidx))[~inside], "f_gap_unrestorable"] = True

    def _flag_strings(self, df: pd.DataFrame) -> pd.Series:
        cols = [(c, df[f"f_{c}"].to_numpy()) for c in FLAGS]
        out = []
        for i in range(len(df)):
            out.append(";".join(c for c, arr in cols if arr[i]))
        return pd.Series(out, index=df.index)

    # ---------- отчёт ----------

    def _fill_report(self, df: pd.DataFrame, rep: QualityReport, stats: pd.DataFrame, now: Any) -> None:
        rep.rows_total = len(df)
        bad = (df["f_gap"] | df["f_non_numeric"]).groupby([df[k] for k in ROW_KEY]).any()
        rep.wide_rows = int(len(bad))
        rep.complete_rows_share = float((~bad).mean()) if len(bad) else 0.0
        rep.complete_rows_share_by_stream = {s: float((~b).mean()) for s, b in bad.groupby(level="stream")}
        gap = df.groupby(["stream", "parameter"])["f_gap"].mean()
        rep.gap_share_by_parameter = {f"{s}.{p}": float(v) for (s, p), v in gap.items()}
        rep.flag_counts = {c: int(df[f"f_{c}"].sum()) for c in FLAGS}
        rep.criteria = json.loads(stats.to_json(orient="records", force_ascii=False))
        if now is None:
            rep.now = df["ts"].max()   # проверка архива: «сейчас» — последнее значение в данных
        else:
            t = pd.Timestamp(now)
            if t.tzinfo is None:
                raise ValueError("Момент проверки свежести нужно указать с часовым поясом")
            rep.now = t.tz_convert("UTC")
        rep.freshness = self.freshness(df, rep.now)

    def freshness(self, df: pd.DataFrame, now: pd.Timestamp) -> list[StreamFreshness]:
        """Поток устарел, если не обновлялся дольше двух своих интервалов (оценка целостности данных)."""
        out = []
        for stream, sub in df.groupby("stream"):
            last = sub["ts"].max()
            interval, est = None, False
            if stream in self.tags.streams:
                interval = self.tags.stream_interval(stream)
            if interval is None:
                steps = sub["ts"].drop_duplicates().sort_values().diff().dropna()
                if len(steps):
                    interval, est = steps.median().to_pytimedelta(), True
            age = (now - last).to_pytimedelta()
            stale = interval is not None and age > self.stale_factor * interval
            out.append(StreamFreshness(stream, last, interval, est, age, bool(stale)))
        return out

    def _monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Усреднённые по месяцам ряды для обновления моделей (СТО Газпром 2-3.3-1203-2020)."""
        # Месяц считается по UTC, как и всё время в базе.
        use = df["value_clean"].notna() & ~df["f_downtime"] & ~df["f_outlier"]
        sub = df[use]
        month = sub["ts"].dt.strftime("%Y-%m")
        m = (sub.assign(month=month).groupby([*SIGNAL_KEY, "month"])["value_clean"]
                .agg(mean="mean", n="count").reset_index())
        return m


def pearson_pair(df: pd.DataFrame, stream: str, a: str, b: str, start=None, end=None) -> float:
    """Корреляция Пирсона двух связанных сигналов по общим моментам времени (Шевченко, Рычков)."""
    s = df[df["stream"] == stream]
    if start is not None:
        s = s[(s["ts"] >= start) & (s["ts"] < end)]
    sa = _series(s, a)
    sb = _series(s, b)
    j = pd.concat([sa, sb], axis=1, join="inner").dropna()
    if len(j) < 3 or j.iloc[:, 0].std() == 0 or j.iloc[:, 1].std() == 0:
        return float("nan")
    return float(j.iloc[:, 0].corr(j.iloc[:, 1]))


def _series(df: pd.DataFrame, ref: str) -> pd.Series:
    obj, par = ref.split(".", 1)
    s = df[(df["object_id"] == obj) & (df["parameter"] == par)]
    return s.set_index("ts")["value_clean"]


def save(conn: sqlite3.Connection, result: QualityResult) -> int:
    rep = result.report
    cur = conn.execute(
        "INSERT INTO quality_runs(started_at, now_utc, report_json) VALUES (?,?,?)",
        (datetime.now(timezone.utc).isoformat(), rep.now.isoformat() if rep.now is not None else None,
         json.dumps(rep.to_dict(), ensure_ascii=False)),
    )
    run_id = int(cur.lastrowid)
    d = result.data
    conn.executemany(
        """INSERT INTO clean_values(measurement_id, run_id, value_clean, value_smooth, equipment_state,
               equipment_state_source, flags) VALUES (?,?,?,?,?,?,?)""",
        [
            (int(i), run_id, None if pd.isna(v) else float(v), None if pd.isna(sm) else float(sm),
             st if isinstance(st, str) else None, src if isinstance(src, str) else None, fl)
            for i, v, sm, st, src, fl in zip(d["id"], d["value_clean"], d["value_smooth"], d["state"],
                                             d["state_source"], d["flags"])
        ],
    )
    conn.executemany(
        "INSERT INTO monthly_means(run_id, stream, object_id, parameter, month, mean, n) VALUES (?,?,?,?,?,?,?)",
        [(run_id, r.stream, r.object_id, r.parameter, r.month, float(r.mean), int(r.n))
         for r in result.monthly.itertuples()],
    )
    conn.commit()
    return run_id


def main(argv: list[str] | None = None) -> int:
    from field_twin import rules
    from field_twin.journal import Journal

    p = argparse.ArgumentParser(description="М2. Контроль качества и очистка; правила П13 и П14")
    p.add_argument("--db", default="data/twin.sqlite")
    p.add_argument("--now", default=None, help="момент проверки свежести, ISO 8601; по умолчанию — последнее значение в данных")
    p.add_argument("--no-journal", action="store_true", help="не записывать срабатывания правил в журнал")
    p.add_argument("--shadow", action="store_true", help="теневой режим: записи в журнале с пометкой")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    conn = storage.connect(args.db)
    pipe = QualityPipeline(config_dir=args.config)
    now = pd.Timestamp(args.now) if args.now else None
    if now is not None and now.tzinfo is None:
        raise SystemExit("--now нужно указать с часовым поясом")
    res = pipe.run(load_measurements(conn), now=now, exclusions=load_exclusions(conn))
    run_id = save(conn, res)
    print(res.report.to_text())
    th = config.thresholds(args.config)
    events = rules.check_p13(res.report, th) + rules.check_p14(res.data, pipe.s, th)
    print(f"\nПравила П13, П14: срабатываний {len(events)}")
    for e in events:
        print(f"  {e.code} {e.ts.isoformat()} {e.message}")
    if not args.no_journal and events:
        rules.log_events(Journal(conn, config_dir=args.config), events, shadow=args.shadow)
    if th["П14"].get("min_pearson") is None:
        print("  П14 выключено: порог min_pearson не задан (config/thresholds.yaml)")
    print(f"\nРезультат сохранён: quality_runs.id = {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
