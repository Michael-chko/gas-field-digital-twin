"""Генератор синтетических данных (ТЗ, «Входные данные» и «Технические решения»).

Выдаёт те же потоки, параметры и интервалы, что и реальные источники, и внедряет дефекты:
простои, сбойные датчики, нечисловые значения, аномальные давления при нулевом расходе
(Шевченко, Рычков); пропуски (Печко и соавторы); шум в рядах давления и технологические
остановки скважин с ростом давления (Савченко и соавторы); эпизоды гидратов по признакам
из статьи Сафроновой; строки с ошибками формата для проверки приёма (М1).

Каждый внедрённый дефект записывается в truth.json — эталон для тестов М2, М3 и М10.
«Строка» в смысле пропусков — один момент времени по одному объекту (широкая запись).

Проигрывание архива (ускоренное воспроизведение CSV) добавляется на этапе 5.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from field_twin import config
from field_twin.config import parse_interval
from field_twin.gateway.tag_dictionary import Signal, TagDictionary

COLUMNS = ["timestamp", "stream", "object_id", "parameter", "value", "unit", "quality", "equipment_state"]
RADAR_SEASON = (6, 9)          # бесснежный сезон, месяцы (оценка целостности данных: зимой радара нет)
RADAR_PER_SEASON = 12          # 9–15 раз за бесснежный сезон (Затырко и соавторы)
ECP_SCHEDULE_DAYS = {"cathodic": 15, "drainage": 7.5, "protector": 182}  # из частоты в П11 (Запевалов и соавторы)


@dataclass
class StreamData:
    name: str
    has_state: bool
    signals: dict[str, list[Signal]]                 # объект -> сигналы
    times: dict[str, list[datetime]]                 # объект -> моменты
    values: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    tokens: dict[tuple[str, str], dict[int, str]] = field(default_factory=dict)
    quality: dict[tuple[str, str], dict[int, str]] = field(default_factory=dict)
    state: dict[str, np.ndarray] = field(default_factory=dict)
    busy: dict[str, np.ndarray] = field(default_factory=dict)

    def params(self, obj: str, kind: str | None = None) -> list[str]:
        return [s.parameter for s in self.signals[obj] if kind is None or s.kind == kind]


@dataclass
class GenerationResult:
    out_dir: Path
    files: dict[str, Path]
    truth: dict[str, Any]


class Generator:
    def __init__(self, config_dir: str | Path | None = None, overrides: dict[str, Any] | None = None):
        self.cfg = config.simulator(config_dir)
        for k, v in (overrides or {}).items():
            self.cfg[k] = v
        self.tags = TagDictionary.load(config_dir)
        self.rng = np.random.default_rng(self.cfg.get("seed", 42))
        self.start = datetime.fromisoformat(self.cfg["start"])
        if self.start.tzinfo is None:
            raise ValueError("simulator.start должен быть с часовым поясом")
        self.end = self.start + timedelta(days=float(self.cfg.get("days", 3)))
        self.defects: list[dict[str, Any]] = []
        self.skipped: dict[str, int] = {}   # дефекты, для которых не нашлось свободного окна
        self.streams: dict[str, StreamData] = {}

    # ---------- базовые ряды ----------

    def _times(self, stream: str, obj: str, attrs: dict[str, Any]) -> list[datetime]:
        if stream == "pipeline_segments":
            return [self.start]             # результаты обследования на начало периода
        # Радар и ручные замеры ЭХЗ редки: для тренда осадки и проверки П11 нужна история.
        hist = timedelta(days=float(self.cfg.get("history_days", {}).get(stream, 0)))
        if stream == "geotech_radar":
            return self._radar_times(self.start - hist)
        if stream == "ecp":
            step = timedelta(days=ECP_SCHEDULE_DAYS[attrs["install_type"]])
            times = _regular(self.start - hist, self.end, step)
            ov = (self.cfg.get("rule_events") or {}).get("ecp_overdue") or {}
            if ov.get("object_id") == obj:
                cutoff = self.end - timedelta(days=ov["last_days_before_end"])
                times = [t for t in times if t <= cutoff]
            return times
        step = self.tags.stream_interval(stream) or parse_interval(self.cfg["synthetic_intervals"][stream])
        return _regular(self.start, self.end, step)

    def _radar_times(self, since: datetime) -> list[datetime]:
        out = []
        for year in range(since.year, self.end.year + 1):
            s = since.replace(year=year, month=RADAR_SEASON[0], day=1, hour=0, minute=0, second=0)
            e = since.replace(year=year, month=RADAR_SEASON[1], day=30, hour=0, minute=0, second=0)
            step = (e - s) / RADAR_PER_SEASON
            t = s
            while t <= e:
                if since <= t < self.end:
                    out.append(t)
                t += step
        return out

    def build_baseline(self) -> None:
        no_state = set(self.cfg["defects"].get("no_equipment_state_streams", []))
        for stream in self.tags.streams:
            by_obj: dict[str, list[Signal]] = {}
            for sig in self.tags.stream_signals(stream):
                by_obj.setdefault(sig.object_id, []).append(sig)
            sd = StreamData(stream, stream not in no_state, by_obj, {})
            for obj, sigs in by_obj.items():
                times = self._times(stream, obj, sigs[0].object_attrs)
                sd.times[obj] = times
                n = len(times)
                hours = np.array([(t - self.start).total_seconds() / 3600 for t in times])
                sd.state[obj] = np.array(["run"] * n, dtype=object)
                sd.busy[obj] = np.zeros(n, dtype=bool)
                if stream == "pipeline_segments":
                    # Факторы КРН — разовое обследование участка (синтетика модуля scc).
                    from field_twin.scc import synthetic_factors
                    f = synthetic_factors(n, self.rng, sigs[0].object_attrs.get("pipeline_class", "field"))
                    for sig in sigs:
                        sd.values[(obj, sig.parameter)] = f[sig.parameter].to_numpy(dtype=float)
                    continue
                for sig in sigs:
                    base, noise = self.cfg["baselines"][f"{stream}.{sig.parameter}"]
                    base = base * (1 + self.rng.normal(0, 0.01))
                    phase = self.rng.uniform(0, 2 * math.pi)
                    daily = 1.5 * noise * np.sin(2 * math.pi * hours / 24 + phase)
                    sd.values[(obj, sig.parameter)] = base + daily + self.rng.normal(0, noise, n) if noise else np.full(n, base)
            self.streams[stream] = sd
        self._apply_physics()
        # Связанные датчики повторяют ведущий с поправкой: графики давления на входе и выходе
        # сепаратора должны быть схожи (Шевченко, Рычков).
        for key, c in (self.cfg.get("coupled") or {}).items():
            stream, obj, param = key.split(".", 2)
            lead_obj, lead_param = c["follows"].split(".", 1)
            sd = self.streams[stream]
            lead = sd.values[(lead_obj, lead_param)]
            sd.values[(obj, param)] = lead + c["offset"] + self.rng.normal(0, c["sd"], len(lead))

    def _apply_physics(self) -> None:
        """Синтетическая физика для моделей Чудина: мощность и обороты ДКС от давлений и расхода,
        дебиты скважин куста от буферного давления и давления на входе в ЗПА."""
        noise = self.cfg.get("physics_noise", 0.005)
        dks = self.streams["dks"]
        for obj in dks.signals:
            v = dks.values
            n = len(v[(obj, "Q")])
            v[(obj, "N_power")] = dks_power(v[(obj, "P_in")], v[(obj, "P_out")], v[(obj, "Q")]) * (1 + self.rng.normal(0, noise, n))
            v[(obj, "n_speed")] = dks_speed(v[(obj, "N_power")]) * (1 + self.rng.normal(0, noise, n))
        cl = self.streams["cluster"]
        wells = [o for o in cl.signals if o != "K-1"]
        pz = cl.values[("K-1", "P_zpa_inlet")]
        total = np.zeros(len(pz))
        for i, w in enumerate(wells):
            pb = np.maximum(cl.values[(w, "P_buffer")], pz + 0.1)
            cl.values[(w, "P_buffer")] = pb
            q = well_rate(pb, pz, cl.values[(w, "T_buffer")], WELL_COEF[i % len(WELL_COEF)])
            cl.values[(w, "Q_well")] = q * (1 + self.rng.normal(0, noise, len(q)))
            total += cl.values[(w, "Q_well")]
        cl.values[("K-1", "Q_plume_total")] = total * (1 + self.rng.normal(0, noise / 2, len(total)))

    # ---------- дефекты ----------

    def _window(self, sd: StreamData, obj: str, length: int, margin: int = 2) -> int | None:
        busy = sd.busy[obj]
        n = len(busy)
        if length + 2 * margin < n:
            for _ in range(500):
                s = int(self.rng.integers(margin, n - length - margin))
                if not busy[s - margin : s + length + margin].any():
                    busy[s : s + length] = True
                    return s
        self.skipped[sd.name] = self.skipped.get(sd.name, 0) + 1
        return None

    def _steps(self, sd: StreamData, obj: str, hours: float) -> int:
        times = sd.times[obj]
        step_h = (times[1] - times[0]).total_seconds() / 3600 if len(times) > 1 else 1
        return max(1, int(round(hours / step_h)))

    def _rec(self, defect: str, sd: StreamData, obj: str, **kw: Any) -> None:
        rec = {"defect": defect, "stream": sd.name, "object_id": obj}
        for k, v in kw.items():
            rec[k] = v.isoformat() if isinstance(v, datetime) else v
        self.defects.append(rec)

    def add_downtime(self) -> None:
        d = self.cfg["defects"]["downtime"]
        for stream in d["streams"]:
            sd = self.streams[stream]
            objs = [o for o in sd.signals if sd.params(o, "flow")]
            for _ in range(d["per_stream"]):
                obj = str(self.rng.choice(objs))
                length = self._steps(sd, obj, self.rng.uniform(*d["hours"]))
                s = self._window(sd, obj, length)
                if s is None:
                    continue
                e = s + length
                for p in sd.params(obj, "flow"):
                    sd.values[(obj, p)][s:e] = 0.0
                if sd.has_state:
                    sd.state[obj][s:e] = "stop"
                t = sd.times[obj]
                self._rec("downtime", sd, obj, start=t[s], end=t[e - 1], steps=length,
                          equipment_state_in_file=sd.has_state)
                pressures = sd.params(obj, "pressure")
                if not pressures:
                    continue
                k = min(d["zero_flow_pressure_points"], length)
                for i, idx in enumerate(sorted(self.rng.choice(np.arange(s, e), k, replace=False))):
                    p = str(self.rng.choice(pressures))
                    # «давление может записаться аномально большим или отрицательным» (Шевченко, Рычков)
                    val = sd.values[(obj, p)][idx] * 5 if i % 2 == 0 else -abs(sd.values[(obj, p)][idx])
                    sd.values[(obj, p)][idx] = val
                    self._rec("zero_flow_pressure", sd, obj, parameter=p, ts=t[idx], value=round(float(val), 4))

    def add_faulty_sensors(self) -> None:
        d = self.cfg["defects"]["faulty_sensor"]
        for stream in d["streams"]:
            sd = self.streams[stream]
            for i in range(d["per_stream"]):
                obj = str(self.rng.choice(list(sd.signals)))
                params = [p for p in sd.params(obj) if self._kind(sd, obj, p) in ("pressure", "temperature", "level")]
                if not params:
                    continue
                p = str(self.rng.choice(params))
                length = self._steps(sd, obj, self.rng.uniform(*d["hours"]))
                s = self._window(sd, obj, length)
                if s is None:
                    continue
                v = sd.values[(obj, p)]
                t = sd.times[obj]
                if i % 2 == 0:
                    v[s : s + length] = v[s]
                    mode = "stuck"
                else:
                    v[s : s + length] = v[s : s + length] * 3
                    mode = "spikes"
                self._rec("faulty_sensor", sd, obj, parameter=p, mode=mode, start=t[s], end=t[s + length - 1])

    def add_non_numeric(self) -> None:
        d = self.cfg["defects"]["non_numeric"]
        for stream in d["streams"]:
            sd = self.streams[stream]
            for i in range(d["per_stream"]):
                obj = str(self.rng.choice(list(sd.signals)))
                s = self._window(sd, obj, 1, margin=0)
                if s is None:
                    continue
                p = str(self.rng.choice(sd.params(obj)))
                token = d["tokens"][i % len(d["tokens"])]
                sd.tokens.setdefault((obj, p), {})[s] = token
                if i % 2 == 0:
                    sd.quality.setdefault((obj, p), {})[s] = "bad"
                self._rec("non_numeric", sd, obj, parameter=p, ts=sd.times[obj][s], token=token)

    # Нефизичное значение по роли параметра: нули давления (Савченко и соавторы),
    # уровень и открытие клапана вне 0–100 %, отрицательный расход, температура ниже абсолютного нуля.
    NONPHYSICAL = {"pressure": 0.0, "level": 130.0, "valve": -5.0, "flow": -10.0, "temperature": -300.0}

    def add_nonphysical(self) -> None:
        d = self.cfg["defects"]["nonphysical"]
        for stream in d["streams"]:
            sd = self.streams[stream]
            for _ in range(d["per_stream"]):
                obj = str(self.rng.choice(list(sd.signals)))
                params = [p for p in sd.params(obj) if self._kind(sd, obj, p) in self.NONPHYSICAL]
                s = self._window(sd, obj, 1, margin=1) if params else None
                if s is None:
                    continue
                p = str(self.rng.choice(params))
                val = self.NONPHYSICAL[self._kind(sd, obj, p)]
                sd.values[(obj, p)][s] = val
                self._rec("nonphysical", sd, obj, parameter=p, ts=sd.times[obj][s], value=val)

    def add_decorrelation(self) -> None:
        """Связанный датчик на время перестаёт повторять ведущий — для проверки П14."""
        d = self.cfg["defects"]["decorrelation"]
        sd = self.streams[d["stream"]]
        obj, p = d["object_id"], d["parameter"]
        length = self._steps(sd, obj, d["hours"])
        s = self._window(sd, obj, length, margin=5)
        if s is None:
            return
        v = sd.values[(obj, p)]
        v[s : s + length] = float(np.median(v)) + self.rng.normal(0, d["sd"], length)
        t = sd.times[obj]
        self._rec("decorrelation", sd, obj, parameter=p, start=t[s], end=t[s + length - 1])

    def add_pressure_noise(self) -> None:
        d = self.cfg["defects"]["pressure_noise"]
        for stream in d["streams"]:
            sd = self.streams[stream]
            for obj in sd.signals:
                free = ~sd.busy[obj]
                for p in d["parameters"]:
                    v = sd.values[(obj, p)]
                    v[free] += self.rng.normal(0, d["sd"], int(free.sum()))
                for _ in range(d["outliers_per_object"]):
                    s = self._window(sd, obj, 1, margin=0)
                    if s is None:
                        continue
                    p = str(self.rng.choice(d["parameters"]))
                    sd.values[(obj, p)][s] += float(self.rng.choice([-1, 1])) * 0.8
                    self._rec("pressure_outlier", sd, obj, parameter=p, ts=sd.times[obj][s])
            self.defects.append({"defect": "pressure_noise", "stream": stream, "parameters": d["parameters"], "sd": d["sd"]})

    def add_well_stops(self) -> None:
        d = self.cfg["defects"]["well_stops"]
        sd = self.streams["wells"]
        objs = list(self.rng.permutation(list(sd.signals)))
        plan = [True] * d["stops"] + [False] * d["short_stops"]
        for long_enough, obj in zip(plan, objs):
            obj = str(obj)
            hours = d["min_hours"] + self.rng.uniform(2, 8) if long_enough else d["min_hours"] / 2
            length = self._steps(sd, obj, hours)
            s = self._window(sd, obj, length, margin=3)
            if s is None:
                continue
            e = s + length
            for p in sd.params(obj, "flow"):
                sd.values[(obj, p)][s:e] = 0.0
            sd.state[obj][s:e] = "stop"
            # Монотонный рост давления после остановки (Савченко и соавторы); форма кривой — синтетика.
            k = np.arange(1, length + 1)
            rise = d["pressure_rise_mpa"] * np.log1p(k) / math.log1p(length)
            for p in ("P_wellhead", "P_bottomhole"):
                sd.values[(obj, p)][s:e] = sd.values[(obj, p)][s - 1] + rise
            t = sd.times[obj]
            self._rec("well_stop", sd, obj, start=t[s], end=t[e - 1], steps=length,
                      hours=round(length * (t[1] - t[0]).total_seconds() / 3600, 2),
                      long_enough=long_enough, min_hours=d["min_hours"])

    def add_hydrates(self) -> None:
        d = self.cfg["defects"]["hydrates"]
        sd = self.streams["absorber"]
        obj, plant = d["object"], "UKPG-1V"
        for k in range(d["episodes"]):
            length = self._steps(sd, obj, d["hours"])
            recover = self._steps(sd, obj, 1)
            if d.get("start_hours") is not None and k == 0:
                # Фиксированное начало эпизода (для проигрывания архива и тестов).
                s = self._steps(sd, obj, d["start_hours"])
                if s + length + recover > len(sd.times[obj]):
                    self.skipped[sd.name] = self.skipped.get(sd.name, 0) + 1
                    continue
                sd.busy[obj][s : s + length + recover] = True
            else:
                s = self._window(sd, obj, length + recover, margin=10)
            if s is None:
                continue
            sd.busy[plant][s : s + length + recover] = True
            ramp = np.concatenate([np.linspace(0, 1, length), np.linspace(1, 0, recover)])
            delayed = np.clip((np.arange(length + recover) - length / 2) / (length / 2), 0, 1) * ramp
            sl = slice(s, s + length + recover)
            # Признаки при ручном контроле (Сафронова): уровень в кубе снижается, клапан-регулятор
            # закрывается; расход газа растёт спустя несколько часов; снижаются расход через СИКГК,
            # уровень в Е-101 и температура газа на выходе. Величины — синтетика.
            sd.values[(obj, "L_cube")][sl] -= 10 * ramp
            sd.values[(obj, "KRU")][sl] -= 15 * ramp
            sd.values[(obj, "Q_gas")][sl] += 0.04 * sd.values[(obj, "Q_gas")][sl] * delayed
            sd.values[(obj, "T_gas_absorber")][sl] -= 1.5 * ramp
            sd.values[(plant, "Q_sikgk")][sl] -= 0.2 * sd.values[(plant, "Q_sikgk")][sl] * ramp
            sd.values[(plant, "L_E101.1")][sl] -= 8 * ramp
            sd.values[(plant, "L_E101.2")][sl] -= 8 * ramp
            t = sd.times[obj]
            self._rec("hydrate_episode", sd, obj, start=t[s], end=t[s + length - 1], recovered=t[s + length + recover - 1])

    def _limit(self, obj: str, param: str) -> float:
        sig = self.tags.get("geotech", obj, param)
        return float(sig.object_attrs["limits"][param])

    def _hold(self, sd: StreamData, obj: str, hours: float | None) -> tuple[int, int] | None:
        """Окно для события правила; hours=None — до конца данных."""
        n = len(sd.times[obj])
        if hours is None:
            s = int(n * 0.6)
            sd.busy[obj][s:] = True
            return s, n
        length = self._steps(sd, obj, hours)
        s = self._window(sd, obj, length, margin=2 if n > length + 6 else 0)
        return None if s is None else (s, s + length)

    def add_rule_events(self) -> None:
        """События для пороговых правил П2–П11 (величины — синтетика, пороги — из config/thresholds.yaml)."""
        ev = self.cfg.get("rule_events") or {}
        geo, radar = self.streams["geotech"], self.streams["geotech_radar"]
        for it in ev.get("geotech_levels", []):
            obj, p = it["object_id"], it["parameter"]
            w = self._hold(geo, obj, None)
            v = geo.values[(obj, p)]
            v[w[0] : w[1]] = self._limit(obj, p) * it["ratio"] + self.rng.normal(0, 0.002, w[1] - w[0])
            self._rec("rule_event", geo, obj, rule="П2", parameter=p, ratio=it["ratio"], start=geo.times[obj][w[0]])
        for it in ev.get("settlement_trend", []):
            obj, rate = it["object_id"], it["mm_per_year"]
            for sd in (geo, radar):
                years = np.array([(t - self.start).total_seconds() / (365.25 * 86400) for t in sd.times[obj]])
                sd.values[(obj, "settlement")] += rate * years
            self._rec("rule_event", geo, obj, rule="П3", parameter="settlement", mm_per_year=rate)
        for it in ev.get("unevenness", []):
            obj = it["object_id"]
            w = self._hold(geo, obj, None)
            geo.values[(obj, "settlement_unevenness")][w[0] : w[1]] = it["value"]
            self._rec("rule_event", geo, obj, rule="П4", parameter="settlement_unevenness", value=it["value"],
                      start=geo.times[obj][w[0]])
        simple = [("gas_velocity", "pipelines", "gas_velocity", "П5"), ("exhaust", "gtu", "T_exhaust", "П6"),
                  ("surge_margin", "gtu", "surge_margin", "П8"), ("c5plus", "gas_composition", "C5plus", "П16")]
        for key, stream, p, rule in simple:
            it = ev.get(key)
            if not it:
                continue
            sd = self.streams[stream]
            obj = it.get("object_id") or next(iter(sd.signals))
            w = self._hold(sd, obj, it.get("hours"))
            if w is None:
                continue
            sd.values[(obj, p)][w[0] : w[1]] = it["value"]
            t = sd.times[obj]
            self._rec("rule_event", sd, obj, rule=rule, parameter=p, value=it["value"], start=t[w[0]], end=t[w[1] - 1])
        if ev.get("safe_life"):
            it, sd = ev["safe_life"], self.streams["pipelines"]
            sd.values[(it["object_id"], "safe_life_years")][:] = it["value"]
            self._rec("rule_event", sd, it["object_id"], rule="П9", parameter="safe_life_years", value=it["value"])
        if ev.get("power_degradation"):
            it, sd = ev["power_degradation"], self.streams["gtu"]
            obj = next(iter(sd.signals))
            v = sd.values[(obj, "N_power")]
            v *= np.linspace(1.0, 1.0 - it["pct"] / 100.0, len(v))
            self._rec("rule_event", sd, obj, rule="П7", parameter="N_power", pct=it["pct"])
        if ev.get("ecp_zero"):
            it, sd = ev["ecp_zero"], self.streams["ecp"]
            obj, p = it["object_id"], it["parameter"]
            if not sd.times[obj]:
                return
            sd.values[(obj, p)][-1] = 0.0
            self._rec("rule_event", sd, obj, rule="П10", parameter=p, ts=sd.times[obj][-1])
        if ev.get("ecp_overdue"):
            it, sd = ev["ecp_overdue"], self.streams["ecp"]
            times = sd.times[it["object_id"]]
            self._rec("rule_event", sd, it["object_id"], rule="П11", last=times[-1] if times else None)

    def add_gaps(self) -> None:
        d = self.cfg["defects"]["gaps"]
        for stream in d["streams"]:
            sd = self.streams[stream]
            rows = complete = single = multi = 0
            cells = []
            for obj in sd.signals:
                params = sd.params(obj)
                others = [p for p in params if p != d["main_parameter"]]
                for i, t in enumerate(sd.times[obj]):
                    if sd.busy[obj][i]:
                        continue
                    rows += 1
                    if self.rng.random() < d["complete_rows_share"]:
                        complete += 1
                        continue
                    if self.rng.random() < d["single_gap_share"]:
                        miss = [d["main_parameter"] if self.rng.random() < 0.85 else str(self.rng.choice(others))]
                        single += 1
                    else:
                        extra = self.rng.choice(others, int(self.rng.integers(1, 3)), replace=False)
                        miss = [d["main_parameter"], *map(str, extra)]
                        multi += 1
                    for p in miss:
                        sd.values[(obj, p)][i] = np.nan
                    cells.append({"object_id": obj, "ts": t.isoformat(), "parameters": miss})
            self.defects.append({
                "defect": "gaps", "stream": stream, "rows": rows, "complete_rows": complete,
                "single_gap_rows": single, "multi_gap_rows": multi, "cells": cells,
            })

    def _kind(self, sd: StreamData, obj: str, param: str) -> str:
        return next(s.kind for s in sd.signals[obj] if s.parameter == param)

    # ---------- запись ----------

    def write(self, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        n_err = int(self.cfg["defects"].get("format_errors", 0))
        err_streams = ["absorber", "wells", "gathering", "dks", "methanol"]
        err_plan: dict[str, int] = {}
        for i in range(n_err):
            s = err_streams[i % len(err_streams)]
            err_plan[s] = err_plan.get(s, 0) + 1
        files = {}
        err_kind = 0
        for stream, sd in self.streams.items():
            path = out_dir / f"{stream}.csv"
            header = COLUMNS if sd.has_state else COLUMNS[:-1]
            rows = list(self._rows(sd))
            # Строки с ошибками формата вставляются в случайные места файла.
            bad_at = set(self.rng.choice(len(rows), min(err_plan.get(stream, 0), len(rows)), replace=False).tolist()) if rows else set()
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                line = 1
                for i, row in enumerate(rows):
                    row = row if sd.has_state else row[:-1]
                    if i in bad_at:
                        bad, flag = _corrupt(row, err_kind)
                        err_kind += 1
                        line += 1
                        w.writerow(bad)
                        self.defects.append({"defect": "format_error", "stream": stream, "file": path.name,
                                             "line_no": line, "expected_flag": flag})
                    line += 1
                    w.writerow(row)
            files[stream] = path
        return files

    def _rows(self, sd: StreamData):
        events = []
        for obj, times in sd.times.items():
            for i, t in enumerate(times):
                events.append((t, obj, i))
        events.sort(key=lambda e: (e[0], e[1]))
        for t, obj, i in events:
            ts = t.isoformat()
            for sig in sd.signals[obj]:
                key = (obj, sig.parameter)
                token = sd.tokens.get(key, {}).get(i)
                if token is not None:
                    value = token
                else:
                    v = sd.values[key][i]
                    value = "" if np.isnan(v) else f"{v:.6g}"
                quality = sd.quality.get(key, {}).get(i, "good")
                yield [ts, sd.name, obj, sig.parameter, value, sig.unit, quality, sd.state[obj][i]]

    def run(self, out_dir: str | Path) -> GenerationResult:
        out_dir = Path(out_dir)
        self.build_baseline()
        # Порядок важен: окна дефектов не пересекаются (маска busy).
        self.add_hydrates()
        self.add_rule_events()
        self.add_well_stops()
        self.add_downtime()
        self.add_decorrelation()
        self.add_faulty_sensors()
        self.add_nonphysical()
        self.add_non_numeric()
        self.add_pressure_noise()
        self.add_gaps()
        files = self.write(out_dir)
        truth = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": self.cfg.get("seed"),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "files": {k: v.name for k, v in files.items()},
            "streams_without_equipment_state": [s for s, sd in self.streams.items() if not sd.has_state],
            "defects": self.defects,
            "skipped_no_window": self.skipped,
        }
        with open(out_dir / "truth.json", "w", encoding="utf-8") as f:
            json.dump(truth, f, ensure_ascii=False, indent=1)
        return GenerationResult(out_dir, files, truth)


# ---------- синтетическая физика (не из источников; нужна, чтобы нейросетям было что учить) ----------

WELL_COEF = [200.0, 180.0, 220.0, 190.0]


def dks_power(p_in, p_out, q):
    """Мощность ДКС, МВт: политропное сжатие, показатель 0,25 — синтетика."""
    return 0.052 * q * ((np.asarray(p_out) / np.asarray(p_in)) ** 0.25 - 1) / 0.25


def dks_speed(power):
    """Скорость вращения, об/мин — синтетика."""
    return 2600 + 600 * np.sqrt(np.maximum(np.asarray(power), 0))


def well_rate(p_buffer, p_zpa, t_buffer, coef):
    """Дебит скважины, тыс. м³/сут, от перепада квадратов давлений — синтетика."""
    dp2 = np.maximum(np.asarray(p_buffer) ** 2 - np.asarray(p_zpa) ** 2, 0)
    return coef * np.sqrt(dp2) * np.sqrt(293.0 / (273.0 + np.asarray(t_buffer)))


def chudin_daily(days: int = 400, seed: int = 0, start: str = "2025-06-01T00:00:00+00:00",
                 noise: float = 0.005) -> "pd.DataFrame":
    """Суточные данные ДКС и куста для обучения моделей Чудина (столбцы как у очищенных данных М2).

    Давления медленно падают (истощение) и колеблются, чтобы выборка покрывала разные режимы.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    t = pd.date_range(pd.Timestamp(start), periods=days, freq="1D")
    k = np.arange(days)
    rows = []

    def add(stream, obj, param, values):
        for ts, v in zip(t, values):
            rows.append((stream, obj, param, ts, float(v)))

    p_in = 3.2 - 0.4 * k / days + rng.normal(0, 0.12, days)
    p_out = 7.5 + rng.normal(0, 0.25, days)
    q = 300 + 40 * np.sin(2 * np.pi * k / 90) + rng.normal(0, 20, days)
    n = dks_power(p_in, p_out, q) * (1 + rng.normal(0, noise, days))
    add("dks", "DKS-1", "P_in", p_in)
    add("dks", "DKS-1", "P_out", p_out)
    add("dks", "DKS-1", "Q", q)
    add("dks", "DKS-1", "N_power", n)
    add("dks", "DKS-1", "n_speed", dks_speed(n) * (1 + rng.normal(0, noise, days)))
    pz = 7.4 - 0.3 * k / days + rng.normal(0, 0.05, days)
    add("cluster", "K-1", "P_zpa_inlet", pz)
    total = np.zeros(days)
    for i, w in enumerate(["W-101", "W-102", "W-103", "W-104"]):
        pb = pz + 0.4 + 0.3 * np.sin(2 * np.pi * (k + 20 * i) / 120) * 0.5 + rng.normal(0, 0.06, days)
        pb = np.maximum(pb, pz + 0.1)
        tb = 15 + rng.normal(0, 1.0, days)
        qw = well_rate(pb, pz, tb, WELL_COEF[i]) * (1 + rng.normal(0, noise, days))
        total += qw
        add("cluster", w, "P_buffer", pb)
        add("cluster", w, "T_buffer", tb)
        add("cluster", w, "Q_well", qw)
    add("cluster", "K-1", "Q_plume_total", total * (1 + rng.normal(0, noise / 2, days)))
    df = pd.DataFrame(rows, columns=["stream", "object_id", "parameter", "ts", "value_clean"])
    df["f_downtime"] = False
    df["f_outlier"] = False
    return df


def _regular(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    out, t = [], start
    while t < end:
        out.append(t)
        t += step
    return out


def _corrupt(row: list[str], kind: int) -> tuple[list[str], str]:
    """Испортить копию строки одним из способов; вернуть строку и ожидаемый флаг М1."""
    bad = list(row)
    k = kind % 5
    if k == 0:
        bad[0] = "2026-13-45 99:00"
        return bad, "bad_timestamp"
    if k == 1:
        bad[0] = bad[0][:19]  # отрезать часовой пояс
        return bad, "timestamp_no_tz"
    if k == 2:
        bad[5] = ""
        return bad, "missing_unit"
    if k == 3:
        return bad[:-2], "wrong_field_count"
    bad[6] = "ok?"
    return bad, "bad_quality"


def generate(out_dir: str | Path, config_dir: str | Path | None = None, **overrides: Any) -> GenerationResult:
    return Generator(config_dir, overrides).run(out_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Генератор синтетических данных с дефектами")
    p.add_argument("--out", default="data/samples", help="папка для CSV и truth.json")
    p.add_argument("--days", type=float, default=None, help="длительность, суток")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--start", default=None, help="начало, ISO 8601 с часовым поясом")
    p.add_argument("--hydrate-episodes", type=int, default=None, help="число эпизодов гидратов (для обучения М3)")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    ov = {k: v for k, v in {"days": args.days, "seed": args.seed, "start": args.start}.items() if v is not None}
    if args.hydrate_episodes is not None:
        defects = config.simulator(args.config)["defects"]
        defects["hydrates"]["episodes"] = args.hydrate_episodes
        ov["defects"] = defects
    res = generate(args.out, args.config, **ov)
    counts: dict[str, int] = {}
    for d in res.truth["defects"]:
        counts[d["defect"]] = counts.get(d["defect"], 0) + 1
    print(f"Записано файлов: {len(res.files)} в {res.out_dir}")
    print("Внедрённые дефекты: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if res.truth["skipped_no_window"]:
        print("Не хватило места для части дефектов (увеличьте --days): "
              + ", ".join(f"{k}: {v}" for k, v in res.truth["skipped_no_window"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
