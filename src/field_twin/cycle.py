"""Цикл обработки двойника: приём уже сделан (М1) → качество (М2) → правила и модели → журнал (М8).

Каждый следующий этап добавляет сюда свой шаг.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from field_twin import chudin_models, config, geotech, hydrates, notify, quality, rules, storage
from field_twin.config import ROOT
from field_twin.gateway.tag_dictionary import TagDictionary
from field_twin.journal import Journal


@dataclass
class CycleResult:
    quality: quality.QualityResult
    geotech: geotech.GeotechAssessment
    events: list[rules.RuleEvent] = field(default_factory=list)
    hydrate_predictions: pd.DataFrame | None = None
    notifications: list[notify.Notification] = field(default_factory=list)
    model_errors: list[dict[str, Any]] = field(default_factory=list)
    run_id: int | None = None
    journal_ids: list[int] = field(default_factory=list)


def run_cycle(conn: sqlite3.Connection, now: pd.Timestamp | None = None, shadow: bool = False,
              config_dir: str | Path | None = None, log: bool = True, save: bool = True,
              hydrate_model: hydrates.HydrateModel | str | Path | None = None,
              chudin_model: chudin_models.ChudinModels | str | Path | None = None) -> CycleResult:
    tags = TagDictionary.load(config_dir)
    th = config.thresholds(config_dir)
    pipe = quality.QualityPipeline(tags=tags, thresholds=th, config_dir=config_dir)
    q = pipe.run(quality.load_measurements(conn), now=now, exclusions=quality.load_exclusions(conn))
    now = q.report.now
    geo = geotech.assess(q.data, tags, th, now)
    events = (rules.check_p13(q.report, th) + rules.check_p14(q.data, pipe.s, th)
              + rules.check_geotech(geo, th, now) + rules.check_series_rules(q.data, th, tags, now))
    events.sort(key=lambda e: (e.ts, e.code))
    res = CycleResult(q, geo, events)
    # М3 и П1: если модель гидратов обучена.
    hs = hydrates.settings(config_dir)
    model = hydrate_model   # False — не использовать модель, None — взять обученную из data/models
    if model is None and (ROOT / hs["model_path"]).exists():
        model = ROOT / hs["model_path"]
    if model is not None and model is not False:
        if not isinstance(model, hydrates.HydrateModel):
            model = hydrates.HydrateModel.load(model)
        feats = hydrates.build_features(q.data, hs)
        if len(feats):
            res.hydrate_predictions = model.predict(feats)
            res.notifications = notify.evaluate_p1(res.hydrate_predictions, th, shadow=shadow)
    # М4 и П15: если модели Чудина обучены.
    cs = chudin_models.settings(config_dir)
    cm = chudin_model
    if cm is None and (ROOT / cs["model_path"]).exists():
        cm = ROOT / cs["model_path"]
    if cm is not None and cm is not False:
        if not isinstance(cm, chudin_models.ChudinModels):
            cm = chudin_models.ChudinModels.load(cm)
        res.model_errors = chudin_models.model_errors(
            cm, chudin_models.daily_table(q.data, "dks", cs["step"]), chudin_models.daily_table(q.data, "cluster", cs["step"]))
        p15 = rules.check_p15(res.model_errors, th, now)
        res.events = events = sorted(events + p15, key=lambda e: (e.ts, e.code))
    if save:
        res.run_id = quality.save(conn, q)
    if log:
        j = Journal(conn, config_dir=config_dir)
        if events:
            res.journal_ids = rules.log_events(j, events, shadow=shadow)
        if res.hydrate_predictions is not None:
            notify.log_model_outputs(j, res.hydrate_predictions, shadow=shadow)
            res.journal_ids += notify.log_notifications(j, res.notifications)
    return res


STATE_RULES = {"П2", "П3", "П4", "П11", "П13", "П15"}   # срабатывают на состояние «сейчас», а не на эпизод


def _event_key(e: rules.RuleEvent) -> tuple:
    if e.code in STATE_RULES:
        return (e.code, e.stream, e.object_id, e.payload.get("parameter"))
    return (e.code, e.stream, e.object_id, e.payload.get("parameter"), e.ts.isoformat())


class CycleRunner:
    """Повторяющийся цикл (проигрывание архива, работа в реальном времени).

    В журнал попадают только новые срабатывания: эпизодное правило — один раз на эпизод,
    правило состояния — когда состояние возникло (снова — после того, как оно прошло),
    П1 — один раз на серию, выходы модели — только новые моменты.
    """

    def __init__(self, conn: sqlite3.Connection, config_dir: str | Path | None = None, shadow: bool = False,
                 hydrate_model: hydrates.HydrateModel | str | Path | None = None,
                 chudin_model: chudin_models.ChudinModels | str | Path | None = None):
        self.chudin = chudin_model
        self.conn, self.config_dir, self.shadow = conn, config_dir, shadow
        self.model = hydrate_model   # False — без модели гидратов
        if self.model not in (None, False) and not isinstance(self.model, hydrates.HydrateModel):
            self.model = hydrates.HydrateModel.load(self.model)
        self.journal = Journal(conn, config_dir=config_dir)
        self.tags = TagDictionary.load(config_dir)
        self.screens = config.load_yaml("screens.yaml", config_dir)
        self.seen: set[tuple] = set()
        self.active: set[tuple] = set()
        self.last_model_ts: dict[str, pd.Timestamp] = {}
        self.cycles = 0

    def step(self, now: pd.Timestamp | None = None) -> CycleResult:
        res = run_cycle(self.conn, now=now, shadow=self.shadow, config_dir=self.config_dir, log=False,
                        save=False, hydrate_model=self.model if self.model is not None else False,
                        chudin_model=self.chudin if self.chudin is not None else False)
        new_events, active = [], set()
        for e in res.events:
            k = _event_key(e)
            if e.code in STATE_RULES:
                active.add(k)
                if k not in self.active:
                    new_events.append(e)
            elif k not in self.seen:
                self.seen.add(k)
                new_events.append(e)
        self.active = active
        res.journal_ids = rules.log_events(self.journal, new_events, shadow=self.shadow)
        if res.hydrate_predictions is not None:
            p = res.hydrate_predictions
            fresh = p[[t > self.last_model_ts.get(o, pd.Timestamp.min.tz_localize("UTC"))
                       for t, o in zip(p["ts"], p["object_id"])]]
            notify.log_model_outputs(self.journal, fresh, shadow=self.shadow)
            for o, g in p.groupby("object_id"):
                self.last_model_ts[o] = g["ts"].max()
            new_notes = []
            for n in res.notifications:
                k = ("П1", n.object_id, n.ts.isoformat())
                if k not in self.seen:
                    self.seen.add(k)
                    new_notes.append(n)
            res.journal_ids += notify.log_notifications(self.journal, new_notes)
            res.notifications = new_notes
        res.events = new_events
        write_screen_state(self.conn, res, self.tags, self.screens)
        self.cycles += 1
        return res


def _iso(t) -> str | None:
    return None if t is None or pd.isna(t) else pd.Timestamp(t).isoformat()


def build_screen_state(res: CycleResult, tags: TagDictionary, screens: dict[str, Any]) -> dict[str, Any]:
    """Снимок для экранов: значения, свежесть потоков, геотехника, даты последних замеров."""
    d = res.quality.data
    rep = res.quality.report
    latest = (d.sort_values("ts").groupby(["stream", "object_id", "parameter"]).tail(1)
              .set_index(["stream", "object_id", "parameter"]))
    groups = {}
    for role, sdef in screens.items():
        out = []
        for g in sdef.get("groups", []):
            rows = []
            objs = [g["object_id"]] if g.get("object_id") else sorted(
                {s.object_id for s in tags.stream_signals(g["stream"])})
            for obj in objs:
                for p in g["parameters"]:
                    sig = tags.get(g["stream"], obj, p)
                    if sig is None:
                        continue
                    key = (g["stream"], obj, p)
                    r = latest.loc[key] if key in latest.index else None
                    rows.append({
                        "object_id": obj, "parameter": p, "title": sig.title, "unit": sig.unit,
                        "value": None if r is None or pd.isna(r["value_clean"]) else float(r["value_clean"]),
                        "ts": None if r is None else _iso(r["ts"]),
                        "flags": "" if r is None else r["flags"],
                    })
            out.append({"title": g["title"], "stream": g["stream"], "rows": rows})
        groups[role] = out
    ecp_last = d[d.stream == "ecp"].groupby("object_id")["ts"].max()
    pipes_last = d[(d.stream == "pipelines") & (d.parameter == "safe_life_years")].groupby("object_id")["ts"].max()
    hyd = None
    if res.hydrate_predictions is not None and len(res.hydrate_predictions):
        p = res.hydrate_predictions
        win = p[p["ts"] > rep.now - pd.Timedelta(minutes=20)]
        hyd = [{"object_id": o, "proba": float(g.sort_values("ts")["proba"].iloc[-1]),
                "positives_20min": int(win[win.object_id == o]["label"].sum())}
               for o, g in p.groupby("object_id")]
    return {
        "now": _iso(rep.now),
        "freshness": [{"stream": f.stream, "title": tags.streams.get(f.stream, {}).get("title", f.stream),
                       "last_ts": _iso(f.last_ts), "stale": f.stale,
                       "interval_s": f.interval.total_seconds() if f.interval else None,
                       "interval_estimated": f.interval_estimated} for f in rep.freshness],
        "geotech": {
            "last_radar": _iso(res.geotech.last_radar),
            "objects": [{"object_id": o.object_id, "color": o.color, "title": geotech.TITLES[o.color],
                         "rate_mm_per_year": o.rate_mm_per_year, "unevenness": o.unevenness,
                         "last_radar": _iso(o.last_radar), "last_sensor": _iso(o.last_sensor)}
                        for o in res.geotech.objects],
            "parameters": [{"object_id": p.object_id, "parameter": p.parameter, "value": p.value, "limit": p.limit,
                            "ratio": p.ratio, "color": p.color} for p in res.geotech.parameters],
        },
        "last_measurements": {
            "ecp": {o: _iso(t) for o, t in ecp_last.items()},
            "pipelines_survey": {o: _iso(t) for o, t in pipes_last.items()},
            "geotech_radar": _iso(res.geotech.last_radar),
        },
        "hydrates": hyd,
        "groups": groups,
        "quality": {"complete_rows_share": rep.complete_rows_share, "stale_streams": rep.stale_streams},
    }


def write_screen_state(conn: sqlite3.Connection, res: CycleResult, tags: TagDictionary, screens: dict[str, Any]) -> None:
    import json
    st = build_screen_state(res, tags, screens)
    conn.execute("INSERT INTO screen_state(now_utc, state_json) VALUES (?, ?)", (st["now"], json.dumps(st, ensure_ascii=False)))
    conn.commit()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Цикл двойника: качество, модели, правила, журнал")
    p.add_argument("--db", default="data/twin.sqlite")
    p.add_argument("--now", default=None, help="момент проверки, ISO 8601 с часовым поясом")
    p.add_argument("--shadow", action="store_true", help="теневой режим")
    p.add_argument("--no-journal", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    now = pd.Timestamp(args.now) if args.now else None
    conn = storage.connect(args.db)
    res = run_cycle(conn, now=now, shadow=args.shadow, config_dir=args.config, log=not args.no_journal)
    write_screen_state(conn, res, TagDictionary.load(args.config), config.load_yaml("screens.yaml", args.config))
    print(res.quality.report.to_text())
    print()
    print(res.geotech.to_text())
    if res.hydrate_predictions is None:
        print("\nМ3: модель гидратов не обучена (field-twin-hydrates), П1 не проверялось")
    else:
        print(f"\nМ3: положительных выходов {int(res.hydrate_predictions['label'].sum())}; П1: предупреждений {len(res.notifications)}")
        for n in res.notifications:
            snd = "со звуком" if n.sound else "без звука (теневой режим)"
            print(f"  П1 {n.ts:%Y-%m-%d %H:%M} {n.object_id} {snd}; от первого срабатывания модели {n.delay}")
    print(f"\nСрабатывания правил: {len(res.events)}")
    for e in res.events:
        print(f"  {e.code:4} {e.ts:%Y-%m-%d %H:%M} → {e.recipient}: {e.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
