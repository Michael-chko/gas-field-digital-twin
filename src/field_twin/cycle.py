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

from field_twin import config, geotech, quality, rules, storage
from field_twin.gateway.tag_dictionary import TagDictionary
from field_twin.journal import Journal


@dataclass
class CycleResult:
    quality: quality.QualityResult
    geotech: geotech.GeotechAssessment
    events: list[rules.RuleEvent] = field(default_factory=list)
    run_id: int | None = None
    journal_ids: list[int] = field(default_factory=list)


def run_cycle(conn: sqlite3.Connection, now: pd.Timestamp | None = None, shadow: bool = False,
              config_dir: str | Path | None = None, log: bool = True, save: bool = True) -> CycleResult:
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
    if save:
        res.run_id = quality.save(conn, q)
    if log and events:
        res.journal_ids = rules.log_events(Journal(conn, config_dir=config_dir), events, shadow=shadow)
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Цикл двойника: качество, модели, правила, журнал")
    p.add_argument("--db", default="data/twin.sqlite")
    p.add_argument("--now", default=None, help="момент проверки, ISO 8601 с часовым поясом")
    p.add_argument("--shadow", action="store_true", help="теневой режим")
    p.add_argument("--no-journal", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    now = pd.Timestamp(args.now) if args.now else None
    res = run_cycle(storage.connect(args.db), now=now, shadow=args.shadow, config_dir=args.config,
                    log=not args.no_journal)
    print(res.quality.report.to_text())
    print()
    print(res.geotech.to_text())
    print(f"\nСрабатывания правил: {len(res.events)}")
    for e in res.events:
        print(f"  {e.code:4} {e.ts:%Y-%m-%d %H:%M} → {e.recipient}: {e.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
