"""Проигрывание архива (ТЗ, «Технические решения»): ускоренное воспроизведение CSV, чтобы
уведомления появлялись на экранах как в реальном времени.

Записи из таблиц подаются в базу порциями по времени данных; после каждой порции выполняется
цикл двойника (качество, модели, правила, журнал, снимок для экранов). Скорость — во сколько раз
время данных идёт быстрее реального.
"""

from __future__ import annotations

import argparse
import csv
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from field_twin import storage
from field_twin.cycle import CycleResult, CycleRunner
from field_twin.ingest import Ingestor


@dataclass
class Row:
    ts: datetime | None
    file: Path
    line: str


def read_rows(files: Iterable[str | Path], delimiter: str = ",", encoding: str = "utf-8") -> tuple[dict[Path, str], list[Row]]:
    headers, rows = {}, []
    for f in files:
        f = Path(f)
        with open(f, encoding=encoding, newline="") as fh:
            lines = fh.read().splitlines()
        if not lines:
            continue
        headers[f] = lines[0]
        for line in lines[1:]:
            cell = next(csv.reader([line], delimiter=delimiter), [""])[0] if line else ""
            try:
                ts = datetime.fromisoformat(cell)
                ts = ts if ts.tzinfo else None
            except ValueError:
                ts = None
            rows.append(Row(ts.astimezone(timezone.utc) if ts else None, f, line))
    return headers, rows


class Playback:
    def __init__(self, db_path: str | Path, files: Iterable[str | Path], speed: float = 60.0,
                 cycle_minutes: float = 10.0, config_dir: str | Path | None = None, shadow: bool = False,
                 hydrate_model=None, start: str | None = None):
        self.conn = storage.connect(db_path)
        self.ingestor = Ingestor(self.conn, config_dir=config_dir)
        self.headers, rows = read_rows(files, self.ingestor.delimiter, self.ingestor.encoding)
        self.undated = [r for r in rows if r.ts is None]      # ошибки формата: в первую порцию
        self.rows = sorted((r for r in rows if r.ts is not None), key=lambda r: r.ts)
        if start is None and self.rows:
            # По умолчанию — с момента, когда данные есть во всех файлах; раньше — история
            # (редкие замеры радара и ЭХЗ за прошлые месяцы).
            firsts = {}
            for r in self.rows:
                firsts.setdefault(r.file, r.ts)
            start = max(firsts.values()).isoformat()
        if start:
            t0 = pd.Timestamp(start).tz_convert("UTC").to_pydatetime()
            history = [r for r in self.rows if r.ts < t0]
            self.rows = [r for r in self.rows if r.ts >= t0]
            self._ingest(history + self.undated, "playback:history")
            self.undated = []
        self.speed = float(speed)
        self.step = pd.Timedelta(minutes=cycle_minutes)
        self.runner = CycleRunner(self.conn, config_dir=config_dir, shadow=shadow, hydrate_model=hydrate_model)
        self.stop_flag = threading.Event()

    def _ingest(self, rows: list[Row], source: str) -> None:
        by_file: dict[Path, list[str]] = {}
        for r in rows:
            by_file.setdefault(r.file, []).append(r.line)
        for f, lines in by_file.items():
            self.ingestor.ingest_lines([self.headers[f], *lines], source=f"{source}:{f.name}")

    def chunks(self):
        if not self.rows:
            return
        t = pd.Timestamp(self.rows[0].ts).floor(self.step) + self.step
        i = 0
        first = True
        while i < len(self.rows):
            chunk = []
            while i < len(self.rows) and pd.Timestamp(self.rows[i].ts) < t:
                chunk.append(self.rows[i])
                i += 1
            if first:
                chunk = self.undated + chunk
                first = False
            yield t, chunk
            t += self.step

    def run(self, realtime: bool = True, max_cycles: int | None = None,
            on_cycle: Callable[[pd.Timestamp, CycleResult], None] | None = None) -> int:
        n = 0
        for t, chunk in self.chunks():
            if self.stop_flag.is_set() or (max_cycles is not None and n >= max_cycles):
                break
            started = time.monotonic()
            self._ingest(chunk, "playback")
            res = self.runner.step(now=t - pd.Timedelta(seconds=1))
            if on_cycle:
                on_cycle(t, res)
            n += 1
            if realtime:
                wait = self.step.total_seconds() / self.speed - (time.monotonic() - started)
                if wait > 0:
                    self.stop_flag.wait(wait)
        return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Проигрывание архива CSV с уведомлениями на экранах")
    p.add_argument("files", nargs="+", help="таблицы CSV (например, data/samples/*.csv)")
    p.add_argument("--db", default="data/playback.sqlite", help="база проигрывания (будет пополняться)")
    p.add_argument("--speed", type=float, default=60.0, help="во сколько раз быстрее реального времени")
    p.add_argument("--cycle-minutes", type=float, default=10.0, help="шаг цикла двойника по времени данных")
    p.add_argument("--start", default=None, help="с какого момента проигрывать (по умолчанию — когда данные есть во всех файлах); более ранние данные — история")
    p.add_argument("--shadow", action="store_true", help="теневой режим: без звука")
    p.add_argument("--model", default=None, help="модель гидратов (по умолчанию data/models/hydrates.cbm, если есть)")
    p.add_argument("--serve", action="store_true", help="одновременно запустить сервер экранов")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    from field_twin import hydrates
    from field_twin.config import ROOT
    model = args.model or (ROOT / hydrates.settings(args.config)["model_path"])
    model = model if Path(model).exists() else False
    pb = Playback(args.db, args.files, args.speed, args.cycle_minutes, args.config, args.shadow, model, args.start)
    if args.serve:
        import uvicorn
        from field_twin.api import create_app
        th = threading.Thread(target=uvicorn.run, args=(create_app(args.db, args.config),),
                              kwargs={"host": "127.0.0.1", "port": args.port, "log_level": "warning"}, daemon=True)
        th.start()
        print(f"Экраны: http://127.0.0.1:{args.port}/operator и /chief")

    def report(t, res):
        msgs = [f"{e.code} {e.object_id or e.stream}" for e in res.events] + [f"П1 {n.object_id}" for n in res.notifications]
        print(f"{t:%Y-%m-%d %H:%M} UTC" + (": " + ", ".join(msgs) if msgs else ""))

    try:
        n = pb.run(realtime=True, on_cycle=report)
    except KeyboardInterrupt:
        n = pb.runner.cycles
    print(f"Проиграно циклов: {n}")
    if args.serve:
        print("Архив закончился; сервер экранов работает, Ctrl+C — выход")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
