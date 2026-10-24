"""Архивная база имитатора АСУ ТП: всё, что сервер отдавал, с метками времени и качеством.

Нужна как отдельный от OPC UA архив (выгрузка за период для обучения моделей); чтение истории
по OPC UA (HistoryRead) обслуживает сам сервер.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


class AsuArchive:
    def __init__(self, path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS archive (
            tag TEXT NOT NULL, ts TEXT NOT NULL, value REAL, quality TEXT NOT NULL, PRIMARY KEY (tag, ts))""")

    def write(self, tag: str, ts: datetime, value: float | None, quality: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO archive VALUES (?,?,?,?)", (tag, ts.isoformat(), value, quality))

    def commit(self) -> None:
        self.conn.commit()

    def query(self, tag: str, start: datetime, end: datetime) -> list[tuple[str, float | None, str]]:
        return self.conn.execute("SELECT ts, value, quality FROM archive WHERE tag = ? AND ts >= ? AND ts <= ? ORDER BY ts",
                                 (tag, start.isoformat(), end.isoformat())).fetchall()

    def export_csv(self, path: str | Path) -> int:
        rows = self.conn.execute("SELECT tag, ts, value, quality FROM archive ORDER BY ts, tag").fetchall()
        with open(path, "w", encoding="utf-8") as f:
            f.write("tag,ts,value,quality\n")
            for r in rows:
                f.write(",".join("" if x is None else str(x) for x in r) + "\n")
        return len(rows)
