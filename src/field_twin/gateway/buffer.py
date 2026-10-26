"""М12. Буфер гарантированной доставки между уровнями (промысел → общество → центр) с досылкой
после обрыва связи. Гарантированная передача между уровнями — Куник и соавторы (с. 38–39);
буфера и досылки у них нет — это Rogtec и решение прототипа.

Записи копятся в файле SQLite; повтор той же записи (сигнал + время + происхождение) не
принимается, поэтому повторная доставка MQTT или перечитанный архив не дают дублей. Запись
помечается отправленной только после успешной передачи получателю.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class DeliveryBuffer:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS buffer (
            id INTEGER PRIMARY KEY, stream TEXT, object_id TEXT, parameter TEXT, ts TEXT, origin TEXT,
            record_json TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0,
            UNIQUE (stream, object_id, parameter, ts, origin))""")

    def put(self, records: list[dict[str, Any]], origin: str = "asu") -> int:
        """Положить записи; вернуть число новых (повторы отброшены)."""
        before = self.conn.total_changes
        self.conn.executemany(
            "INSERT OR IGNORE INTO buffer(stream, object_id, parameter, ts, origin, record_json) VALUES (?,?,?,?,?,?)",
            [(r["stream"], r["object_id"], r["parameter"], _iso(r["ts"]), origin,
              json.dumps({**r, "ts": _iso(r["ts"])}, ensure_ascii=False)) for r in records])
        self.conn.commit()
        return self.conn.total_changes - before

    def pending(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM buffer WHERE sent = 0").fetchone()[0]

    def flush(self, sink: Callable[[list[dict[str, Any]]], None], batch: int = 5000) -> int:
        """Передать неотправленное по порядку; при сбое получателя записи остаются в буфере."""
        sent = 0
        while True:
            rows = self.conn.execute("SELECT id, record_json FROM buffer WHERE sent = 0 ORDER BY ts, id LIMIT ?",
                                     (batch,)).fetchall()
            if not rows:
                return sent
            recs = []
            for _, j in rows:
                r = json.loads(j)
                r["ts"] = datetime.fromisoformat(r["ts"])
                recs.append(r)
            sink(recs)                       # исключение — связь оборвана, ничего не помечаем
            self.conn.executemany("UPDATE buffer SET sent = 1 WHERE id = ?", [(i,) for i, _ in rows])
            self.conn.commit()
            sent += len(rows)


def _iso(ts) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
