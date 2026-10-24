"""М8. Журнал и разметка.

- записывает каждое срабатывание модели и правила, каждое уведомление и решение человека с временем;
- при подтверждении или отклонении человек указывает причину аномалии из справочника —
  это разметка для обучения моделей с учителем (Аксютин и соавторы);
- в теневом режиме журнал — основа для сравнения сигналов с фактом (Сафронова).
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from field_twin import config, storage

EVENT_KINDS = {"model", "rule", "notification", "asu_alarm", "recommendation", "execution_check"}
SOURCES = {"twin", "asu"}  # уведомления двойника и сигнализация АСУ хранятся раздельно (ТЗ, М12)
ROLES = {"operator", "chief_engineer"}
DECISIONS = {"confirm", "reject"}

LABEL_COLUMNS = [
    "event_id", "event_ts_utc", "kind", "code", "source", "stream", "object_id", "message", "shadow",
    "decision", "reason_code", "reason_title", "decided_at_utc", "user_name", "role", "comment",
]
PERIOD_COLUMNS = ["id", "stream", "object_id", "start_utc", "end_utc", "reason_code", "reason_title", "user_name", "comment"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(ts: datetime | str | None) -> str:
    if ts is None:
        return _now()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            raise ValueError("Время без часового пояса")
        return ts.astimezone(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        raise ValueError(f"Время без часового пояса: {ts!r}")
    return parsed.astimezone(timezone.utc).isoformat()


class Journal:
    def __init__(self, conn: sqlite3.Connection, config_dir: str | Path | None = None):
        self.conn = conn
        self.sync_reasons(config.anomaly_reasons(config_dir))

    def sync_reasons(self, reasons: list[dict[str, Any]]) -> None:
        """Загрузить справочник причин. Коды не удаляются: на них ссылается разметка."""
        for r in reasons:
            self.conn.execute(
                """INSERT INTO anomaly_reasons(code, title, source, active) VALUES (?, ?, ?, 1)
                   ON CONFLICT(code) DO UPDATE SET title = excluded.title, source = excluded.source, active = 1""",
                (r["code"], r["title"], r.get("source")),
            )
        self.conn.commit()

    def reasons(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT code, title FROM anomaly_reasons WHERE active = 1 ORDER BY code")
        return {r["code"]: r["title"] for r in rows}

    def log_event(
        self,
        kind: str,
        code: str | None = None,
        message: str = "",
        ts: datetime | str | None = None,
        source: str = "twin",
        stream: str | None = None,
        object_id: str | None = None,
        payload: dict[str, Any] | None = None,
        shadow: bool = False,
    ) -> int:
        if kind not in EVENT_KINDS:
            raise ValueError(f"Неизвестный вид события: {kind!r}; допустимо: {sorted(EVENT_KINDS)}")
        if source not in SOURCES:
            raise ValueError(f"Неизвестный источник: {source!r}; допустимо: {sorted(SOURCES)}")
        cur = self.conn.execute(
            """INSERT INTO journal_events(ts_utc, kind, code, source, stream, object_id, message,
                   payload_json, shadow, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (_iso(ts), kind, code, source, stream, object_id, message,
             json.dumps(payload, ensure_ascii=False) if payload is not None else None, int(shadow), _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_many(self, events: list[dict[str, Any]]) -> None:
        """Пакетная запись событий (например, всех положительных выходов модели)."""
        rows = []
        for e in events:
            kind, source = e["kind"], e.get("source", "twin")
            if kind not in EVENT_KINDS or source not in SOURCES:
                raise ValueError(f"Недопустимое событие: {kind!r}/{source!r}")
            payload = e.get("payload")
            rows.append((_iso(e.get("ts")), kind, e.get("code"), source, e.get("stream"), e.get("object_id"),
                         e.get("message", ""), json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                         int(e.get("shadow", False)), _now()))
        self.conn.executemany(
            """INSERT INTO journal_events(ts_utc, kind, code, source, stream, object_id, message,
                   payload_json, shadow, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)""", rows)
        self.conn.commit()

    def record_decision(
        self,
        event_id: int,
        decision: str,
        reason_code: str,
        user_name: str,
        role: str,
        comment: str | None = None,
        ts: datetime | str | None = None,
    ) -> int:
        if decision not in DECISIONS:
            raise ValueError(f"Решение должно быть одним из {sorted(DECISIONS)}")
        if role not in ROLES:
            raise ValueError(f"Роль должна быть одной из {sorted(ROLES)}")
        if not user_name:
            raise ValueError("Не указан пользователь")
        self._check_reason(reason_code)
        if self.conn.execute("SELECT 1 FROM journal_events WHERE id = ?", (event_id,)).fetchone() is None:
            raise ValueError(f"Нет события с id={event_id}")
        cur = self.conn.execute(
            """INSERT INTO journal_decisions(event_id, ts_utc, user_name, role, decision, reason_code, comment)
               VALUES (?,?,?,?,?,?,?)""",
            (event_id, _iso(ts), user_name, role, decision, reason_code, comment),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def label_period(
        self,
        stream: str,
        start: datetime | str,
        end: datetime | str,
        reason_code: str,
        user_name: str,
        object_id: str | None = None,
        comment: str | None = None,
    ) -> int:
        """Разметить период (например, гидратообразование для обучения М3)."""
        self._check_reason(reason_code)
        start_iso, end_iso = _iso(start), _iso(end)
        if end_iso <= start_iso:
            raise ValueError("Конец периода должен быть позже начала")
        cur = self.conn.execute(
            """INSERT INTO period_labels(stream, object_id, start_utc, end_utc, reason_code, user_name, comment, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (stream, object_id, start_iso, end_iso, reason_code, user_name, comment, _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def _check_reason(self, reason_code: str) -> None:
        if not reason_code:
            raise ValueError("Причина аномалии обязательна (ТЗ, М8)")
        if reason_code not in self.reasons():
            raise ValueError(f"Причины {reason_code!r} нет в справочнике config/anomaly_reasons.yaml")

    def events(self, kind: str | None = None, code: str | None = None, shadow: bool | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM journal_events WHERE 1=1", []
        if kind:
            sql, args = sql + " AND kind = ?", args + [kind]
        if code:
            sql, args = sql + " AND code = ?", args + [code]
        if shadow is not None:
            sql, args = sql + " AND shadow = ?", args + [int(shadow)]
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY ts_utc, id", args)]

    def labels(self) -> list[dict[str, Any]]:
        """Разметка: события с решением человека и причиной аномалии."""
        rows = self.conn.execute(
            """SELECT e.id AS event_id, e.ts_utc AS event_ts_utc, e.kind, e.code, e.source, e.stream, e.object_id,
                      e.message, e.shadow, d.decision, d.reason_code, r.title AS reason_title,
                      d.ts_utc AS decided_at_utc, d.user_name, d.role, d.comment
               FROM journal_decisions d
               JOIN journal_events e ON e.id = d.event_id
               JOIN anomaly_reasons r ON r.code = d.reason_code
               ORDER BY e.ts_utc, d.id"""
        )
        return [dict(r) for r in rows]

    def period_labels(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT p.id, p.stream, p.object_id, p.start_utc, p.end_utc, p.reason_code, r.title AS reason_title,
                      p.user_name, p.comment
               FROM period_labels p JOIN anomaly_reasons r ON r.code = p.reason_code ORDER BY p.start_utc"""
        )
        return [dict(r) for r in rows]

    def export_labels(self, path: str | Path) -> int:
        return _write_csv(path, LABEL_COLUMNS, self.labels())

    def export_period_labels(self, path: str | Path) -> int:
        return _write_csv(path, PERIOD_COLUMNS, self.period_labels())


def _write_csv(path: str | Path, columns: list[str], rows: list[dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="М8. Выгрузка разметки из журнала")
    p.add_argument("--db", default="data/twin.sqlite")
    p.add_argument("--labels", default="data/labels.csv", help="разметка по решениям")
    p.add_argument("--periods", default="data/period_labels.csv", help="размеченные периоды")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    j = Journal(storage.connect(args.db), config_dir=args.config)
    print(f"Решений с причинами: {j.export_labels(args.labels)} -> {args.labels}")
    print(f"Размеченных периодов: {j.export_period_labels(args.periods)} -> {args.periods}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
