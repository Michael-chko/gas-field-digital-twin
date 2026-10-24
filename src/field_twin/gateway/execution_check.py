"""М12. Проверка выполнения рекомендаций по данным АСУ (замыкание цикла; в статьях этого нет —
решение прототипа).

После подтверждения рекомендации двойник смотрит по данным АСУ, изменился ли режим: значение
параметра, который предлагалось изменить, приблизилось к предложенному (в пределах допуска или
хотя бы на половину пути). Результат — запись в журнале (вид execution_check).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pandas as pd

from field_twin import config
from field_twin.journal import Journal


def check_execution(conn: sqlite3.Connection, journal: Journal, now: pd.Timestamp,
                    settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    s = settings or config.load_yaml("gateway.yaml")["execution_check"]
    window = pd.Timedelta(hours=s.get("window_hours", 2))
    tol = s.get("tolerance_pct", 2) / 100
    done = {json.loads(r[0]).get("event_id") for r in conn.execute(
        "SELECT payload_json FROM journal_events WHERE kind = 'execution_check'")}
    rows = conn.execute(
        """SELECT e.id, e.payload_json, e.object_id, d.ts_utc FROM journal_events e
           JOIN journal_decisions d ON d.event_id = e.id
           WHERE e.kind = 'recommendation' AND d.decision = 'confirm'""").fetchall()
    out = []
    for eid, payload, obj, decided in rows:
        if eid in done:
            continue
        decided = pd.Timestamp(decided)
        changes = [c for c in (json.loads(payload or "{}").get("changes") or []) if c.get("proposed") is not None]
        if not changes:
            continue
        verdicts = []
        for c in changes:
            stream, o, p = c["parameter"].split(".", 2)
            vals = conn.execute(
                """SELECT ts_utc, value FROM measurements WHERE stream = ? AND object_id = ? AND parameter = ?
                   AND ts_utc > ? AND value IS NOT NULL ORDER BY ts_utc""",
                (stream, o, p, decided.isoformat())).fetchall()
            vals = [(pd.Timestamp(t), v) for t, v in vals if pd.Timestamp(t) <= now]
            if not vals:
                verdicts.append({"parameter": c["parameter"], "executed": False, "value": None})
                continue
            v = vals[-1][1]
            cur, prop = c.get("current"), c["proposed"]
            near = abs(v - prop) <= tol * abs(prop)
            halfway = cur is not None and abs(v - prop) <= 0.5 * abs(cur - prop)
            verdicts.append({"parameter": c["parameter"], "executed": bool(near or halfway), "value": v, "proposed": prop})
        executed = all(v["executed"] for v in verdicts)
        if not executed and now < decided + window:
            continue                       # ещё ждём: окно не истекло
        msg = ("Рекомендация выполнена: режим изменился по данным АСУ" if executed
               else "Рекомендация не выполнена: режим по данным АСУ не изменился")
        details = "; ".join(f"{v['parameter']} = {v['value'] if v['value'] is not None else '—'}"
                            f" (предложено {v.get('proposed', '—')})" for v in verdicts)
        journal.log_event("execution_check", code="П12", message=f"{msg}. {details}", ts=now.to_pydatetime(),
                          object_id=obj, payload={"event_id": eid, "executed": executed, "checks": verdicts,
                                                  "recipient": "chief_engineer"})
        out.append({"event_id": eid, "executed": executed, "checks": verdicts})
    return out
