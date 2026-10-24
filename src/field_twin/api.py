"""Сервер и экраны оператора пульта и главного инженера (М7; Сафронова; Аксютин и соавторы).

FastAPI и HTML-страницы без внешних библиотек из интернета; звуковой сигнал — локальный файл.
Сервер только читает снимок состояния и журнал и записывает решения человека (М8).
В АСУ ТП двойник ничего не пишет.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from field_twin import storage
from field_twin.config import ROOT
from field_twin.journal import Journal

WEB = ROOT / "web"
ROLES = {"operator", "chief_engineer"}
SHOWN_KINDS = ("notification", "rule", "recommendation", "execution_check")


class DecisionIn(BaseModel):
    event_id: int
    decision: str
    reason_code: str
    user_name: str
    role: str
    comment: str | None = None


def _events(conn, role: str, limit: int = 200) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""SELECT e.*, d.decision, d.reason_code, d.user_name AS decided_by, d.ts_utc AS decided_at
            FROM journal_events e
            LEFT JOIN journal_decisions d ON d.id = (SELECT MAX(id) FROM journal_decisions WHERE event_id = e.id)
            WHERE e.source = 'twin' AND e.kind IN ({','.join('?' * len(SHOWN_KINDS))})
            ORDER BY e.ts_utc DESC, e.id DESC LIMIT ?""", (*SHOWN_KINDS, limit * 3)).fetchall()
    out = []
    for r in rows:
        payload = json.loads(r["payload_json"]) if r["payload_json"] else {}
        if payload.get("recipient", "chief_engineer") != role:
            continue
        out.append({"id": r["id"], "ts": r["ts_utc"], "kind": r["kind"], "code": r["code"], "source": r["source"],
                    "stream": r["stream"], "object_id": r["object_id"], "message": r["message"],
                    "shadow": bool(r["shadow"]), "sound": bool(payload.get("sound", False)),
                    "actions": payload.get("actions"), "delay_s": payload.get("delay_s"),
                    "decision": r["decision"], "reason_code": r["reason_code"], "decided_by": r["decided_by"],
                    "decided_at": r["decided_at"]})
        if len(out) >= limit:
            break
    return out


def _asu_alarms(conn, limit: int = 100) -> list[dict[str, Any]]:
    rows = conn.execute("""SELECT id, ts_utc, code, stream, object_id, message FROM journal_events
                           WHERE source = 'asu' ORDER BY ts_utc DESC, id DESC LIMIT ?""", (limit,)).fetchall()
    return [{"id": r["id"], "ts": r["ts_utc"], "code": r["code"], "stream": r["stream"], "object_id": r["object_id"],
             "message": r["message"], "source": "asu"} for r in rows]


def create_app(db_path: str | Path, config_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Цифровой двойник газового промысла")
    db_path = str(db_path)

    def conn():
        return storage.connect(db_path)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return ("<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>Цифровой двойник</title>"
                "<link rel='stylesheet' href='/web/style.css'></head><body><main class='index'>"
                "<h1>Цифровой двойник газового промысла</h1>"
                "<p><a href='/operator'>Экран оператора пульта</a></p>"
                "<p><a href='/chief'>Экран главного инженера</a></p></main></body></html>")

    @app.get("/operator")
    def operator():
        return FileResponse(WEB / "operator.html", media_type="text/html; charset=utf-8")

    @app.get("/chief")
    def chief():
        return FileResponse(WEB / "chief_engineer.html", media_type="text/html; charset=utf-8")

    @app.get("/web/{name}")
    def web(name: str):
        path = (WEB / name).resolve()
        if path.parent != WEB.resolve() or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/alarm.wav")
    def alarm():
        return FileResponse(WEB / "alarm.wav", media_type="audio/wav")

    @app.get("/api/state")
    def state(role: str = "operator"):
        if role not in ROLES:
            raise HTTPException(400, f"роль должна быть одной из {sorted(ROLES)}")
        c = conn()
        try:
            row = c.execute("SELECT state_json FROM screen_state ORDER BY id DESC LIMIT 1").fetchone()
            snap = json.loads(row["state_json"]) if row else None
            events = _events(c, role)
            sound = next((e["id"] for e in events if e["code"] == "П1" and e["sound"] and not e["decision"]), None)
            reasons = Journal(c, config_dir=config_dir).reasons()
            return JSONResponse({
                "role": role, "snapshot": snap, "events": events,
                "asu_alarms": _asu_alarms(c) if role == "operator" else [],
                "sound_event_id": sound, "reasons": reasons,
            })
        finally:
            c.close()

    @app.post("/api/decision")
    def decision(d: DecisionIn):
        c = conn()
        try:
            did = Journal(c, config_dir=config_dir).record_decision(
                d.event_id, d.decision, d.reason_code, d.user_name, d.role, d.comment)
            return {"id": did}
        except ValueError as e:
            raise HTTPException(400, str(e))
        finally:
            c.close()

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    p = argparse.ArgumentParser(description="Сервер экранов оператора и главного инженера")
    p.add_argument("--db", default="data/twin.sqlite")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    print(f"Экран оператора: http://{args.host}:{args.port}/operator")
    print(f"Экран главного инженера: http://{args.host}:{args.port}/chief")
    uvicorn.run(create_app(args.db, args.config), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
