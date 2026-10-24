"""Хранилище прототипа: один файл SQLite без отдельного сервера (ТЗ, «Технические решения»)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
-- М1: принятые записи. Строки с ошибками формата тоже сохраняются (ingest_status = 'error').
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL REFERENCES ingest_batches(id),
    line_no INTEGER NOT NULL,
    ts_utc TEXT,                      -- ISO 8601 в целевом часовом поясе; NULL, если время не разобрано
    ts_raw TEXT,
    stream TEXT,
    object_id TEXT,
    parameter TEXT,
    value REAL,                       -- NULL: пусто или нечисловое значение (см. value_raw)
    value_raw TEXT,
    unit TEXT,
    quality TEXT,
    equipment_state TEXT,
    equipment_state_source TEXT,      -- input | absent | derived (М2)
    origin TEXT NOT NULL DEFAULT 'csv',  -- csv | asu | manual
    ingest_status TEXT NOT NULL,      -- ok | warning | error
    ingest_flags TEXT NOT NULL DEFAULT '',
    raw_line TEXT
);
CREATE INDEX IF NOT EXISTS ix_meas_signal ON measurements(stream, object_id, parameter, ts_utc);
CREATE INDEX IF NOT EXISTS ix_meas_status ON measurements(ingest_status);

CREATE TABLE IF NOT EXISTS ingest_batches (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    started_at TEXT NOT NULL,
    report_json TEXT
);

-- М8: журнал срабатываний, уведомлений и решений.
CREATE TABLE IF NOT EXISTS anomaly_reasons (
    code TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS journal_events (
    id INTEGER PRIMARY KEY,
    ts_utc TEXT NOT NULL,
    kind TEXT NOT NULL,               -- model | rule | notification | asu_alarm | recommendation | execution_check
    code TEXT,                        -- П1–П16, М3 и т. п.
    source TEXT NOT NULL,             -- twin | asu
    stream TEXT,
    object_id TEXT,
    message TEXT,
    payload_json TEXT,
    shadow INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_ts ON journal_events(ts_utc);

CREATE TABLE IF NOT EXISTS journal_decisions (
    id INTEGER PRIMARY KEY,
    event_id INTEGER NOT NULL REFERENCES journal_events(id),
    ts_utc TEXT NOT NULL,
    user_name TEXT NOT NULL,
    role TEXT NOT NULL,               -- operator | chief_engineer
    decision TEXT NOT NULL,           -- confirm | reject
    reason_code TEXT NOT NULL REFERENCES anomaly_reasons(code),
    comment TEXT
);

-- М2: результаты контроля качества. Исходные записи не меняются.
CREATE TABLE IF NOT EXISTS quality_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    now_utc TEXT,
    report_json TEXT
);

CREATE TABLE IF NOT EXISTS clean_values (
    measurement_id INTEGER NOT NULL REFERENCES measurements(id),
    run_id INTEGER NOT NULL REFERENCES quality_runs(id),
    value_clean REAL,                 -- NULL: пропуск, нечисловое, нефизичное или исключённое
    value_smooth REAL,                -- значение сглаживающего полинома (если включено)
    equipment_state TEXT,
    equipment_state_source TEXT,      -- input | derived
    flags TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, measurement_id)
);

CREATE TABLE IF NOT EXISTS monthly_means (
    run_id INTEGER NOT NULL REFERENCES quality_runs(id),
    stream TEXT NOT NULL,
    object_id TEXT NOT NULL,
    parameter TEXT NOT NULL,
    month TEXT NOT NULL,              -- ГГГГ-ММ (СТО Газпром 2-3.3-1203-2020)
    mean REAL,
    n INTEGER NOT NULL
);

-- Выбросы, исключение которых подтвердил пользователь (Шевченко, Рычков; Печко и соавторы).
CREATE TABLE IF NOT EXISTS quality_exclusions (
    measurement_id INTEGER PRIMARY KEY REFERENCES measurements(id),
    user_name TEXT NOT NULL,
    comment TEXT,
    created_at TEXT NOT NULL
);

-- Снимок состояния для экранов (М7): пишется циклом двойника, читается сервером.
CREATE TABLE IF NOT EXISTS screen_state (
    id INTEGER PRIMARY KEY,
    now_utc TEXT NOT NULL,
    state_json TEXT NOT NULL
);

-- Периоды, размеченные человеком (например, гидратообразование для М3).
CREATE TABLE IF NOT EXISTS period_labels (
    id INTEGER PRIMARY KEY,
    stream TEXT NOT NULL,
    object_id TEXT,
    start_utc TEXT NOT NULL,
    end_utc TEXT NOT NULL,
    reason_code TEXT NOT NULL REFERENCES anomaly_reasons(code),
    user_name TEXT NOT NULL,
    comment TEXT,
    created_at TEXT NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Открыть (и при необходимости создать) базу двойника."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")   # проигрывание пишет, сервер экранов читает
    conn.executescript(SCHEMA)
    return conn
