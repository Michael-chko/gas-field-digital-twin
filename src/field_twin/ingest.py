"""М1. Приём данных.

Читает таблицы CSV длинного формата (ТЗ, «Технические решения»):
timestamp, stream, object_id, parameter, value, unit, quality, equipment_state.

- разделитель и кодировка — из config/quality.yaml (Запевалов и соавторы);
- время приводится к одному часовому поясу; время без часового пояса — ошибка;
- проверяются обязательные поля, включая единицу измерения и состояние оборудования;
- строки с ошибками формата не отбрасываются: сохраняются с пометкой и попадают в отчёт;
- сигнал без привязки в словаре параметров попадает в отчёт (М12).

Нечисловые значения и пропуски здесь не исправляются: это задача М2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from field_twin import config, storage
from field_twin.gateway.tag_dictionary import TagDictionary

REQUIRED_COLUMNS = ["timestamp", "stream", "object_id", "parameter", "value", "unit", "quality"]
OPTIONAL_COLUMNS = ["equipment_state"]

# Ошибки формата: строка сохраняется со статусом error.
ERRORS = {
    "wrong_field_count": "Число полей не совпадает с заголовком",
    "bad_timestamp": "Время не разобрано",
    "timestamp_no_tz": "Время без часового пояса",
    "missing_stream": "Не указан поток",
    "missing_object": "Не указан объект",
    "missing_parameter": "Не указан параметр",
    "missing_unit": "Не указана единица измерения",
    "missing_quality": "Не указан признак качества",
    "bad_quality": "Недопустимый признак качества",
    "bad_equipment_state": "Недопустимое состояние оборудования",
}
# Предупреждения: строка принимается со статусом warning.
WARNINGS = {
    "missing_equipment_state": "Нет состояния оборудования — выведет М2",
    "unmapped_signal": "Сигнал не привязан в словаре параметров",
    "unit_mismatch": "Единица не совпадает со словарём",
}


@dataclass
class RowIssue:
    line_no: int
    flags: list[str]
    raw: str


@dataclass
class IngestReport:
    source: str
    rows_total: int = 0
    rows_ok: int = 0
    rows_warning: int = 0
    rows_error: int = 0
    flag_counts: Counter = field(default_factory=Counter)
    error_rows: list[RowIssue] = field(default_factory=list)
    unmapped_signals: set[tuple[str, str, str]] = field(default_factory=set)
    rows_by_stream: Counter = field(default_factory=Counter)
    equipment_state_column: bool = True
    missing_columns: list[str] = field(default_factory=list)
    unbound_source_signals: list[str] = field(default_factory=list)   # есть в АСУ, нет в словаре (М12)
    signals_without_setpoints: list[str] = field(default_factory=list)  # уставки не заполнены (М12)
    batch_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "rows_total": self.rows_total,
            "rows_ok": self.rows_ok,
            "rows_warning": self.rows_warning,
            "rows_error": self.rows_error,
            "flag_counts": dict(self.flag_counts),
            "error_rows": [{"line_no": r.line_no, "flags": r.flags, "raw": r.raw} for r in self.error_rows],
            "unmapped_signals": [".".join(s) for s in sorted(self.unmapped_signals)],
            "rows_by_stream": dict(self.rows_by_stream),
            "equipment_state_column": self.equipment_state_column,
            "missing_columns": self.missing_columns,
            "unbound_source_signals": self.unbound_source_signals,
            "signals_without_setpoints": self.signals_without_setpoints,
        }

    def to_text(self) -> str:
        lines = [
            f"Отчёт о приёме: {self.source}",
            f"  строк всего: {self.rows_total}; без замечаний: {self.rows_ok}; "
            f"с предупреждениями: {self.rows_warning}; с ошибками формата: {self.rows_error}",
        ]
        if self.missing_columns:
            lines.append(f"  нет обязательных столбцов: {', '.join(self.missing_columns)} — файл не принят")
        if not self.equipment_state_column:
            lines.append("  в файле нет столбца equipment_state: состояние оборудования выведет М2")
        if self.rows_by_stream:
            lines.append("  по потокам: " + ", ".join(f"{k or '—'}: {v}" for k, v in sorted(self.rows_by_stream.items())))
        for code, n in sorted(self.flag_counts.items()):
            title = ERRORS.get(code) or WARNINGS.get(code) or code
            kind = "ошибка" if code in ERRORS else "предупреждение"
            lines.append(f"  [{kind}] {title} ({code}): {n}")
        if self.error_rows:
            lines.append("  строки с ошибками формата:")
            for r in self.error_rows[:50]:
                lines.append(f"    строка {r.line_no}: {', '.join(r.flags)} | {r.raw}")
            if len(self.error_rows) > 50:
                lines.append(f"    … ещё {len(self.error_rows) - 50}")
        if self.unmapped_signals:
            lines.append("  сигналы без привязки в словаре:")
            for s in sorted(self.unmapped_signals)[:50]:
                lines.append(f"    {'.'.join(s)}")
        if self.unbound_source_signals:
            lines.append("  сигналы источника АСУ без привязки к объекту двойника:")
            for t in self.unbound_source_signals[:50]:
                lines.append(f"    {t}")
        if self.signals_without_setpoints:
            lines.append(f"  уставки не заполнены (незаполненные сигналы словаря): {len(self.signals_without_setpoints)}")
            for t in self.signals_without_setpoints[:20]:
                lines.append(f"    {t}")
            if len(self.signals_without_setpoints) > 20:
                lines.append(f"    … ещё {len(self.signals_without_setpoints) - 20}")
        return "\n".join(lines)


def _target_tz(name: str) -> tzinfo:
    return timezone.utc if name.upper() == "UTC" else ZoneInfo(name)


def parse_timestamp(text: str, target: tzinfo) -> tuple[str | None, str | None]:
    """Вернуть (время ISO 8601 в целевом поясе, код ошибки)."""
    text = (text or "").strip()
    if not text:
        return None, "bad_timestamp"
    try:
        ts = datetime.fromisoformat(text)
    except ValueError:
        return None, "bad_timestamp"
    if ts.tzinfo is None:
        return None, "timestamp_no_tz"
    return ts.astimezone(target).isoformat(), None


def _to_float(text: str) -> float | None:
    try:
        v = float(text.replace(",", ".")) if text.strip() else None
    except ValueError:
        return None
    if v is not None and v != v:  # NaN
        return None
    return v


class Ingestor:
    def __init__(
        self,
        conn: sqlite3.Connection,
        tags: TagDictionary | None = None,
        settings: dict[str, Any] | None = None,
        config_dir: str | Path | None = None,
    ):
        self.conn = conn
        self.tags = tags or TagDictionary.load(config_dir)
        s = settings or config.quality(config_dir).get("ingest", {})
        self.delimiter = s.get("delimiter", ",")
        self.encoding = s.get("encoding", "utf-8")
        self.target_tz = _target_tz(s.get("target_timezone", "UTC"))
        self.quality_values = set(s.get("quality_values", ["good", "uncertain", "bad"]))
        self.equipment_states = set(s.get("equipment_states", ["run", "stop", "repair"]))

    def ingest_file(self, path: str | Path, origin: str = "csv") -> IngestReport:
        path = Path(path)
        with open(path, encoding=self.encoding, newline="") as f:
            return self.ingest_lines(f, source=str(path), origin=origin)

    def ingest_lines(self, lines: Iterable[str], source: str, origin: str = "csv") -> IngestReport:
        report = IngestReport(source=source)
        cur = self.conn.execute(
            "INSERT INTO ingest_batches(source, started_at) VALUES (?, ?)",
            (source, datetime.now(timezone.utc).isoformat()),
        )
        report.batch_id = cur.lastrowid
        lines = list(lines)
        reader = csv.reader(lines, delimiter=self.delimiter)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            header = []
        report.missing_columns = [c for c in REQUIRED_COLUMNS if c not in header]
        report.equipment_state_column = "equipment_state" in header
        if report.missing_columns:
            self._finish(report)
            return report

        idx = {name: header.index(name) for name in header}
        records = []
        for row_no, row in enumerate(reader, start=2):
            raw = self.delimiter.join(row)
            if not any(cell.strip() for cell in row):
                continue
            report.rows_total += 1
            rec, flags = self._parse_row(row, idx, len(header))
            errors = [f for f in flags if f in ERRORS]
            status = "error" if errors else ("warning" if flags else "ok")
            report.flag_counts.update(flags)
            report.rows_by_stream[rec["stream"]] += 1
            if status == "error":
                report.rows_error += 1
                report.error_rows.append(RowIssue(row_no, flags, raw))
            elif status == "warning":
                report.rows_warning += 1
            else:
                report.rows_ok += 1
            if "unmapped_signal" in flags:
                report.unmapped_signals.add((rec["stream"], rec["object_id"], rec["parameter"]))
            records.append(
                (
                    report.batch_id, row_no, rec["ts_utc"], rec["ts_raw"], rec["stream"], rec["object_id"],
                    rec["parameter"], rec["value"], rec["value_raw"], rec["unit"], rec["quality"],
                    rec["equipment_state"], rec["equipment_state_source"], origin, status, ";".join(flags), raw,
                )
            )
        self.conn.executemany(
            """INSERT INTO measurements(batch_id, line_no, ts_utc, ts_raw, stream, object_id, parameter,
                   value, value_raw, unit, quality, equipment_state, equipment_state_source, origin,
                   ingest_status, ingest_flags, raw_line)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            records,
        )
        self._finish(report)
        return report

    def ingest_records(self, records: list[dict[str, Any]], source: str, origin: str = "asu",
                       unbound: list[str] | None = None, without_setpoints: list[str] | None = None) -> IngestReport:
        """Приём записей от шлюза АСУ (М12) или ручного ввода тем же путём, что и CSV."""
        import io
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=self.delimiter, lineterminator="\n")
        # Столбец состояния — только если источник его передаёт (в архивах АСУ его нет; выведет М2).
        with_state = any(r.get("equipment_state") for r in records)
        w.writerow(REQUIRED_COLUMNS + (OPTIONAL_COLUMNS if with_state else []))
        for r in records:
            ts = r["ts"].isoformat() if hasattr(r["ts"], "isoformat") else str(r["ts"])
            val = r.get("value_raw")
            if val is None:
                val = "" if r.get("value") is None else repr(float(r["value"]))
            unit = r.get("unit")
            if unit is None:
                sig = self.tags.get(r["stream"], r["object_id"], r["parameter"])
                unit = sig.unit if sig else ""
            row = [ts, r["stream"], r["object_id"], r["parameter"], val, unit, r.get("quality", "good")]
            w.writerow(row + ([r.get("equipment_state") or ""] if with_state else []))
        report = self.ingest_lines(buf.getvalue().splitlines(), source=source, origin=origin)
        report.unbound_source_signals = list(unbound or [])
        report.signals_without_setpoints = list(without_setpoints or [])
        self._finish(report)
        return report

    def _parse_row(self, row: list[str], idx: dict[str, int], n_fields: int) -> tuple[dict[str, Any], list[str]]:
        flags: list[str] = []
        if len(row) != n_fields:
            flags.append("wrong_field_count")

        def cell(name: str) -> str:
            i = idx.get(name)
            return row[i].strip() if i is not None and i < len(row) else ""

        ts_raw = cell("timestamp")
        ts_utc, ts_err = parse_timestamp(ts_raw, self.target_tz)
        if ts_err:
            flags.append(ts_err)
        stream, object_id, parameter = cell("stream"), cell("object_id"), cell("parameter")
        unit, quality, state = cell("unit"), cell("quality"), cell("equipment_state")
        value_raw = cell("value")
        for name, val, code in (
            ("stream", stream, "missing_stream"),
            ("object_id", object_id, "missing_object"),
            ("parameter", parameter, "missing_parameter"),
            ("unit", unit, "missing_unit"),
            ("quality", quality, "missing_quality"),
        ):
            if not val:
                flags.append(code)
        if quality and quality not in self.quality_values:
            flags.append("bad_quality")
        if state:
            state_source = "input"
            if state not in self.equipment_states:
                flags.append("bad_equipment_state")
        else:
            state_source = "absent"
            # Нет столбца — отмечено один раз в отчёте; пустое значение в столбце — предупреждение.
            if "equipment_state" in idx:
                flags.append("missing_equipment_state")
        if stream and object_id and parameter:
            sig = self.tags.get(stream, object_id, parameter)
            if sig is None:
                flags.append("unmapped_signal")
            elif unit and unit != sig.unit:
                flags.append("unit_mismatch")
        return (
            {
                "ts_utc": ts_utc,
                "ts_raw": ts_raw,
                "stream": stream,
                "object_id": object_id,
                "parameter": parameter,
                "value": _to_float(value_raw),
                "value_raw": value_raw,
                "unit": unit,
                "quality": quality,
                "equipment_state": state or None,
                "equipment_state_source": state_source,
            },
            flags,
        )

    def _finish(self, report: IngestReport) -> None:
        self.conn.execute(
            "UPDATE ingest_batches SET report_json = ? WHERE id = ?",
            (json.dumps(report.to_dict(), ensure_ascii=False), report.batch_id),
        )
        self.conn.commit()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="М1. Приём таблиц CSV в базу двойника")
    p.add_argument("files", nargs="+", help="файлы CSV длинного формата")
    p.add_argument("--db", default="data/twin.sqlite", help="файл базы SQLite")
    p.add_argument("--config", default=None, help="папка конфигурации (по умолчанию config/)")
    args = p.parse_args(argv)
    conn = storage.connect(args.db)
    ing = Ingestor(conn, config_dir=args.config)
    for f in args.files:
        print(ing.ingest_file(f).to_text())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
