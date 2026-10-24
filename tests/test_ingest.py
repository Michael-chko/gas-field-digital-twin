import textwrap

import pytest

from field_twin.gateway.tag_dictionary import TagDictionary
from field_twin.ingest import Ingestor, parse_timestamp
from datetime import timezone

HEADER = "timestamp,stream,object_id,parameter,value,unit,quality,equipment_state\n"
GOOD = "2026-07-01T05:00:00+05:00,absorber,A-1,Q_gas,450.1,тыс. м3/ч,good,run\n"


def write(tmp_path, text, name="in.csv", encoding="utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return p


def statuses(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM measurements ORDER BY line_no")]


def test_file_with_bad_row_loads_and_row_is_reported(tmp_path, conn):
    # Критерий приёмки: файл с ошибочной строкой загружается; строка помечена и видна в отчёте.
    p = write(tmp_path, HEADER + GOOD + "not-a-time,absorber,A-1,Q_gas,451,тыс. м3/ч,good,run\n" + GOOD)
    rep = Ingestor(conn).ingest_file(p)
    assert (rep.rows_total, rep.rows_ok, rep.rows_error) == (3, 2, 1)
    assert rep.error_rows[0].line_no == 3 and "bad_timestamp" in rep.error_rows[0].flags
    assert "строка 3" in rep.to_text()
    rows = statuses(conn)
    assert len(rows) == 3  # ошибочная строка не отброшена
    assert rows[1]["ingest_status"] == "error" and "bad_timestamp" in rows[1]["ingest_flags"]
    assert rows[1]["raw_line"].startswith("not-a-time")


def test_time_normalized_to_one_zone(tmp_path, conn):
    p = write(tmp_path, HEADER + GOOD + "2026-07-01T03:00:00+03:00,absorber,A-1,Q_gas,450,тыс. м3/ч,good,run\n")
    Ingestor(conn).ingest_file(p)
    ts = [r["ts_utc"] for r in statuses(conn)]
    assert ts == ["2026-07-01T00:00:00+00:00", "2026-07-01T00:00:00+00:00"]


def test_target_timezone_configurable(tmp_path, conn):
    p = write(tmp_path, HEADER + GOOD)
    Ingestor(conn, settings={"target_timezone": "Asia/Yekaterinburg"}).ingest_file(p)
    assert statuses(conn)[0]["ts_utc"] == "2026-07-01T05:00:00+05:00"


@pytest.mark.parametrize("text,flag", [
    ("2026-07-01T05:00:00", "timestamp_no_tz"),
    ("", "bad_timestamp"),
    ("01.07.2026 05:00", "bad_timestamp"),
])
def test_parse_timestamp_errors(text, flag):
    assert parse_timestamp(text, timezone.utc) == (None, flag)


@pytest.mark.parametrize("row,flag", [
    ("2026-07-01T05:00:00+05:00,absorber,A-1,Q_gas,450,,good,run", "missing_unit"),
    ("2026-07-01T05:00:00+05:00,,A-1,Q_gas,450,тыс. м3/ч,good,run", "missing_stream"),
    ("2026-07-01T05:00:00+05:00,absorber,,Q_gas,450,тыс. м3/ч,good,run", "missing_object"),
    ("2026-07-01T05:00:00+05:00,absorber,A-1,,450,тыс. м3/ч,good,run", "missing_parameter"),
    ("2026-07-01T05:00:00+05:00,absorber,A-1,Q_gas,450,тыс. м3/ч,,run", "missing_quality"),
    ("2026-07-01T05:00:00+05:00,absorber,A-1,Q_gas,450,тыс. м3/ч,ok?,run", "bad_quality"),
    ("2026-07-01T05:00:00+05:00,absorber,A-1,Q_gas,450,тыс. м3/ч,good,сломан", "bad_equipment_state"),
    ("2026-07-01T05:00:00+05:00,absorber,A-1,Q_gas,450,тыс. м3/ч", "wrong_field_count"),
])
def test_required_fields_checked(tmp_path, conn, row, flag):
    rep = Ingestor(conn).ingest_file(write(tmp_path, HEADER + row + "\n"))
    assert rep.rows_error == 1 and flag in rep.error_rows[0].flags


def test_missing_equipment_state_value_is_warning(tmp_path, conn):
    rep = Ingestor(conn).ingest_file(write(tmp_path, HEADER + GOOD.replace(",run", ",")))
    assert (rep.rows_warning, rep.rows_error) == (1, 0)
    row = statuses(conn)[0]
    assert row["equipment_state"] is None and row["equipment_state_source"] == "absent"


def test_file_without_equipment_state_column(tmp_path, conn):
    header = HEADER.replace(",equipment_state", "")
    rep = Ingestor(conn).ingest_file(write(tmp_path, header + GOOD.replace(",run", "")))
    assert not rep.equipment_state_column and rep.rows_ok == 1
    assert "выведет М2" in rep.to_text()
    assert statuses(conn)[0]["equipment_state_source"] == "absent"


def test_missing_required_column_rejects_file(tmp_path, conn):
    rep = Ingestor(conn).ingest_file(write(tmp_path, "timestamp,stream,value\n2026-07-01T05:00:00+05:00,absorber,1\n"))
    assert set(rep.missing_columns) == {"object_id", "parameter", "unit", "quality"}
    assert "не принят" in rep.to_text()


def test_unmapped_signal_and_unit_mismatch_reported(tmp_path, conn):
    text = HEADER + GOOD.replace("Q_gas", "Q_unknown") + GOOD.replace("тыс. м3/ч", "м3/ч")
    rep = Ingestor(conn).ingest_file(write(tmp_path, text))
    assert ("absorber", "A-1", "Q_unknown") in rep.unmapped_signals
    assert rep.flag_counts["unit_mismatch"] == 1
    assert "absorber.A-1.Q_unknown" in rep.to_text()
    assert rep.rows_warning == 2


def test_non_numeric_and_empty_values_kept_raw(tmp_path, conn):
    text = HEADER + GOOD.replace("450.1", "#N/A") + GOOD.replace("450.1", "")
    rep = Ingestor(conn).ingest_file(write(tmp_path, text))
    assert rep.rows_ok == 2  # не ошибки формата: обработает М2
    rows = statuses(conn)
    assert rows[0]["value"] is None and rows[0]["value_raw"] == "#N/A"
    assert rows[1]["value"] is None and rows[1]["value_raw"] == ""


def test_delimiter_and_encoding_from_settings(tmp_path, conn):
    text = (HEADER + GOOD).replace(",", ";")
    p = write(tmp_path, text, encoding="cp1251")
    rep = Ingestor(conn, settings={"delimiter": ";", "encoding": "cp1251"}).ingest_file(p)
    assert rep.rows_ok == 1
    assert statuses(conn)[0]["unit"] == "тыс. м3/ч"


def test_report_saved_in_batch(tmp_path, conn):
    rep = Ingestor(conn).ingest_file(write(tmp_path, HEADER + GOOD))
    saved = conn.execute("SELECT report_json FROM ingest_batches WHERE id = ?", (rep.batch_id,)).fetchone()[0]
    assert '"rows_ok": 1' in saved


def test_generated_samples_all_format_errors_found(samples, truth, conn):
    ing = Ingestor(conn)
    reports = {s: ing.ingest_file(p) for s, p in samples.files.items()}
    expected = [d for d in truth["defects"] if d["defect"] == "format_error"]
    for d in expected:
        rep = reports[d["stream"]]
        hit = [r for r in rep.error_rows if r.line_no == d["line_no"]]
        assert hit and d["expected_flag"] in hit[0].flags, d
    assert sum(r.rows_error for r in reports.values()) == len(expected)
    assert all(not r.unmapped_signals for r in reports.values())
    assert sum(r.flag_counts["unit_mismatch"] for r in reports.values()) == 0
