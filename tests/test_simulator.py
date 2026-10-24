import csv
from datetime import datetime, timedelta

import pandas as pd
import pytest

from field_twin.gateway.tag_dictionary import TagDictionary
from field_twin.simulator import COLUMNS, generate


def read(samples, stream):
    rows = list(csv.reader(open(samples.files[stream], encoding="utf-8")))
    header = rows[0]
    good = [r for r in rows[1:] if len(r) == len(header)]
    df = pd.DataFrame(good, columns=header)
    ok = df["timestamp"].str.match(r"^\d{4}-\d{2}-\d{2}T.*\+05:00$")
    df = df[ok].copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["v"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def defects(truth, kind, stream=None):
    return [d for d in truth["defects"] if d["defect"] == kind and (stream is None or d["stream"] == stream)]


def test_all_streams_written_in_long_format(samples):
    tags = TagDictionary.load()
    assert set(samples.files) == set(tags.streams)
    for stream, path in samples.files.items():
        header = next(csv.reader(open(path, encoding="utf-8")))
        assert header in (COLUMNS, COLUMNS[:-1]), stream


def test_intervals_match_dictionary(samples):
    for stream, step in [("absorber", "2min"), ("gathering", "5min"), ("geotech", "30min"), ("cluster", "1D")]:
        df = read(samples, stream)
        diffs = df["ts"].drop_duplicates().sort_values().diff().dropna().unique()
        assert list(diffs) == [pd.Timedelta(step)], stream


def test_deterministic_with_seed(tmp_path):
    a = generate(tmp_path / "a", days=0.5, seed=3)
    b = generate(tmp_path / "b", days=0.5, seed=3)
    for s in a.files:
        assert a.files[s].read_bytes() == b.files[s].read_bytes()


def test_archive_stream_has_no_equipment_state(samples, truth):
    # В архивах признака состояния нет (Шевченко, Рычков).
    assert "gathering" in truth["streams_without_equipment_state"]
    header = next(csv.reader(open(samples.files["gathering"], encoding="utf-8")))
    assert "equipment_state" not in header


def test_downtime_zero_flow(samples, truth):
    tags = TagDictionary.load()
    found = defects(truth, "downtime")
    assert found
    for d in found:
        df = read(samples, d["stream"])
        win = df[(df.object_id == d["object_id"]) & df.ts.between(pd.Timestamp(d["start"]), pd.Timestamp(d["end"]))]
        flows = [s.parameter for s in tags.stream_signals(d["stream"]) if s.kind == "flow" and s.object_id == d["object_id"]]
        assert (win[win.parameter.isin(flows)]["v"].fillna(0) == 0).all()
        if d["equipment_state_in_file"]:
            assert (win["equipment_state"] == "stop").all()


def test_zero_flow_pressure_is_anomalous(samples, truth):
    found = defects(truth, "zero_flow_pressure")
    assert found
    for d in found:
        df = read(samples, d["stream"])
        row = df[(df.object_id == d["object_id"]) & (df.parameter == d["parameter"]) & (df.ts == pd.Timestamp(d["ts"]))]
        v = row["v"].iloc[0]
        normal = df[(df.object_id == d["object_id"]) & (df.parameter == d["parameter"])]["v"].median()
        assert v < 0 or v > 3 * normal


def test_non_numeric_tokens_written(samples, truth):
    found = defects(truth, "non_numeric")
    assert len(found) >= 10
    for d in found:
        df = read(samples, d["stream"])
        row = df[(df.object_id == d["object_id"]) & (df.parameter == d["parameter"]) & (df.ts == pd.Timestamp(d["ts"]))]
        assert row["value"].iloc[0] == d["token"]


def test_gaps_share_and_main_parameter(samples, truth):
    g = defects(truth, "gaps", "wells")[0]
    share = g["complete_rows"] / g["rows"]
    assert 0.07 < share < 0.22   # «чуть менее 15 %» строк без пропусков (Печко и соавторы)
    assert g["single_gap_rows"] > g["multi_gap_rows"] > 0
    missing = [p for c in g["cells"] for p in c["parameters"]]
    assert max(set(missing), key=missing.count) == "water_cut"
    df = read(samples, "wells")
    c = g["cells"][0]
    row = df[(df.object_id == c["object_id"]) & (df.ts == pd.Timestamp(c["ts"])) & df.parameter.isin(c["parameters"])]
    assert (row["value"] == "").all()


def test_well_stops_monotonic_pressure(samples, truth):
    found = defects(truth, "well_stop")
    assert any(d["long_enough"] for d in found) and any(not d["long_enough"] for d in found)
    df = read(samples, "wells")
    for d in found:
        w = df[(df.object_id == d["object_id"]) & df.ts.between(pd.Timestamp(d["start"]), pd.Timestamp(d["end"]))]
        assert (w[w.parameter == "Q_gas"]["v"] == 0).all()
        p = w[w.parameter == "P_wellhead"].sort_values("ts")["v"].to_numpy()
        assert (p[1:] > p[:-1]).all()
        assert (d["hours"] >= d["min_hours"]) == d["long_enough"]


def test_hydrate_episode_signs(samples, truth):
    d = defects(truth, "hydrate_episode")[0]
    df = read(samples, "absorber")
    start, end = pd.Timestamp(d["start"]), pd.Timestamp(d["end"])
    before = df[df.ts.between(start - pd.Timedelta("2h"), start - pd.Timedelta("1min"))]
    late = df[df.ts.between(end - pd.Timedelta("1h"), end)]

    def mean(frame, obj, p):
        return frame[(frame.object_id == obj) & (frame.parameter == p)]["v"].mean()

    # Признаки Сафроновой: уровень в кубе ↓, клапан ↓, расход газа ↑, СИКГК ↓, Е-101 ↓, температура ↓.
    assert mean(late, "A-2", "L_cube") < mean(before, "A-2", "L_cube")
    assert mean(late, "A-2", "KRU") < mean(before, "A-2", "KRU")
    assert mean(late, "A-2", "Q_gas") > mean(before, "A-2", "Q_gas")
    assert mean(late, "A-2", "T_gas_absorber") < mean(before, "A-2", "T_gas_absorber")
    assert mean(late, "UKPG-1V", "Q_sikgk") < mean(before, "UKPG-1V", "Q_sikgk")
    assert mean(late, "UKPG-1V", "L_E101.1") < mean(before, "UKPG-1V", "L_E101.1")


def test_format_errors_recorded(truth):
    found = defects(truth, "format_error")
    assert len(found) == 5
    assert {d["expected_flag"] for d in found} == {
        "bad_timestamp", "timestamp_no_tz", "missing_unit", "wrong_field_count", "bad_quality"}


def test_radar_only_in_snowless_season(tmp_path):
    winter = generate(tmp_path / "w", days=1, seed=1, start="2026-01-10T00:00:00+05:00")
    rows = list(csv.reader(open(winter.files["geotech_radar"], encoding="utf-8")))
    assert len(rows) == 1  # только заголовок: зимой радара нет
