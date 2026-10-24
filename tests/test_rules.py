import copy

import numpy as np
import pandas as pd
import pytest

from field_twin import config, quality, rules, storage
from field_twin.journal import Journal


def stream_frame(stream, obj, param, start, n, step):
    t0 = pd.Timestamp(start)
    rows = [{"id": i + 1, "ts_utc": (t0 + i * pd.Timedelta(step)).isoformat(), "stream": stream, "object_id": obj,
             "parameter": param, "value": 7.5, "value_raw": "7.5", "unit": "МПа", "quality": "good",
             "equipment_state": "run", "equipment_state_source": "input", "origin": "csv"} for i in range(n)]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    return df


# ---- П13 ----

@pytest.mark.parametrize("extra,stale", [("10min", False), ("10min 1s", True), ("9min", False), ("1h", True)])
def test_p13_two_intervals_boundary(extra, stale):
    # Поток с интервалом 5 мин устарел, если не обновлялся ДОЛЬШЕ двух интервалов (10 мин).
    df = stream_frame("gathering", "N-01", "P", "2026-07-01T00:00:00Z", 12, "5min")
    last = df["ts"].max()
    res = quality.QualityPipeline().run(df, now=last + pd.Timedelta(extra))
    events = rules.check_p13(res.report, config.thresholds())
    assert bool(events) == stale
    if stale:
        e = events[0]
        assert (e.code, e.stream, e.recipient) == ("П13", "gathering", "operator")
        assert "не обновлялся" in e.message


def test_p13_factor_from_config():
    df = stream_frame("gathering", "N-01", "P", "2026-07-01T00:00:00Z", 12, "5min")
    th = copy.deepcopy(config.thresholds())
    th["П13"]["stale_after_intervals"] = 4
    res = quality.QualityPipeline(thresholds=th).run(df, now=df["ts"].max() + pd.Timedelta("15min"))
    assert not rules.check_p13(res.report, th)


def test_p13_interval_estimated_when_unknown():
    df = stream_frame("wells", "W-101", "P_wellhead", "2026-07-01T00:00:00Z", 10, "1h")
    res = quality.QualityPipeline().run(df, now=df["ts"].max() + pd.Timedelta("3h"))
    f = res.report.freshness[0]
    assert f.interval_estimated and f.interval == pd.Timedelta("1h") and f.stale
    assert "оценён по данным" in rules.check_p13(res.report, config.thresholds())[0].message


def test_p13_needs_timezone():
    df = stream_frame("gathering", "N-01", "P", "2026-07-01T00:00:00Z", 3, "5min")
    with pytest.raises(ValueError):
        quality.QualityPipeline().run(df, now=pd.Timestamp("2026-07-01T05:00:00"))


def test_p13_on_samples_archive_mode(cleaned):
    # Проверка архива: «сейчас» — последнее значение; ни один поток с известным интервалом не устарел.
    assert not cleaned.report.stale_streams


# ---- П14 ----

def thresholds_with(min_pearson):
    th = copy.deepcopy(config.thresholds())
    th["П14"]["min_pearson"] = min_pearson
    return th


def test_p14_disabled_without_threshold(cleaned):
    assert config.thresholds()["П14"]["min_pearson"] is None
    assert rules.check_p14(cleaned.data, config.quality(), config.thresholds()) == []


def test_p14_fires_on_decorrelation(cleaned, truth):
    d = [x for x in truth["defects"] if x["defect"] == "decorrelation"][0]
    start, end = pd.Timestamp(d["start"]), pd.Timestamp(d["end"])
    events = rules.check_p14(cleaned.data, config.quality(), thresholds_with(0.8))
    assert events
    inside = [e for e in events if pd.Timestamp(e.payload["start"]) >= start - pd.Timedelta("2h")
              and pd.Timestamp(e.payload["end"]) <= end + pd.Timedelta("2h")]
    assert len(inside) >= 2               # окна по 2 ч внутри 6-часового эпизода
    assert len(inside) == len(events)     # вне эпизода корреляция выше порога
    assert all(e.code == "П14" and e.recipient == "operator" and e.object_id == "SEP-1" for e in events)


def test_p14_threshold_boundary(cleaned):
    win = rules.p14_windows(cleaned.data, config.quality()).dropna()
    r = win["pearson"].min()
    assert not rules.check_p14(cleaned.data, config.quality(), thresholds_with(r))          # не ниже порога
    assert len(rules.check_p14(cleaned.data, config.quality(), thresholds_with(r + 1e-9))) == 1


def test_p14_ignores_downtime_and_outliers():
    rng = np.random.default_rng(1)
    t0 = pd.Timestamp("2026-07-01T00:00:00Z")
    rows = []
    for i in range(48):
        base = 7.4 + 0.05 * np.sin(i / 5) + rng.normal(0, 0.02)
        for p, v in (("P_in", base), ("P_out", base - 0.2 + rng.normal(0, 0.003))):
            rows.append({"id": len(rows) + 1, "ts_utc": (t0 + pd.Timedelta(minutes=5 * i)).isoformat(),
                         "stream": "gathering", "object_id": "SEP-1", "parameter": p, "value": v,
                         "value_raw": str(v), "unit": "МПа", "quality": "good",
                         "equipment_state": "stop" if 30 <= i < 36 else "run",
                         "equipment_state_source": "input", "origin": "csv"})
    df = pd.DataFrame(rows)
    df.loc[(df.parameter == "P_out") & df.ts_utc.str.contains("T02:3"), "value"] = 0.5   # мусор во время простоя
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    res = quality.QualityPipeline().run(df)
    assert rules.check_p14(res.data, config.quality(), thresholds_with(0.8)) == []


def test_rule_events_logged_to_journal(cleaned):
    conn = storage.connect(":memory:")
    j = Journal(conn)
    events = rules.check_p14(cleaned.data, config.quality(), thresholds_with(0.8))
    ids = rules.log_events(j, events, shadow=True)
    logged = j.events(code="П14")
    assert len(logged) == len(ids) == len(events)
    assert all(e["kind"] == "rule" and e["shadow"] == 1 for e in logged)
    assert '"recipient": "operator"' in logged[0]["payload_json"]
