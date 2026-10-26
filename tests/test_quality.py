import copy
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from field_twin import config, quality, storage
from field_twin.gateway.tag_dictionary import TagDictionary


def defects(truth, kind):
    return [d for d in truth["defects"] if d["defect"] == kind]


def cell(data, d, param=None):
    m = (data.stream == d["stream"]) & (data.object_id == d["object_id"]) & (data.parameter == (param or d["parameter"]))
    return data[m & (data.ts == pd.Timestamp(d["ts"]))]


def window(data, d):
    m = (data.stream == d["stream"]) & (data.object_id == d["object_id"])
    return data[m & data.ts.between(pd.Timestamp(d["start"]), pd.Timestamp(d["end"]))]


# ---- критерии приёмки этапа 2 ----

def test_all_downtimes_flagged(cleaned, truth):
    tags = TagDictionary.load()
    found = defects(truth, "downtime")
    assert found
    for d in found:
        w = window(cleaned.data, d)
        assert len(w) and w["f_downtime"].all(), d
        if not d["equipment_state_in_file"]:
            # В архиве признака нет — выведен по нулевому расходу и помечен как вычисленный.
            assert (w["state_source"] == "derived").all() and w["f_state_derived"].all()


def test_well_stops_are_downtime(cleaned, truth):
    for d in defects(truth, "well_stop"):
        assert window(cleaned.data, d)["f_downtime"].all()


def test_all_non_numeric_flagged(cleaned, truth):
    found = defects(truth, "non_numeric")
    assert len(found) >= 10
    for d in found:
        c = cell(cleaned.data, d)
        assert c["f_non_numeric"].iloc[0] and pd.isna(c["value_clean"].iloc[0])


def test_all_nonphysical_flagged(cleaned, truth):
    found = defects(truth, "nonphysical")
    assert found
    for d in found:
        c = cell(cleaned.data, d)
        assert c["f_nonphysical"].iloc[0], d
        assert pd.isna(c["value_clean"].iloc[0])


def test_all_zero_flow_pressures_flagged(cleaned, truth):
    found = defects(truth, "zero_flow_pressure")
    assert found
    for d in found:
        assert cell(cleaned.data, d)["f_zero_flow_pressure"].iloc[0], d
    # Ничего лишнего: рост давления на остановленных скважинах — не аномалия.
    flagged = cleaned.data[cleaned.data["f_zero_flow_pressure"]]
    keys = {(d["stream"], d["object_id"], d["parameter"], pd.Timestamp(d["ts"])) for d in found}
    assert {(r.stream, r.object_id, r.parameter, r.ts) for r in flagged.itertuples()} == keys


def test_outliers_flagged_not_removed(cleaned, truth):
    found = defects(truth, "pressure_outlier")
    assert found
    for d in found:
        c = cell(cleaned.data, d)
        assert c["f_outlier"].iloc[0], d
        assert c["value_clean"].iloc[0] == c["value"].iloc[0]   # не удалён


def test_single_gap_restored_multi_gap_only_flagged(cleaned, truth):
    g = defects(truth, "gaps")[0]
    data = cleaned.data
    restored = unrestorable = 0
    for c in g["cells"]:
        rows = data[(data.stream == "wells") & (data.object_id == c["object_id"]) & (data.ts == pd.Timestamp(c["ts"]))
                    & data.parameter.isin(c["parameters"])]
        assert rows["f_gap"].all()
        if len(c["parameters"]) == 1:
            r = rows.iloc[0]
            if r["f_restored"]:
                restored += 1
                assert not pd.isna(r["value_clean"])
            else:
                unrestorable += 1
                assert r["f_gap_unrestorable"]
        else:
            assert rows["f_gap_multi"].all() and rows["value_clean"].isna().all() and not rows["f_restored"].any()
    assert restored > 0 and unrestorable < restored / 5


def test_unrestorable_only_at_series_edges(cleaned):
    data = cleaned.data
    bad = data[data["f_gap_unrestorable"] & ~data["f_downtime"]]
    for r in bad.itertuples():
        sig = data[(data.stream == r.stream) & (data.object_id == r.object_id) & (data.parameter == r.parameter)]
        ok = sig[sig["value_clean"].notna() & ~sig["f_restored"]]
        assert ok.empty or r.ts <= ok.ts.min() or r.ts >= ok.ts.max()


def test_restored_value_is_interpolated():
    df = _frame([("2026-07-01T00:00:00+00:00", "1.0"), ("2026-07-01T01:00:00+00:00", ""),
                 ("2026-07-01T02:00:00+00:00", "3.0")], parameter="P_wellhead", stream="wells", obj="W-101")
    res = quality.QualityPipeline().run(df)
    r = res.data.sort_values("ts").iloc[1]
    assert r["f_restored"] and r["value_clean"] == pytest.approx(2.0)


def test_quality_report_contents(cleaned):
    rep = cleaned.report
    assert 0 < rep.complete_rows_share_by_stream["wells"] < 0.35
    wells = {k: v for k, v in rep.gap_share_by_parameter.items() if k.startswith("wells.")}
    assert max(wells, key=wells.get) == "wells.water_cut"
    assert rep.flag_counts["restored"] > 0 and rep.flag_counts["gap_multi"] > 0
    assert {f.stream for f in rep.freshness} == set(TagDictionary.load().streams)
    text = rep.to_text()
    for s in ("доля строк без пропусков", "доля пропусков по параметрам", "wells.water_cut", "свежесть потоков"):
        assert s in text


def test_three_criteria_per_parameter(cleaned):
    crit = pd.DataFrame(cleaned.report.criteria)
    assert {"mean_median_diff", "share_within_3sigma", "pearson"} <= set(crit.columns)
    pin = crit[(crit.object_id == "SEP-1") & (crit.parameter == "P_in")].iloc[0]
    assert pin["pearson_with"] == "SEP-1.P_out" and pin["pearson"] > 0.5
    assert crit["share_within_3sigma"].dropna().between(0.9, 1).all()
    # Для постоянного ряда (нулевое отклонение) критерий не вычисляется.
    life = crit[crit.parameter == "safe_life_years"]
    assert life["share_within_3sigma"].isna().all()


def test_monthly_means(cleaned):
    m = cleaned.monthly
    # Данные начинаются 2026-07-01 00:00 +05:00 = 2026-06-30 19:00 UTC.
    assert set(m[m.stream == "absorber"].month) == {"2026-06", "2026-07"}
    assert len(set(m[m.stream == "geotech_radar"].month)) > 3   # история радарных замеров
    row = m[(m.stream == "absorber") & (m.object_id == "A-1") & (m.parameter == "Q_gas") & (m.month == "2026-07")].iloc[0]
    assert 200 < row["mean"] < 240 and row["n"] > 1000   # база 219,8 (табл. 3 Сафроновой)


def test_duplicates_dropped(ingested_db, samples, tmp_path):
    from field_twin.ingest import Ingestor
    c = storage.connect(tmp_path / "dup.sqlite")
    ing = Ingestor(c)
    ing.ingest_file(samples.files["dks"])
    ing.ingest_file(samples.files["dks"])
    res = quality.QualityPipeline().run(quality.load_measurements(c))
    assert res.report.duplicates_dropped == res.report.rows_total


# ---- исключение выбросов пользователем ----

def test_user_confirmed_exclusion(ingested_db, truth, tmp_path):
    import shutil
    db = tmp_path / "x.sqlite"
    shutil.copy(ingested_db, db)
    c = storage.connect(db)
    res = quality.QualityPipeline().run(quality.load_measurements(c))
    d = defects(truth, "pressure_outlier")[0]
    mid = int(cell(res.data, d)["id"].iloc[0])
    with pytest.raises(ValueError):
        quality.confirm_exclusion(c, [mid], user_name="")
    quality.confirm_exclusion(c, [mid], user_name="Иванов", comment="сбой датчика")
    res2 = quality.QualityPipeline().run(quality.load_measurements(c), exclusions=quality.load_exclusions(c))
    r = res2.data[res2.data.id == mid].iloc[0]
    assert r["f_excluded"] and pd.isna(r["value_clean"])


# ---- сглаживание полиномом ----

def test_smoothing_disabled_without_parameters(cleaned):
    assert not cleaned.report.smoothing_enabled
    assert not cleaned.data["f_smooth_removed"].any()


def test_smoothing_removes_points_off_polynomial(ingested_db, truth):
    s = copy.deepcopy(config.quality())
    s["smoothing"].update({"enabled": True, "degree": 2, "threshold_mode": "sigma", "threshold": 3, "window": 12})
    c = storage.connect(ingested_db)
    res = quality.QualityPipeline(settings=s).run(quality.load_measurements(c))
    assert res.report.smoothing_enabled
    wells = res.data[(res.data.stream == "wells") & res.data.parameter.isin(["P_wellhead", "P_bottomhole"])]
    assert wells["value_smooth"].notna().sum() > 0.5 * len(wells)
    assert (res.data[res.data.stream != "wells"]["value_smooth"].isna()).all()


def test_smoothing_on_noisy_series():
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2026-07-01T00:00:00Z")
    rows = []
    for i in range(24):
        v = 8 + 0.01 * i + rng.normal(0, 0.01)
        if i == 10:
            v += 0.5
        rows.append(((t0 + pd.Timedelta(hours=i)).isoformat(), f"{v:.5f}"))
    df = _frame(rows, parameter="P_wellhead", stream="wells", obj="W-101")
    s = copy.deepcopy(config.quality())
    s["smoothing"].update({"enabled": True, "degree": 2, "threshold_mode": "sigma", "threshold": 3, "window": 24})
    res = quality.QualityPipeline(settings=s).run(df).data.sort_values("ts")
    assert res["f_smooth_removed"].tolist() == [i == 10 for i in range(24)]


def test_smoothing_savchenko_mode_absolute_threshold_whole_series():
    # Режим статьи Савченко и соавторов: полином 5-й степени по всему ряду, порог 0,5 (абсолютный).
    s = copy.deepcopy(config.quality())
    assert s["smoothing"]["degree"] == 5 and s["smoothing"]["threshold_mode"] == "absolute"
    assert s["smoothing"]["window"] is None and s["smoothing"]["enabled"] is False
    rng = np.random.default_rng(1)
    t0 = pd.Timestamp("2026-07-01T00:00:00Z")
    rows = []
    for i in range(60):
        v = 9 + 0.02 * i + rng.normal(0, 0.05)
        if i in (20, 41):
            v += 0.8 if i == 20 else -0.7
        rows.append(((t0 + pd.Timedelta(hours=i)).isoformat(), f"{v:.5f}"))
    df = _frame(rows, parameter="P_bottomhole", stream="wells", obj="W-101")
    s["smoothing"]["enabled"] = True
    res = quality.QualityPipeline(settings=s).run(df)
    assert res.report.smoothing_enabled
    flags = res.data.sort_values("ts")["f_smooth_removed"].tolist()
    assert flags == [i in (20, 41) for i in range(60)]


# ---- настройки без правки кода ----

def test_user_range_narrows_physical(tmp_path):
    df = _frame([("2026-07-01T00:00:00+00:00", "5"), ("2026-07-01T00:05:00+00:00", "50")],
                parameter="P", stream="gathering", obj="N-01")
    s = copy.deepcopy(config.quality())
    assert not quality.QualityPipeline(settings=s).run(df).data["f_nonphysical"].any()
    s["ranges"] = {"gathering.P": {"min": 1, "max": 20}}
    flagged = quality.QualityPipeline(settings=s).run(df).data.sort_values("ts")["f_nonphysical"].tolist()
    assert flagged == [False, True]


def test_zero_pressure_is_nonphysical():
    df = _frame([("2026-07-01T00:00:00+00:00", "0")], parameter="P", stream="gathering", obj="N-01")
    assert quality.QualityPipeline().run(df).data["f_nonphysical"].iloc[0]


def test_results_saved(ingested_db, tmp_path):
    import shutil
    db = tmp_path / "s.sqlite"
    shutil.copy(ingested_db, db)
    c = storage.connect(db)
    res = quality.QualityPipeline().run(quality.load_measurements(c))
    run_id = quality.save(c, res)
    n = c.execute("SELECT COUNT(*) FROM clean_values WHERE run_id = ?", (run_id,)).fetchone()[0]
    assert n == res.report.rows_total
    flagged = c.execute("SELECT COUNT(*) FROM clean_values WHERE flags LIKE '%restored%'").fetchone()[0]
    assert flagged == res.report.flag_counts["restored"]
    assert c.execute("SELECT COUNT(*) FROM monthly_means").fetchone()[0] == len(res.monthly)
    # исходные записи не меняются
    assert c.execute("SELECT COUNT(*) FROM measurements WHERE value IS NULL AND value_raw = ''").fetchone()[0] > 0


def _frame(rows, parameter, stream, obj, unit="МПа"):
    out = []
    for i, (ts, raw) in enumerate(rows):
        out.append({"id": i + 1, "ts_utc": ts, "stream": stream, "object_id": obj, "parameter": parameter,
                    "value": float(raw) if raw not in ("",) else np.nan, "value_raw": raw, "unit": unit,
                    "quality": "good", "equipment_state": "run", "equipment_state_source": "input", "origin": "csv"})
    df = pd.DataFrame(out)
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    return df
