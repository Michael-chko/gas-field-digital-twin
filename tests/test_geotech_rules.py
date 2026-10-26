"""Этап 3: геотехника (М5) и пороговые правила П2–П11."""

import copy

import numpy as np
import pandas as pd
import pytest

from field_twin import config, geotech, quality, rules
from field_twin.cycle import run_cycle
from field_twin.gateway.tag_dictionary import TagDictionary

TH = config.thresholds()
TAGS = TagDictionary.load()


def frame(stream, obj, param, values, start="2026-07-01T00:00:00Z", step="1h", state="run"):
    t0 = pd.Timestamp(start)
    rows = []
    for i, v in enumerate(values):
        rows.append({"id": i + 1, "ts_utc": (t0 + i * pd.Timedelta(step)).isoformat(), "stream": stream,
                     "object_id": obj, "parameter": param, "value": float(v), "value_raw": repr(float(v)),
                     "unit": "", "quality": "good", "equipment_state": state, "equipment_state_source": "input",
                     "origin": "csv"})
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    return df


def clean(*frames):
    df = pd.concat(frames, ignore_index=True)
    df["id"] = range(1, len(df) + 1)
    return quality.QualityPipeline().run(df).data


# ---- П2: классы относительно предела ----

@pytest.mark.parametrize("ratio,color", [(0.5, "green"), (2 / 3, "yellow"), (0.9, "yellow"), (1.0, "red"), (1.2, "red")])
def test_p2_classes(ratio, color):
    # Критерий приёмки: 0,5, 2/3 и 1,0 от предела — зелёный, жёлтый, красный.
    for limit in (2, 20, 40, 3):
        assert geotech.classify(limit * ratio, limit) == color


def test_p2_just_below_two_thirds_is_green():
    assert geotech.classify(20 * 2 / 3 - 0.001, 20) == "green"
    assert geotech.classify(20 - 0.001, 20) == "yellow"


def test_p2_titles():
    assert geotech.TITLES == {"green": "ниже предельного", "yellow": "ниже предельного, внимание!",
                              "red": "выше предельного"}


def test_object_class_is_worst_parameter():
    data = clean(frame("geotech", "GP-01", "settlement", [10] * 5, step="30min"),
                 frame("geotech", "GP-01", "tilt", [1.5] * 5, step="30min"))
    a = geotech.assess(data, TAGS, TH, data["ts"].max())
    obj = next(o for o in a.objects if o.object_id == "GP-01")
    assert obj.color == "yellow"   # наклон 1,5 из 2 = 75 %
    events = rules.check_geotech(a, TH, data["ts"].max())
    assert [e.payload["parameter"] for e in events if e.code == "П2"] == ["tilt"]


# ---- П3: скорость осадки ----

def settlement_series(rate_mm_y, days=60):
    hours = np.arange(days * 24)
    return frame("geotech", "GP-03", "settlement", 10 + rate_mm_y * hours / (365.25 * 24))


@pytest.mark.parametrize("rate,fires", [(14.5, False), (15.5, True)])
def test_p3_rate_threshold(rate, fires):
    data = clean(settlement_series(rate))
    a = geotech.assess(data, TAGS, TH, data["ts"].max())
    o = next(o for o in a.objects if o.object_id == "GP-03")
    assert o.rate_mm_per_year == pytest.approx(rate, rel=1e-3)
    assert any(e.code == "П3" for e in rules.check_geotech(a, TH, data["ts"].max())) == fires


def test_p3_needs_long_series():
    data = clean(settlement_series(100, days=5))
    a = geotech.assess(data, TAGS, TH, data["ts"].max())
    assert next(o for o in a.objects if o.object_id == "GP-03").rate_mm_per_year is None


# ---- П4–П10: чуть ниже и чуть выше порога ----

CASES = [
    ("П4", "geotech", "GP-04", "settlement_unevenness", 1.99, 2.01),
    ("П5", "pipelines", "PL-01", "gas_velocity", 19.9, 20.1),
    ("П6", "gtu", "GTU-T32-1", "T_exhaust", 599, 601),
    ("П8", "gtu", "GTU-T32-1", "surge_margin", 14.1, 13.9),
    ("П8", "gtu", "GTU-T32-1", "surge_margin", 19.9, 20.1),
    ("П9", "pipelines", "PL-01", "safe_life_years", 5.1, 4.9),
    ("П10", "ecp", "ECP-C1", "current", 0.01, 0.0),
    ("П10", "ecp", "ECP-C1", "potential", -0.5, 0.0),
]


@pytest.mark.parametrize("code,stream,obj,param,below,above", CASES)
def test_threshold_just_below_and_above(code, stream, obj, param, below, above):
    def fired(v):
        data = clean(frame(stream, obj, param, [v] * 3))
        now = data["ts"].max()
        if code == "П4":
            evs = rules.check_geotech(geotech.assess(data, TAGS, TH, now), TH, now)
        else:
            evs = rules.check_series_rules(data, TH, TAGS, now)
        return [e for e in evs if e.code == code]
    assert not fired(below)
    ev = fired(above)
    assert len(ev) == 1 and ev[0].object_id == obj and ev[0].recipient == "chief_engineer"


def test_p10_near_zero_configurable():
    th = copy.deepcopy(TH)
    th["П10"]["near_zero_abs"] = 0.05
    data = clean(frame("ecp", "ECP-C1", "current", [0.04]))
    assert rules.check_p10(data, th) and not rules.check_p10(data, TH)


@pytest.mark.parametrize("drop,fires", [(3.9, False), (4.1, True)])
def test_p7_power_drop_from_nominal(drop, fires):
    # Снижение по сравнению с номиналом 31,2 МВт (Поярков, Филиппов, Юн, табл. 1).
    nom = TH["П7"]["nominal"]["N_power"]
    data = clean(frame("gtu", "GTU-T32-1", "N_power", [nom * (1 - drop / 100)] * 48))
    evs = rules.check_p7(data, TH)
    assert bool(evs) == fires
    if fires:
        assert evs[0].payload["drop_pct"] == pytest.approx(drop, rel=1e-6)
        assert evs[0].payload["baseline"] == nom and "от номинала" in evs[0].message


@pytest.mark.parametrize("drop,fires", [(1.9, False), (2.1, True)])
def test_p7_efficiency_drop_from_nominal(drop, fires):
    nom = TH["П7"]["nominal"]["efficiency"]
    data = clean(frame("gtu", "GTU-T32-1", "efficiency", [nom * (1 - drop / 100)] * 48))
    assert bool(rules.check_p7(data, TH)) == fires


@pytest.mark.parametrize("drop,fires", [(3.9, False), (4.1, True)])
def test_p7_period_start_mode(drop, fires):
    th = copy.deepcopy(TH)
    th["П7"]["baseline"] = "period_start"
    base = [32.0] * 24 + [32.0 * (1 - drop / 100)] * 48
    evs = rules.check_p7(clean(frame("gtu", "GTU-T32-1", "N_power", base)), th)
    assert bool(evs) == fires
    if fires:
        assert evs[0].payload["drop_pct"] == pytest.approx(drop, rel=1e-6)


@pytest.mark.parametrize("obj,days,fires", [
    ("ECP-C1", 14.9, False), ("ECP-C1", 15.1, True),      # катодная: 2 раза в месяц
    ("ECP-D1", 7.4, False), ("ECP-D1", 7.6, True),        # дренажная: 4 раза в месяц
    ("ECP-P1", 181, False), ("ECP-P1", 183, True),        # протекторная: раз в 6 месяцев
])
def test_p11_maintenance_overdue(obj, days, fires):
    data = clean(frame("ecp", obj, "current", [5.0]))
    now = data["ts"].max() + pd.Timedelta(days=days)
    evs = [e for e in rules.check_p11(data, TH, TAGS, now) if e.object_id == obj]
    assert bool(evs) == fires


def test_episode_fires_once():
    data = clean(frame("gtu", "GTU-T32-1", "T_exhaust", [500, 620, 630, 610, 500, 500, 650, 500]))
    evs = rules.check_p6(data, TH)
    assert [e.payload["points"] for e in evs] == [3, 1]


def test_rules_ignore_downtime():
    data = clean(frame("gtu", "GTU-T32-1", "T_exhaust", [700] * 3, state="stop"))
    assert not rules.check_p6(data, TH)


def test_configurable_recipient():
    th = copy.deepcopy(TH)
    th["configurable_recipient"] = "operator"
    data = clean(frame("gtu", "GTU-T32-1", "T_exhaust", [650]))
    assert rules.check_p6(data, th)[0].recipient == "operator"


# ---- весь цикл на синтетике ----

def test_cycle_finds_all_rule_events(ingested_db, truth, tmp_path):
    import shutil
    from field_twin import storage
    db = tmp_path / "c.sqlite"
    shutil.copy(ingested_db, db)
    res = run_cycle(storage.connect(db))
    # П16 без порога (задаёт пользователь) не срабатывает — проверяется отдельно в test_plant_stops.py
    expected = [d for d in truth["defects"] if d["defect"] == "rule_event" and d["rule"] != "П16"]
    got = {(e.code, e.object_id) for e in res.events}
    for d in expected:
        assert (d["rule"], d["object_id"]) in got, d
    # лишних срабатываний П2–П11 нет
    want = {(d["rule"], d["object_id"]) for d in expected}
    want |= {("П2", d["object_id"]) for d in expected if d["rule"] == "П4"}   # 2,5 мм/м — ещё и красный класс
    assert {k for k in got if k[0] not in ("П12", "П13", "П14")} == want
    assert len(res.journal_ids) == len(res.events)


def test_last_radar_date_reported(cleaned):
    a = geotech.assess(cleaned.data, TAGS, TH, cleaned.report.now)
    assert a.last_radar is not None
    assert all(o.last_radar is not None for o in a.objects)
    assert "последний радарный замер" in a.to_text()
