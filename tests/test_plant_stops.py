"""Этап 8: технологические остановки скважин (М10), стык с заводом (М11), правило П16."""

import copy
import json

import pandas as pd
import pytest

from field_twin import config, plant_link, rules, stops, storage
from field_twin.gateway.tag_dictionary import TagDictionary

TAGS = TagDictionary.load()
S = stops.settings()


def wells_frame(obj, rate, pressure, start="2026-07-01T00:00:00Z", step="1h"):
    t0 = pd.Timestamp(start)
    rows = []
    for i, (q, p) in enumerate(zip(rate, pressure)):
        ts = t0 + i * pd.Timedelta(step)
        for param, v in (("Q_gas", q), ("P_wellhead", p)):
            rows.append({"stream": "wells", "object_id": obj, "parameter": param, "ts": ts, "value": v,
                         "value_clean": v})
    return pd.DataFrame(rows)


# ---- М10 ----

def test_all_long_monotonic_stops_found(cleaned, truth):
    # Критерий приёмки: М10 находит на синтетике все остановки не короче заданной длительности
    # с монотонным ростом давления.
    found = stops.find_stops(cleaned.data, TAGS, S)
    for d in [x for x in truth["defects"] if x["defect"] == "well_stop"]:
        hit = [w for w in found if w.object_id == d["object_id"] and w.start == pd.Timestamp(d["start"])]
        assert hit, d
        w = hit[0]
        assert w.hours == pytest.approx(d["hours"]) and w.monotonic
        assert w.long_enough == d["long_enough"] and w.candidate == d["long_enough"]
    # других остановок нет: нули дебита только там, где их внедрил генератор
    assert len(found) == len([x for x in truth["defects"] if x["defect"] == "well_stop"])


@pytest.mark.parametrize("hours,candidate", [(3, False), (4, True), (6, True)])
def test_min_duration_high_permeability(hours, candidate):
    rate = [500] * 3 + [0] * hours + [500] * 3
    p = [8.0] * 3 + [8.0 + 0.1 * (i + 1) for i in range(hours)] + [8.0] * 3
    w = stops.find_stops(wells_frame("W-101", rate, p), TAGS, S)
    assert len(w) == 1 and w[0].candidate == candidate and w[0].min_hours == 4


def test_min_duration_low_permeability():
    rate = [500] + [0] * 8 + [500]
    p = [8.0] + [8.0 + 0.1 * (i + 1) for i in range(8)] + [8.0]
    w = stops.find_stops(wells_frame("W-105", rate, p), TAGS, S)[0]
    assert w.min_hours == 12 and not w.candidate


def test_non_monotonic_pressure_not_candidate():
    rate = [500] + [0] * 6 + [500]
    p = [8.0, 8.1, 8.2, 8.15, 8.3, 8.4, 8.5, 8.0]
    w = stops.find_stops(wells_frame("W-101", rate, p), TAGS, S)[0]
    assert w.long_enough and not w.monotonic and not w.candidate


def test_gap_inside_stop_bridged():
    rate = [500] + [0, 0, float("nan"), 0, 0] + [500]
    p = [8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.0]
    w = stops.find_stops(wells_frame("W-101", rate, p), TAGS, S)
    assert len(w) == 1 and w[0].hours == 5


def test_stops_saved_as_candidates(cleaned):
    conn = storage.connect(":memory:")
    found = stops.find_stops(cleaned.data, TAGS, S)
    stops.save_stops(conn, found)
    stops.save_stops(conn, found)      # повторное сохранение не дублирует
    assert len(stops.candidates(conn)) == sum(w.candidate for w in found)
    assert conn.execute("SELECT COUNT(*) FROM well_stops").fetchone()[0] == len(found)


# ---- П16 ----

def c5_frame(values):
    t0 = pd.Timestamp("2026-07-01T00:00:00Z")
    return pd.DataFrame([{"stream": "gas_composition", "object_id": "UKPG-1V-OUT", "parameter": "C5plus",
                          "ts": t0 + i * pd.Timedelta("1h"), "value": v, "value_clean": v, "f_downtime": False,
                          "f_outlier": False} for i, v in enumerate(values)])


def th16(lim):
    th = copy.deepcopy(config.thresholds())
    th["П16"]["max_c5plus_g_m3"] = lim
    return th


def test_p16_below_above():
    assert not rules.check_p16(c5_frame([1.49] * 3), th16(1.5))
    ev = rules.check_p16(c5_frame([1.51] * 3), th16(1.5))
    assert len(ev) == 1 and ev[0].recipient == "chief_engineer" and "вымерзают" in ev[0].message


def test_p16_disabled_without_threshold():
    assert config.thresholds()["П16"]["max_c5plus_g_m3"] is None
    assert rules.check_p16(c5_frame([99]), config.thresholds()) == []


def test_p16_on_synthetic(cleaned, truth):
    d = next(x for x in truth["defects"] if x["defect"] == "rule_event" and x["rule"] == "П16")
    ev = rules.check_p16(cleaned.data, th16(6.0))   # база 4 г/м³, событие 8 г/м³ — синтетика
    assert [e.ts for e in ev] == [pd.Timestamp(d["start"])]


# ---- М11 ----

def test_field_to_plant_message(cleaned):
    msg = plant_link.build_message(cleaned.data, cleaned.report.now)
    d = json.loads(msg.to_json())
    assert d["direction"] == "field_to_plant" and d["format_version"] == plant_link.FORMAT_VERSION
    assert set(d) >= {"composition", "volume", "separate", "drive_power", "windows"}   # пять потоков стыка
    assert d["composition"]["c5plus_g_m3"] is not None and d["volume"]["flow_thousand_m3_h"] is not None
    assert set(d["drive_power"]["available_mw"]) == {"GTU-T32-1", "DKS-1"}
    assert d["windows"][0]["kind"] == "supply"
    assert d["separate"]["ethane_rich_thousand_m3_h"] is None   # не измеряется — честно пусто


def test_plant_to_field_stub():
    text = json.dumps({"direction": "plant_to_field", "format_version": "1.0", "ts": "2026-07-01T00:00:00+05:00",
                       "intake_limit_pct": 70, "start": "2026-07-05T00:00:00+05:00",
                       "end": "2026-07-07T00:00:00+05:00", "reason": "ремонт линии сжижения"})
    msg = plant_link.PlantToField.from_json(text)
    note = plant_link.handle_plant_message(msg)
    assert note["recipient"] == "chief_engineer" and "70 %" in note["message"] and "режим кустов" in note["message"]
    bad = json.loads(text) | {"intake_limit_pct": 120}
    with pytest.raises(ValueError):
        plant_link.PlantToField.from_json(json.dumps(bad))
    with pytest.raises(ValueError, match="часового пояса"):
        plant_link.PlantToField.from_json(json.dumps(json.loads(text) | {"start": "2026-07-05T00:00:00"}))
