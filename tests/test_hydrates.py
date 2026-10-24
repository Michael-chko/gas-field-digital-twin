"""Этап 4: классификатор гидратов (М3), правило П1, теневой режим."""

import copy

import pandas as pd
import pytest

from field_twin import config, hydrates, notify, quality, storage
from field_twin.ingest import Ingestor
from field_twin.journal import Journal
from field_twin.simulator import generate

T0 = pd.Timestamp("2026-07-01T00:00:00Z")


def preds(minutes, obj="A-2", total=60):
    """Выход модели каждые 2 мин; 1 — в указанные минуты."""
    rows = [{"ts": T0 + pd.Timedelta(minutes=m), "object_id": obj, "proba": float(m in minutes),
             "label": int(m in minutes)} for m in range(0, total, 2)]
    return pd.DataFrame(rows)


# ---- П1: критерии приёмки ----

def test_p1_two_positives_no_warning():
    assert notify.evaluate_p1(preds({10, 20})) == []


def test_p1_three_positives_warning_with_sound():
    notes = notify.evaluate_p1(preds({10, 20, 30}))
    assert len(notes) == 1
    n = notes[0]
    assert (n.code, n.object_id, n.sound, n.shadow, n.recipient) == ("П1", "A-2", True, False, "operator")
    assert n.ts == T0 + pd.Timedelta(minutes=30)
    assert "уносы из абсорбера А-2" in n.message and "метанол" in n.message
    assert n.delay == pd.Timedelta(minutes=20)    # от первого срабатывания модели


def test_p1_shadow_logged_without_sound():
    conn = storage.connect(":memory:")
    j = Journal(conn)
    notes = notify.evaluate_p1(preds({10, 20, 30}), shadow=True)
    notify.log_notifications(j, notes)
    ev = j.events(code="П1")
    assert len(ev) == 1 and ev[0]["shadow"] == 1 and '"sound": false' in ev[0]["payload_json"]


@pytest.mark.parametrize("minutes,fires", [({0, 10, 20}, True), ({0, 10, 22}, False), ({0, 2, 4}, True)])
def test_p1_window_boundary(minutes, fires):
    assert bool(notify.evaluate_p1(preds(minutes))) == fires


def test_p1_fires_once_per_streak_and_rearms():
    notes = notify.evaluate_p1(preds(set(range(10, 40, 2)) | {80, 82, 84}, total=100))
    assert [n.ts for n in notes] == [T0 + pd.Timedelta(minutes=14), T0 + pd.Timedelta(minutes=84)]


def test_p1_per_object():
    p = pd.concat([preds({10, 20, 30}, "A-1"), preds({10, 20}, "A-2")])
    assert [n.object_id for n in notify.evaluate_p1(p)] == ["A-1"]


def test_p1_threshold_from_config():
    th = copy.deepcopy(config.thresholds())
    th["П1"]["min_positive"] = 2
    assert notify.evaluate_p1(preds({10, 20}), th)


def test_shadow_comparison():
    notes = notify.evaluate_p1(preds({10, 20, 30}) , shadow=True) + notify.evaluate_p1(preds({40, 42, 44}, "A-1"), shadow=True)
    periods = [{"object_id": "A-2", "start_utc": T0.isoformat(), "end_utc": (T0 + pd.Timedelta("1h")).isoformat()},
               {"object_id": "A-2", "start_utc": (T0 + pd.Timedelta("5h")).isoformat(),
                "end_utc": (T0 + pd.Timedelta("6h")).isoformat()}]
    r = notify.shadow_comparison(notes, periods)
    assert (r["warnings"], r["hits"], r["false_alarms"], r["missed_episodes"]) == (2, 1, 1, 1)


# ---- М3 ----

@pytest.fixture(scope="module")
def training(tmp_path_factory):
    cfg = config.simulator()
    cfg["defects"]["hydrates"]["episodes"] = 3
    res = generate(tmp_path_factory.mktemp("h"), days=6, seed=11, defects=cfg["defects"])
    c = storage.connect(":memory:")
    ing = Ingestor(c)
    for s in ("absorber", "methanol"):
        ing.ingest_file(res.files[s])
    data = quality.QualityPipeline().run(quality.load_measurements(c)).data
    s = hydrates.settings()
    feats = hydrates.build_features(data, s)
    y = hydrates.labels_from_periods(feats, hydrates.periods_from_truth(res.truth))
    return res, feats, y, s


def test_features_have_lags_and_methanol(training):
    _, feats, _, s = training
    cols = hydrates.feature_columns(feats)
    for c in ("L_cube", "L_cube.d1", "L_cube.d200", "plant.Q_sikgk", "plant.Q_sikgk.d200", "methanol.M-1"):
        assert c in cols
    assert set(feats["object_id"]) == {"A-1", "A-2"}
    step = feats[feats.object_id == "A-1"]["ts"].diff().dropna().unique()
    assert list(step) == [pd.Timedelta("2min")]


def test_model_trains_and_reports_metrics(training, tmp_path):
    # Критерий приёмки: М3 обучается на синтетических эпизодах и выводит матрицу ошибок и метрики.
    _, feats, y, s = training
    m = hydrates.train(feats, y, s, trials=3)
    met = m.metrics
    assert met.tp + met.fn > 0 and met.tn > 0
    assert met.f1 > 0.6 and met.recall > 0.6
    text = met.to_text(s["reference_metrics"])
    for word in ("Матрица ошибок", "Точность", "Полнота", "F1", "Доля правильных", "ориентир"):
        assert word in text
    path = tmp_path / "m.cbm"
    m.save(path)
    m2 = hydrates.HydrateModel.load(path)
    pd.testing.assert_frame_equal(m.predict(feats.head(500)), m2.predict(feats.head(500)))
    assert m2.metrics.to_dict() == met.to_dict()


def test_model_finds_episodes_and_p1_warns(training):
    res, feats, y, s = training
    m = hydrates.train(feats, y, s, trials=3)
    notes = notify.evaluate_p1(m.predict(feats))
    cmp = notify.shadow_comparison(notes, hydrates.periods_from_truth(res.truth))
    assert cmp["missed_episodes"] == 0 and cmp["hits"] >= 1


def test_training_needs_both_classes(training):
    _, feats, y, s = training
    with pytest.raises(ValueError):
        hydrates.train(feats, y * 0, s, trials=1)


def test_labels_from_journal(training):
    _, feats, _, _ = training
    conn = storage.connect(":memory:")
    j = Journal(conn)
    t = feats["ts"].iloc[1000]
    e1 = j.log_event("notification", code="П1", ts=t.to_pydatetime(), object_id="A-2")
    j.record_decision(e1, "confirm", "hydrate", "Иванов", "operator")
    e2 = j.log_event("notification", code="П1", ts=t.to_pydatetime(), object_id="A-1")
    j.record_decision(e2, "reject", "false_alarm", "Иванов", "operator")
    y, known = hydrates.labels_from_journal(conn, feats)
    assert y[(feats.object_id == "A-2") & (feats.ts == t)].iloc[0] == 1
    assert y[(feats.object_id == "A-1") & (feats.ts == t)].iloc[0] == 0
    assert known[(feats.object_id == "A-1") & (feats.ts == t)].iloc[0]


def test_cycle_with_model_logs_p1(training, ingested_db, tmp_path):
    import shutil
    from field_twin.cycle import run_cycle
    _, feats, y, s = training
    m = hydrates.train(feats, y, s, trials=2)
    db = tmp_path / "c.sqlite"
    shutil.copy(ingested_db, db)
    conn = storage.connect(db)
    res = run_cycle(conn, hydrate_model=m, shadow=True)
    assert res.notifications and all(not n.sound for n in res.notifications)
    j = Journal(conn)
    assert len(j.events(code="П1")) == len(res.notifications)
    assert len(j.events(kind="model", code="М3")) == int(res.hydrate_predictions["label"].sum())
