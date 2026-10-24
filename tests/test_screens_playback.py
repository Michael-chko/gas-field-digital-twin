"""Этап 5: экраны оператора и главного инженера, проигрывание архива (М7)."""

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from field_twin import config, hydrates, quality, storage
from field_twin.api import WEB, create_app
from field_twin.ingest import Ingestor
from field_twin.playback import Playback
from field_twin.simulator import generate

STREAMS = ["absorber", "methanol", "gathering", "ecp", "geotech", "geotech_radar", "pipelines"]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    cfg = config.simulator()
    cfg["defects"]["hydrates"]["episodes"] = 3
    res = generate(tmp_path_factory.mktemp("train"), days=6, seed=11, defects=cfg["defects"])
    c = storage.connect(":memory:")
    ing = Ingestor(c)
    for s in ("absorber", "methanol"):
        ing.ingest_file(res.files[s])
    data = quality.QualityPipeline().run(quality.load_measurements(c)).data
    s = hydrates.settings()
    feats = hydrates.build_features(data, s)
    return hydrates.train(feats, hydrates.labels_from_periods(feats, hydrates.periods_from_truth(res.truth)), s, trials=2)


@pytest.fixture(scope="module")
def played(model, tmp_path_factory):
    """Архив 18 ч с эпизодом гидратов с 9-го часа; поток сборной сети обрывается через 6 ч."""
    out = tmp_path_factory.mktemp("archive")
    cfg = config.simulator()
    cfg["defects"]["hydrates"]["start_hours"] = 9
    cfg["defects"]["format_errors"] = 0
    res = generate(out, days=0.75, seed=5, defects=cfg["defects"])
    gath = res.files["gathering"]
    lines = gath.read_text(encoding="utf-8").splitlines()
    cut = pd.Timestamp(res.truth["start"]) + pd.Timedelta("6h")
    keep = [lines[0]] + [l for l in lines[1:] if pd.Timestamp(l.split(",")[0]) < cut]
    gath.write_text("\n".join(keep) + "\n", encoding="utf-8")
    db = out / "pb.sqlite"
    pb = Playback(db, [res.files[s] for s in STREAMS], speed=1e9, cycle_minutes=60, hydrate_model=model)
    cycles = pb.run(realtime=False)
    return res, db, cycles


@pytest.fixture
def client(played):
    return TestClient(create_app(played[1]))


def test_playback_runs_cycles(played):
    _, db, cycles = played
    assert cycles == 18
    c = storage.connect(db)
    assert c.execute("SELECT COUNT(*) FROM screen_state").fetchone()[0] == 18


def test_p1_on_operator_screen_with_sound_and_decision(played, client):
    # Критерий приёмки: при проигрывании П1 появляется на экране оператора, звучит сигнал,
    # решение попадает в журнал.
    res, db, _ = played
    st = client.get("/api/state", params={"role": "operator"}).json()
    p1 = [e for e in st["events"] if e["code"] == "П1"]
    assert p1, "П1 не появилось на экране оператора"
    ep = res.truth["defects"]
    hyd = next(d for d in ep if d["defect"] == "hydrate_episode")
    assert pd.Timestamp(hyd["start"]) <= pd.Timestamp(p1[-1]["ts"]) <= pd.Timestamp(hyd["recovered"])
    assert st["sound_event_id"] == p1[-1]["id"] and p1[-1]["sound"]
    assert "уносы из абсорбера А-2" in p1[-1]["actions"]
    page = client.get("/operator").text
    assert 'src="/alarm.wav"' in page
    assert client.get("/alarm.wav").headers["content-type"] == "audio/wav"
    r = client.post("/api/decision", json={"event_id": p1[-1]["id"], "decision": "confirm", "reason_code": "hydrate",
                                           "user_name": "Иванов", "role": "operator", "comment": "подали метанол"})
    assert r.status_code == 200
    st2 = client.get("/api/state", params={"role": "operator"}).json()
    assert st2["sound_event_id"] is None
    e = next(x for x in st2["events"] if x["id"] == p1[-1]["id"])
    assert (e["decision"], e["reason_code"], e["decided_by"]) == ("confirm", "hydrate", "Иванов")
    c = storage.connect(db)
    assert c.execute("SELECT COUNT(*) FROM journal_decisions WHERE reason_code = 'hydrate'").fetchone()[0] == 1


def test_stale_flags_and_last_measurement_dates(client):
    snap = client.get("/api/state", params={"role": "operator"}).json()["snapshot"]
    fresh = {f["stream"]: f for f in snap["freshness"]}
    assert fresh["gathering"]["stale"] is True          # поток оборвался
    assert fresh["absorber"]["stale"] is False
    lm = snap["last_measurements"]
    assert lm["geotech_radar"] and set(lm["ecp"]) == {"ECP-C1", "ECP-C2", "ECP-D1", "ECP-P1"}
    assert lm["pipelines_survey"]
    assert all(o["color"] in ("green", "yellow", "red") for o in snap["geotech"]["objects"])
    st = client.get("/api/state", params={"role": "operator"}).json()
    assert any(e["code"] == "П13" and e["stream"] == "gathering" for e in st["events"])


def test_no_duplicate_journal_entries(played):
    _, db, _ = played
    c = storage.connect(db)
    rows = c.execute("SELECT kind, code, stream, object_id, ts_utc, payload_json FROM journal_events").fetchall()
    keys = Counter((r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows)
    assert max(keys.values()) == 1
    p1 = c.execute("SELECT COUNT(*) FROM journal_events WHERE code = 'П1'").fetchone()[0]
    assert 1 <= p1 <= 3


def test_roles_separate(client):
    op = client.get("/api/state", params={"role": "operator"}).json()
    ch = client.get("/api/state", params={"role": "chief_engineer"}).json()
    assert all(e["code"] in ("П1", "П13", "П14") for e in op["events"])
    assert all(e["code"] not in ("П1", "П13", "П14") for e in ch["events"])
    assert ch["asu_alarms"] == []
    assert client.get("/api/state", params={"role": "director"}).status_code == 400
    assert {g["title"] for g in ch["snapshot"]["groups"]["chief_engineer"]} >= {"ГТУ Т-32", "Газопроводы"}


def test_decision_validation(client, played):
    _, db, _ = played
    c = storage.connect(db)
    eid = c.execute("SELECT id FROM journal_events WHERE code = 'П13' LIMIT 1").fetchone()[0]
    r = client.post("/api/decision", json={"event_id": eid, "decision": "reject", "reason_code": "",
                                           "user_name": "Иванов", "role": "operator"})
    assert r.status_code == 400 and "обязательна" in r.json()["detail"]


def test_pages_have_no_internet_dependencies(client):
    for name in ("operator.html", "chief_engineer.html", "app.js", "style.css"):
        text = (WEB / name).read_text(encoding="utf-8")
        assert not re.search(r"https?://", text), name
    assert client.get("/chief").status_code == 200
    assert client.get("/web/app.js").status_code == 200
    assert client.get("/web/../pyproject.toml").status_code == 404


def test_realtime_playback_respects_speed(model, tmp_path):
    res = generate(tmp_path / "a", days=0.2, seed=2)
    pb = Playback(tmp_path / "rt.sqlite", [res.files["gathering"]], speed=60 * 60 * 20, cycle_minutes=60,
                  hydrate_model=False)
    import time
    t = time.monotonic()
    n = pb.run(realtime=True, max_cycles=2)
    assert n == 2 and time.monotonic() - t >= 0.1   # 1 ч данных за 0,05 с


def test_empty_state_before_first_cycle(tmp_path):
    db = tmp_path / "empty.sqlite"
    storage.connect(db)
    st = TestClient(create_app(db)).get("/api/state").json()
    assert st["snapshot"] is None and st["events"] == []
