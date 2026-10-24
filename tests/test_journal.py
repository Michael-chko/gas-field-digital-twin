import csv

import pytest

from field_twin.journal import Journal


@pytest.fixture
def journal(conn):
    return Journal(conn)


def test_reasons_loaded_from_config(journal):
    reasons = journal.reasons()
    assert {"hydrate", "downtime", "sensor_fault", "false_alarm"} <= set(reasons)


def test_decision_reason_saved_and_exported(journal, tmp_path):
    # Критерий приёмки: причина аномалии, указанная человеком, сохраняется и выгружается как разметка.
    ev = journal.log_event("notification", code="П1", message="Гидраты в абсорбере А-2",
                           ts="2026-07-01T10:20:00+05:00", stream="absorber", object_id="A-2")
    journal.record_decision(ev, "confirm", "hydrate", user_name="Иванов", role="operator",
                            comment="подали метанол в линию орошения")
    labels = journal.labels()
    assert len(labels) == 1
    lab = labels[0]
    assert (lab["code"], lab["decision"], lab["reason_code"], lab["reason_title"]) == (
        "П1", "confirm", "hydrate", "Гидратообразование")
    assert lab["event_ts_utc"] == "2026-07-01T05:20:00+00:00"
    out = tmp_path / "labels.csv"
    assert journal.export_labels(out) == 1
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert rows[0]["reason_code"] == "hydrate" and rows[0]["comment"] == "подали метанол в линию орошения"


def test_reason_required_and_from_dictionary(journal):
    ev = journal.log_event("rule", code="П13", message="Поток устарел")
    with pytest.raises(ValueError, match="обязательна"):
        journal.record_decision(ev, "reject", "", user_name="Иванов", role="operator")
    with pytest.raises(ValueError, match="справочник"):
        journal.record_decision(ev, "reject", "плохая погода", user_name="Иванов", role="operator")


@pytest.mark.parametrize("kwargs,msg", [
    ({"decision": "maybe"}, "Решение"),
    ({"role": "director"}, "Роль"),
    ({"user_name": ""}, "пользователь"),
    ({"event_id": 999}, "Нет события"),
])
def test_decision_validation(journal, kwargs, msg):
    ev = journal.log_event("model", code="М3")
    args = {"event_id": ev, "decision": "reject", "reason_code": "false_alarm", "user_name": "Петров", "role": "operator"}
    args.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        journal.record_decision(**args)


def test_event_validation(journal):
    with pytest.raises(ValueError):
        journal.log_event("gossip")
    with pytest.raises(ValueError):
        journal.log_event("rule", source="internet")
    with pytest.raises(ValueError, match="часового пояса"):
        journal.log_event("rule", ts="2026-07-01T10:00:00")


def test_shadow_events_and_sources_kept_apart(journal):
    journal.log_event("notification", code="П1", shadow=True)
    journal.log_event("asu_alarm", code="PAHH-101", source="asu")
    journal.log_event("rule", code="П13")
    assert len(journal.events(shadow=True)) == 1
    assert [e["source"] for e in journal.events(kind="asu_alarm")] == ["asu"]
    assert len(journal.events()) == 3


def test_period_labels(journal, tmp_path):
    journal.label_period("absorber", "2026-07-01T10:00:00+05:00", "2026-07-01T16:00:00+05:00",
                         "hydrate", user_name="Сидоров", object_id="A-2")
    with pytest.raises(ValueError, match="позже"):
        journal.label_period("absorber", "2026-07-01T16:00:00+05:00", "2026-07-01T10:00:00+05:00",
                             "hydrate", user_name="Сидоров")
    out = tmp_path / "periods.csv"
    assert journal.export_period_labels(out) == 1
    row = next(csv.DictReader(open(out, encoding="utf-8")))
    assert (row["object_id"], row["reason_code"], row["start_utc"]) == ("A-2", "hydrate", "2026-07-01T05:00:00+00:00")


def test_reason_sync_is_idempotent(journal):
    before = journal.reasons()
    journal.sync_reasons([{"code": "hydrate", "title": "Гидратообразование"}])
    assert journal.reasons() == before
