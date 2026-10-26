"""Этап 9: имитатор АСУ ТП (OPC UA, архив, сигнализация) и шлюз связки (М12)."""

import asyncio
import json
import socket
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import yaml
from asyncua import ua
from fastapi.testclient import TestClient

from asu_simulator.alarms import AlarmEngine
from asu_simulator.opcua_server import NAMESPACE, AsuSimulator, Sample
from field_twin import quality, storage
from field_twin.api import create_app
from field_twin.gateway.buffer import DeliveryBuffer
from field_twin.gateway.execution_check import check_execution
from field_twin.gateway.manual_input import ManualInput
from field_twin.gateway.mqtt_reader import MqttReader
from field_twin.gateway.opcua_reader import OpcUaReader
from field_twin.gateway.service import GatewayService
from field_twin.gateway.tag_dictionary import TagDictionary
from field_twin.ingest import Ingestor
from field_twin.journal import Journal

TAGS = TagDictionary.load()
T0 = datetime(2026, 7, 1, tzinfo=timezone.utc)
SIGNALS = ["absorber.A-1.Q_gas", "absorber.A-2.L_cube", "gathering.SEP-1.P_out", "methanol.M-2.Q_methanol"]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def url():
    return f"opc.tcp://127.0.0.1:{free_port()}/asu"


def samples(minutes, base=None, tags=SIGNALS):
    base = base or {"absorber.A-1.Q_gas": 450.0, "absorber.A-2.L_cube": 280.0, "gathering.SEP-1.P_out": 7.2,
                    "methanol.M-2.Q_methanol": 150.0}
    return [Sample(t, T0 + timedelta(minutes=m), base[t] + 0.01 * m) for m in minutes for t in tags]


def make_service(readers, db, tags=TAGS):
    conn = storage.connect(db)
    return GatewayService(readers, DeliveryBuffer(), Ingestor(conn, tags=tags), Journal(conn)), conn


def count(conn, origin="asu"):
    return conn.execute("SELECT COUNT(*) FROM measurements WHERE origin = ?", (origin,)).fetchone()[0]


# ---- только чтение ----

def test_adapters_have_no_write_methods():
    for cls in (OpcUaReader, MqttReader):
        public = [m for m in dir(cls) if not m.startswith("_")]
        assert not [m for m in public if any(w in m.lower() for w in ("write", "set_value", "publish", "send"))], cls


def test_twin_cannot_change_value_in_simulator():
    # Критерий приёмки: тест подтверждает, что двойник не может изменить значение в имитаторе.
    async def scenario():
        sim = AsuSimulator(url(), TAGS)
        await sim.start()
        await sim.publish(samples([0]))
        r = OpcUaReader(sim.url, TAGS, NAMESPACE)
        await r.connect()
        node = r.client.get_node(ua.NodeId("absorber.A-1.Q_gas", r.ns))
        with pytest.raises(ua.UaStatusCodeError):
            await node.write_value(ua.DataValue(ua.Variant(1.0, ua.VariantType.Double)))
        cur = await r.read_current(["absorber.A-1.Q_gas"])
        await r.disconnect()
        await sim.stop()
        return cur
    cur = asyncio.run(scenario())
    assert cur[0]["value"] == 450.0


# ---- словарь, привязка, уставки ----

def test_every_simulator_signal_bound_and_unbound_reported(tmp_path):
    # Критерии приёмки: каждый сигнал имитатора привязан к объекту двойника; сигнал без привязки
    # попадает в отчёт о приёме; сигнал без уставок попадает в отчёт как незаполненный.
    async def scenario():
        sim = AsuSimulator(url(), TAGS, extra_tags=["gathering.N-99.P"])
        await sim.start()
        await sim.publish(samples([0, 2]))
        svc, conn = make_service([OpcUaReader(sim.url, TAGS, NAMESPACE)], tmp_path / "b.sqlite")
        await svc.connect()
        res = await svc.poll()
        await svc.close()
        await sim.stop()
        return svc.binding, res
    binding, res = asyncio.run(scenario())
    assert binding.unbound_on_server == ["gathering.N-99.P"] and binding.missing_on_server == []
    for s in TAGS.signals.values():
        assert s.object_id and s.stream    # у каждого сигнала словаря есть объект двойника
    rep = res.report
    assert rep.unbound_source_signals == ["gathering.N-99.P"]
    assert "absorber.A-1.Q_gas" in rep.signals_without_setpoints
    assert "absorber.A-2.L_cube" not in rep.signals_without_setpoints     # уставки заданы
    text = rep.to_text()
    assert "без привязки к объекту двойника" in text and "уставки не заполнены" in text


def test_records_bound_to_twin_objects(tmp_path):
    async def scenario():
        sim = AsuSimulator(url(), TAGS)
        await sim.start()
        await sim.publish(samples([0]))
        svc, conn = make_service([OpcUaReader(sim.url, TAGS, NAMESPACE)], tmp_path / "r.sqlite")
        await svc.poll()
        await svc.close()
        await sim.stop()
        return conn
    conn = asyncio.run(scenario())
    rows = conn.execute("SELECT stream, object_id, parameter, value, quality, ingest_status FROM measurements "
                        "WHERE parameter = 'L_cube'").fetchall()
    assert [tuple(r) for r in rows] == [("absorber", "A-2", "L_cube", 280.0, "good", "ok")]


# ---- обрыв связи и буфер ----

def test_disconnect_backfill_without_loss_or_duplicates(tmp_path):
    # Критерий приёмки: при обрыве связи данные копятся и после восстановления досылаются без потерь и дублей.
    async def scenario():
        sim = AsuSimulator(url(), TAGS)
        await sim.start()
        svc, conn = make_service([OpcUaReader(sim.url, TAGS, NAMESPACE)], tmp_path / "d.sqlite")
        await sim.publish(samples([0, 2, 4]))
        await svc.poll()
        await svc.active.disconnect()                      # обрыв связи с имитатором
        await sim.publish(samples([6, 8, 10, 12]))          # АСУ продолжает работать
        res = await svc.poll()                              # связь восстановлена — досылка из архива
        await svc.poll()                                    # повторный опрос не даёт дублей
        await svc.close()
        await sim.stop()
        return conn, res
    conn, res = asyncio.run(scenario())
    assert res.backfilled > 0
    rows = conn.execute("SELECT stream, object_id, parameter, ts_utc FROM measurements WHERE origin = 'asu'").fetchall()
    keys = [tuple(r) for r in rows]
    assert len(keys) == len(set(keys)) == 7 * len(SIGNALS)   # все 7 моментов по каждому сигналу, без дублей


def test_central_link_failure_keeps_buffer(tmp_path):
    async def scenario():
        sim = AsuSimulator(url(), TAGS)
        await sim.start()
        svc, conn = make_service([OpcUaReader(sim.url, TAGS, NAMESPACE)], tmp_path / "c.sqlite")
        await sim.publish(samples([0]))
        svc.sink_ok = False                                 # нет связи с центром
        r1 = await svc.poll()
        n1, pend = count(conn), svc.buffer.pending()
        await sim.publish(samples([2]))
        await svc.poll()
        svc.sink_ok = True
        r3 = await svc.poll()
        await svc.close()
        await sim.stop()
        return r1, n1, pend, r3, count(conn), svc.buffer.pending()
    r1, n1, pend, r3, n_end, pend_end = asyncio.run(scenario())
    assert r1.delivery_error and n1 == 0 and pend == len(SIGNALS)
    assert n_end == 2 * len(SIGNALS) and pend_end == 0


def test_buffer_dedup():
    b = DeliveryBuffer()
    r = {"stream": "wells", "object_id": "W-101", "parameter": "Q_gas", "ts": T0, "value": 1.0}
    assert b.put([r, dict(r)]) == 1 and b.put([r]) == 0
    got = []
    assert b.flush(got.extend) == 1 and b.flush(got.extend) == 0 and len(got) == 1


# ---- резервирование ----

def test_failover_to_second_source(tmp_path):
    # Критерий приёмки: при отказе одного из двух независимых источников приём продолжается со второго.
    async def scenario():
        a, b = AsuSimulator(url(), TAGS), AsuSimulator(url(), TAGS)
        await a.start()
        await b.start()
        for sim in (a, b):
            await sim.publish(samples([0, 2]))
        svc, conn = make_service([OpcUaReader(a.url, TAGS, NAMESPACE, "primary"),
                                  OpcUaReader(b.url, TAGS, NAMESPACE, "secondary")], tmp_path / "f.sqlite")
        r1 = await svc.poll()
        await a.stop()                                      # отказ основного источника
        await b.publish(samples([4, 6]))
        r2 = await svc.poll()
        await svc.close()
        await b.stop()
        return r1, r2, conn
    r1, r2, conn = asyncio.run(scenario())
    assert r1.source == "primary" and r2.source == "secondary" and r2.failover
    ts = {r[0] for r in conn.execute("SELECT DISTINCT ts_utc FROM measurements WHERE origin = 'asu'")}
    assert len(ts) == 4                                    # 0, 2 с основного; 4, 6 — с резервного


# ---- другой сервер: только адрес и словарь ----

def test_other_server_needs_only_address_and_dictionary(tmp_path):
    # Критерий приёмки: переход на другой сервер OPC UA — смена адреса и словаря, без правки кода.
    raw = yaml.safe_load(open("config/tags.yaml", encoding="utf-8"))
    raw["streams"]["absorber"]["parameters"]["Q_gas"]["tags"] = {"A-1": "UKPG1V_FIC101_PV", "A-2": "UKPG1V_FIC201_PV"}
    raw["streams"]["absorber"]["parameters"]["L_cube"]["tags"] = {"A-1": "UKPG1V_LIC102_PV", "A-2": "UKPG1V_LIC202_PV"}
    path = tmp_path / "other_tags.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    other = TagDictionary.from_file(path)

    async def scenario():
        sim = AsuSimulator(url(), other, namespace="urn:other-vendor:scada")
        await sim.start()
        await sim.publish([Sample("UKPG1V_FIC201_PV", T0, 455.0), Sample("UKPG1V_LIC202_PV", T0, 47.0)])
        svc, conn = make_service([OpcUaReader(sim.url, other, "urn:other-vendor:scada")], tmp_path / "o.sqlite", other)
        await svc.poll()
        await svc.close()
        await sim.stop()
        return conn
    conn = asyncio.run(scenario())
    rows = {(r[0], r[1]): r[2] for r in conn.execute(
        "SELECT object_id, parameter, value FROM measurements WHERE stream = 'absorber' AND value != 0")}
    assert rows == {("A-2", "Q_gas"): 455.0, ("A-2", "L_cube"): 47.0}


# ---- сигнализация АСУ ----

def test_alarm_engine_levels():
    e = AlarmEngine({"x": {"alarm_low": 30, "warn_low": 40, "warn_high": 80, "alarm_high": 90}})
    assert [getattr(e.evaluate("x", v), "level", None) for v in (50, 39, 35, 29, 29, 50, 95)] == \
           [None, "L", None, "LL", None, "NORMAL", "HH"]
    assert e.evaluate("no-setpoints", 1) is None


def test_asu_alarms_separate_on_operator_screen(tmp_path):
    # Критерий приёмки: сигнализация АСУ и уведомления двойника хранятся раздельно и помечены источником.
    async def scenario():
        sim = AsuSimulator(url(), TAGS)
        await sim.start()
        await sim.publish(samples([0]))
        svc, conn = make_service([OpcUaReader(sim.url, TAGS, NAMESPACE)], tmp_path / "a.sqlite")
        await svc.connect()
        await svc.subscribe_alarms()
        await asyncio.sleep(0.3)
        await sim.publish([Sample("absorber.A-2.L_cube", T0 + timedelta(minutes=2), 190.0)])   # ниже LL = 200 мм
        for _ in range(30):
            if svc.alarms:
                break
            await asyncio.sleep(0.1)
        await svc.close()
        await sim.stop()
        return conn
    conn = asyncio.run(scenario())
    j = Journal(conn)
    j.log_event("notification", code="П1", message="Гидраты", object_id="A-2", payload={"recipient": "operator"})
    alarms = j.events(kind="asu_alarm")
    assert len(alarms) == 1 and alarms[0]["source"] == "asu" and "LL" in alarms[0]["message"]
    assert alarms[0]["object_id"] == "A-2"
    st = TestClient(create_app(tmp_path / "a.sqlite")).get("/api/state", params={"role": "operator"}).json()
    assert [a["source"] for a in st["asu_alarms"]] == ["asu"]
    assert all(e["source"] == "twin" for e in st["events"]) and any(e["code"] == "П1" for e in st["events"])
    page = (TestClient(create_app(tmp_path / "a.sqlite")).get("/web/app.js")).text
    assert "АСУ ТП" in page and "Двойник" in page          # метки источника на экране


# ---- ручной ввод ----

def test_manual_input_marked_and_checked(tmp_path):
    # Критерий приёмки: ручной ввод сохраняется с пометкой «ручной» и проходит те же проверки качества.
    conn = storage.connect(tmp_path / "m.sqlite")
    mi = ManualInput(Ingestor(conn))
    mi.add("ecp", "ECP-C1", "current", 4.8, T0, user_name="Сидоров")
    mi.add("wells", "W-101", "Q_gas", -5.0, T0, user_name="Сидоров")          # нефизичное значение
    rep = mi.add("ecp", "ECP-C1", "voltage", 20.0, T0.replace(tzinfo=None).replace(tzinfo=timezone.utc), "Сидоров")
    assert rep.rows_ok == 1
    with pytest.raises(ValueError):
        mi.add("ecp", "ECP-C1", "current", 4.8, T0.replace(tzinfo=None), user_name="Сидоров")
    rows = conn.execute("SELECT parameter, origin FROM measurements").fetchall()
    assert {r[1] for r in rows} == {"manual"}
    data = quality.QualityPipeline().run(quality.load_measurements(conn)).data
    q = data[(data.object_id == "W-101") & (data.parameter == "Q_gas")].iloc[0]
    assert q["f_nonphysical"] and q["origin"] == "manual"


# ---- проверка выполнения рекомендаций ----

def rec_event(j, proposed=165.0):
    eid = j.log_event("recommendation", code="П12", ts=T0, object_id="A-2", message="Увеличить метанол",
                      payload={"recipient": "chief_engineer", "changes": [
                          {"parameter": "methanol.M-2.Q_methanol", "current": 150.0, "proposed": proposed, "unit": "кг/ч"}]})
    j.record_decision(eid, "confirm", "hydrate", "Петров", "chief_engineer", ts=T0)
    return eid


@pytest.mark.parametrize("after,executed", [(164.0, True), (151.0, False)])
def test_execution_check_after_confirmation(tmp_path, after, executed):
    # Критерий приёмки: после подтверждения рекомендации двойник по данным АСУ отмечает, изменился ли режим.
    conn = storage.connect(tmp_path / "e.sqlite")
    j = Journal(conn)
    eid = rec_event(j)
    ing = Ingestor(conn)
    ing.ingest_records([{"stream": "methanol", "object_id": "M-2", "parameter": "Q_methanol",
                         "ts": T0 + timedelta(minutes=m), "value": after} for m in (10, 20)], source="asu", origin="asu")
    now = pd.Timestamp(T0 + timedelta(hours=3))
    out = check_execution(conn, j, now)
    assert len(out) == 1 and out[0]["executed"] == executed
    ev = j.events(kind="execution_check")
    assert len(ev) == 1 and json.loads(ev[0]["payload_json"])["event_id"] == eid
    assert ("выполнена" in ev[0]["message"]) and (("не выполнена" in ev[0]["message"]) != executed)
    assert check_execution(conn, j, now) == []            # второй раз не пишется


def test_execution_check_waits_for_window(tmp_path):
    conn = storage.connect(tmp_path / "w.sqlite")
    j = Journal(conn)
    rec_event(j)
    assert check_execution(conn, j, pd.Timestamp(T0 + timedelta(minutes=30))) == []


# ---- MQTT ----

@pytest.fixture(scope="module")
def broker():
    from amqtt.broker import Broker
    port = free_port()
    cfg = {"listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{port}"}}, "sys_interval": 0,
           "auth": {"allow-anonymous": True}, "topic-check": {"enabled": False}}
    loop = asyncio.new_event_loop()
    started = threading.Event()

    async def start():
        b = Broker(cfg)
        await b.start()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start())
        started.set()
        loop.run_forever()
    import warnings
    warnings.simplefilter("ignore", DeprecationWarning)
    threading.Thread(target=run, daemon=True).start()
    assert started.wait(10)
    yield port


def test_mqtt_remote_object_to_db_without_duplicates(broker, tmp_path):
    import paho.mqtt.client as mqtt
    conn = storage.connect(tmp_path / "q.sqlite")
    buf, ing = DeliveryBuffer(), Ingestor(conn)
    got = []
    reader = MqttReader("127.0.0.1", broker, TAGS, prefix="field")
    reader.start(got.append)
    time.sleep(0.5)
    pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    pub.connect("127.0.0.1", broker)
    pub.loop_start()
    msg = json.dumps({"v": 8.02, "q": "good", "t": "2026-07-01T05:00:00+05:00", "state": "run"})
    for _ in range(2):                                     # повторная доставка QoS 1
        pub.publish("field/wells/W-101/P_wellhead", msg, qos=1).wait_for_publish()
    pub.publish("field/wells/W-999/P_wellhead", msg, qos=1).wait_for_publish()
    for _ in range(50):
        if len(got) >= 2 and reader.unbound:
            break
        time.sleep(0.1)
    pub.loop_stop()
    reader.stop()
    assert len(got) == 2 and reader.unbound == {"field/wells/W-999/P_wellhead"}
    buf.put(got, origin="mqtt")
    buf.flush(lambda b: ing.ingest_records(b, source="mqtt", origin="asu"))
    rows = conn.execute("SELECT object_id, parameter, value, equipment_state FROM measurements").fetchall()
    assert [tuple(r) for r in rows] == [("W-101", "P_wellhead", 8.02, "run")]


def test_mqtt_rejects_bad_payload():
    r = MqttReader("127.0.0.1", 1, TAGS)
    assert r.parse("field/wells/W-101/P_wellhead", b'{"v": 1, "t": "2026-07-01T05:00:00"}') is None
    assert r.errors and "часового пояса" in r.errors[0]
