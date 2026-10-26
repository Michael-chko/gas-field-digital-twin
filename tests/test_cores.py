"""Этап 7: внешние расчётные ядра (М9), адаптер HYSYS, правило П12."""

import copy
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from field_twin import config, rules, storage
from field_twin.api import create_app
from field_twin.cores import base
from field_twin.cores.aerosim import AerosimCore
from field_twin.cores.hysys import FakeBackend, HysysCore
from field_twin.cores.stub import StubCore
from field_twin.cores.tnavigator import TNavigatorCore
from field_twin.cycle import CycleRunner, run_cycle

NOW = pd.Timestamp("2026-07-01T12:00:00Z")


def inp(**kw):
    m = {"absorber.A-1.Q_gas": 450.0, "absorber.A-2.Q_gas": 451.0, "absorber.A-2.L_cube": 280.0,
         "methanol.M-2.Q_methanol": 150.0}
    m.update({k.replace("__", "."): v for k, v in kw.items()})
    return base.CoreInput(NOW, m)


class OtherCore(base.CoreAdapter):
    """Другое ядро, подключаемое только через конфигурацию."""
    name = "other"

    def calculate(self, i):
        r = base.CoreResult(self.name, i.now, calculated={"x": 1.0})
        r.recommendations.append(base.Recommendation("other:1", "Совет другого ядра", "Сделать так", "A-1"))
        return r


class BrokenCore(base.CoreAdapter):
    def calculate(self, i):
        raise RuntimeError("ядро упало")


def test_registry_loads_stub_from_config():
    cores = base.load_cores(config.load_yaml("cores.yaml"))
    assert [c.name for c in cores] == ["stub"] and isinstance(cores[0], StubCore)
    all_cores = base.load_cores(config.load_yaml("cores.yaml"), only_enabled=False)
    assert {c.name for c in all_cores} == {"stub", "aerosim", "tnavigator", "hysys"}


def test_stub_replaceable_without_code_change():
    # Критерий приёмки: заглушку можно заменить другим адаптером без правки остального кода.
    cfg = {"cores": [{"name": "other", "class": "test_cores:OtherCore", "enabled": True}]}
    cores = base.load_cores(cfg)
    res = base.run_cores(cores, inp())
    ev = rules.check_p12(res, config.thresholds(), NOW)
    assert [e.code for e in ev] == ["П12"] and "Совет другого ядра" in ev[0].message


def test_stub_recommendations():
    r = StubCore({"imbalance_pct": 3, "level_drop_pct": 10}).calculate(inp(absorber__A__1__Q_gas=0))
    assert r.recommendations == []
    r = StubCore({"imbalance_pct": 3}).calculate(inp(**{"absorber__A-1__Q_gas": 480.0}))
    assert [x.key for x in r.recommendations] == ["redistribute:A-1->A-2"]
    assert r.recommendations[0].changes[0].proposed == pytest.approx((480 + 451) / 2)
    r = StubCore({}).calculate(inp(**{"absorber__A-2__L_cube": 240.0}))
    assert [x.key for x in r.recommendations] == ["methanol:A-2"]


def test_aerosim_tnavigator_are_described_stubs():
    for cls in (AerosimCore, TNavigatorCore):
        core = cls()
        assert not core.is_available()
        with pytest.raises(NotImplementedError, match="Открытые вопросы"):
            core.calculate(inp())
        with pytest.raises(NotImplementedError):
            core.connect()
    res = base.run_cores([AerosimCore(), BrokenCore()], inp())
    assert [r.status for r in res] == ["not_connected", "error"]
    assert "ядро упало" in res[1].error


def test_p12_on_chief_screen_and_decision(ingested_db, tmp_path):
    # Критерий приёмки: рекомендация заглушки появляется на экране главного инженера через П12.
    db = tmp_path / "c.sqlite"
    shutil.copy(ingested_db, db)
    conn = storage.connect(db)
    runner = CycleRunner(conn, hydrate_model=False, chudin_model=False)
    runner.cores = base.load_cores({"cores": [{"name": "other", "class": "test_cores:OtherCore"}]})
    runner.step()
    runner.step()      # повторный цикл не дублирует рекомендацию
    c = TestClient(create_app(db))
    st = c.get("/api/state", params={"role": "chief_engineer"}).json()
    recs = [e for e in st["events"] if e["code"] == "П12"]
    assert len(recs) == 1 and recs[0]["kind"] == "recommendation" and "Совет другого ядра" in recs[0]["message"]
    assert not [e for e in c.get("/api/state", params={"role": "operator"}).json()["events"] if e["code"] == "П12"]
    r = c.post("/api/decision", json={"event_id": recs[0]["id"], "decision": "confirm", "reason_code": "other",
                                      "user_name": "Петров", "role": "chief_engineer", "comment": "введено в АСУ"})
    assert r.status_code == 200


def test_cycle_survives_broken_core(ingested_db, tmp_path):
    db = tmp_path / "b.sqlite"
    shutil.copy(ingested_db, db)
    res = run_cycle(storage.connect(db), cores=[BrokenCore()], hydrate_model=False, chudin_model=False, log=False)
    assert res.core_results[0].status == "error"


# ---- HYSYS ----

def plant_model(v):
    """Заменитель модели завода: СПГ ~ расход газа, мощность компрессоров ~ производительность."""
    feed = v[("MaterialStreams", "Feed", "StdGasFlow")]
    lng = feed * 0.00072 * 0.9          # м³/ч → т/ч, 90 % на сжижение
    out = {("MaterialStreams", "LNG", "MassFlow"): lng, ("MaterialStreams", "NH3", "MassFlow"): 20.0,
           ("MaterialStreams", "Urea", "MassFlow"): 35.0, ("MaterialStreams", "ПГ на сжижение", "MassFlow"): lng * 1.02}
    for k in ("100", "101", "102", "104", "105", "106", "107"):
        out[("EnergyStreams", f"Q-{k}", "HeatFlow")] = lng * 50.0
    return out


def hysys_settings(**rec):
    s = copy.deepcopy(config.load_yaml("hysys.yaml"))
    s["recommendations"]["specific_energy"].update(rec)
    return s


def test_hysys_sets_inputs_solves_and_reads(tmp_path):
    fb = FakeBackend(plant_model)
    core = HysysCore(backend=fb, settings=hysys_settings())
    i = base.CoreInput(NOW, {"gathering.SEP-1.Q_sep": 300.0, "gathering.SEP-1.P_out": 7.2,
                             "absorber.A-1.T_gas_absorber": -25.0})
    r = core.calculate(i)
    assert fb.opened == "C:/models/plant.hsc" and fb.solved == 1
    assert fb.values[("MaterialStreams", "Feed", "StdGasFlow")] == 300000.0      # тыс. м³/ч → м³/ч
    assert fb.values[("MaterialStreams", "Feed", "Pressure")] == pytest.approx(7200.0)
    assert r.calculated["lng_t_h"] == pytest.approx(300000 * 0.00072 * 0.9)
    assert r.calculated["specific_energy_kwh_t"] == pytest.approx(7 * 50.0)
    assert r.recommendations == []     # базовое значение не задано


@pytest.mark.parametrize("base_kwh,fires", [(345.0, False), (300.0, True)])
def test_hysys_specific_energy_recommendation(base_kwh, fires):
    core = HysysCore(backend=FakeBackend(plant_model), settings=hysys_settings(baseline_kwh_t=base_kwh, max_increase_pct=10))
    r = core.calculate(base.CoreInput(NOW, {"gathering.SEP-1.Q_sep": 300.0}))
    assert bool(r.recommendations) == fires
    if fires:
        rec = r.recommendations[0]
        assert "компрессорами хладагента" in rec.message and rec.key == "hysys:specific_energy"
        ev = rules.check_p12([r], config.thresholds(), NOW)
        assert ev[0].recipient == "chief_engineer"


def test_hysys_missing_inputs_counted():
    r = HysysCore(backend=FakeBackend(plant_model), settings=hysys_settings()).calculate(
        base.CoreInput(NOW, {"gathering.SEP-1.Q_sep": 300.0}))
    assert r.calculated["missing_inputs"] == 2


def test_hysys_com_backend_explains_platform():
    core = HysysCore(settings=hysys_settings())
    res = base.run_cores([core], base.CoreInput(NOW, {}))
    assert res[0].status == "error" and ("Windows" in res[0].error or "HYSYS" in res[0].error)
