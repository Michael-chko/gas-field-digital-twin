"""Этап 6: нейросети Чудина (М4), адаптация, связывающий алгоритм, правило П15."""

import copy

import numpy as np
import pandas as pd
import pytest

from field_twin import chudin_models as cm
from field_twin import config, rules
from field_twin.simulator import chudin_daily

S = cm.settings()


@pytest.fixture(scope="module")
def data():
    d = chudin_daily(300, seed=1)
    return cm.daily_table(d, "dks"), cm.daily_table(d, "cluster")


@pytest.fixture(scope="module")
def models(data):
    return cm.train(*data, S)


def test_architecture_from_source(models):
    assert len(models.dks.net.hidden_layer_sizes) == 5           # 5 скрытых слоёв (Чудин)
    assert tuple(models.transport.net.hidden_layer_sizes) == (4,)  # 1 слой из 4 нейронов (Чудин)
    assert models.dks.net.solver == models.transport.net.solver == "adam"
    assert models.dks.inputs == ["DKS-1.P_in", "DKS-1.P_out", "DKS-1.Q"]
    assert models.dks.outputs == ["DKS-1.N_power", "DKS-1.n_speed"]
    assert len(models.transport.inputs) == 10 and len(models.transport.outputs) == 4


def test_split_60_40_and_mean_relative_error(models, data):
    # Критерий приёмки: деление 60 / 40 %, вывод средней относительной ошибки.
    for m in (models.dks, models.transport):
        assert m.train_rows == int(300 * 0.6) and m.test_rows == 300 - int(300 * 0.6)
        assert set(m.test_error_pct) == set(m.outputs)
    assert models.dks.test_error_pct["DKS-1.N_power"] < 5
    assert max(models.transport.test_error_pct.values()) < 5
    rep = models.report()
    assert "средняя относительная ошибка" in rep and "ориентир Чудина 1.9 %" in rep


def test_forecast_74_days(models, data):
    _, ct = data
    last = ct.iloc[-74:]
    err = cm.mean_relative_error_pct(last[models.transport.outputs], models.transport.predict(last))
    assert err.max() < 5      # ориентир Чудина — не более 2 %


def test_retrain_on_new_data(data):
    dt, _ = data
    m = cm.train(dt, data[1], S).dks
    shifted = dt.copy()
    shifted["DKS-1.N_power"] *= 1.08            # оборудование изменилось: мощность на 8 % выше
    before = cm.mean_relative_error_pct(shifted[m.outputs].iloc[-60:], m.predict(shifted.iloc[-60:]))[0]
    m.retrain(shifted.iloc[:-60], epochs=150)
    after = cm.mean_relative_error_pct(shifted[m.outputs].iloc[-60:], m.predict(shifted.iloc[-60:]))[0]
    assert after < before / 2


def test_adaptation_fixed_during_forecast(data):
    # Критерий приёмки: после адаптации на базовом периоде поправочные коэффициенты в прогнозе не меняются.
    dt, ct = data
    m = cm.train(dt, ct, S).dks
    biased = dt.copy()
    biased[m.outputs] *= 1.05                   # замеры на 5 % выше расчёта
    base, forecast = biased.iloc[:30], biased.iloc[30:]
    raw_err = cm.mean_relative_error_pct(base[m.outputs], m.predict(base))
    k = m.adapt(base)
    assert k == pytest.approx([1.05, 1.05], rel=0.02)
    adapted_err = cm.mean_relative_error_pct(base[m.outputs], m.predict(base))
    assert adapted_err.max() < raw_err.min()
    m.predict(forecast)
    m.predict(forecast * 1.2)
    np.testing.assert_array_equal(m.corrections, k)


def test_linked_algorithm_exchanges_data(models, data):
    dt, ct = data
    out = cm.run_linked(models, ct.iloc[-10:], dt.iloc[-10:])
    assert len(out) == 10
    wells = out[[c for c in out.columns if c.endswith(".Q_well")]].sum(axis=1)
    np.testing.assert_allclose(out["dks_Q_from_transport"], wells * S["dks_flow_factor"] / 24)
    # ДКС посчитана на расходе из звена транспорта, а не на замере
    x = pd.DataFrame({"DKS-1.P_in": dt["DKS-1.P_in"].iloc[-10:], "DKS-1.P_out": dt["DKS-1.P_out"].iloc[-10:],
                      "DKS-1.Q": out["dks_Q_from_transport"].to_numpy()}, index=out.index)
    np.testing.assert_allclose(out["DKS-1.N_power"], models.dks.predict(x)["DKS-1.N_power"])


def test_save_load(models, tmp_path):
    p = tmp_path / "m.joblib"
    models.save(p)
    m2 = cm.ChudinModels.load(p)
    x = cm.daily_table(chudin_daily(20, seed=3), "dks")
    pd.testing.assert_frame_equal(models.dks.predict(x), m2.dks.predict(x))


# ---- П15 ----

def p15(errors_pct, threshold):
    th = copy.deepcopy(config.thresholds())
    th["П15"]["max_error_pct"] = threshold
    errs = [{"model": "ДКС", "output": "DKS-1.N_power", "error_pct": e, "days": 7, "end": None} for e in errors_pct]
    return rules.check_p15(errs, th, pd.Timestamp("2026-07-01", tz="UTC"))


def test_p15_below_above():
    assert not p15([4.9], 5)
    ev = p15([5.1], 5)
    assert len(ev) == 1 and ev[0].recipient == "chief_engineer" and "перекалибровать" in ev[0].message


def test_p15_disabled_without_threshold():
    assert config.thresholds()["П15"]["max_error_pct"] is None
    assert p15([99], None) == []


def test_model_errors_detect_drift(models, data):
    dt, ct = data
    drift = dt.copy()
    drift.loc[drift.index[-7:], "DKS-1.N_power"] *= 1.3     # датчик мощности «уплыл»
    errs = {e["output"]: e["error_pct"] for e in cm.model_errors(models, drift, ct)}
    assert errs["DKS-1.N_power"] > 20 and errs["DKS-1.n_speed"] < 5


def test_cycle_runs_models_on_db(models, ingested_db, tmp_path):
    import shutil
    from field_twin import storage
    from field_twin.cycle import run_cycle
    db = tmp_path / "c.sqlite"
    shutil.copy(ingested_db, db)
    th = copy.deepcopy(config.thresholds())
    res = run_cycle(storage.connect(db), chudin_model=models, log=False, save=False)
    assert {e["output"] for e in res.model_errors} >= {"DKS-1.N_power", "W-101.Q_well"}
