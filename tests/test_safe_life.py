"""Срок безопасной эксплуатации (Харионовский, Велиюлин, Городниченко, 2026): проверка по табл. 1–3."""

import pytest

from field_twin import config, safe_life, storage
from field_twin.quality import load_measurements

RATES = {"shnp": 0.0433, "krn": 0.0433, "km": 0.0217}   # табл. 1


@pytest.mark.parametrize("d_ratio,lam,years", [
    (3.504, 0.1625, 6.9),     # табл. 2, № 1
    (1.786, 0.1253, 8.9),     # № 4
    (1.105, 0.1105, 10.1),    # № 9
])
def test_table2_corrosion(d_ratio, lam, years):
    got = safe_life.failure_rate(RATES, d_ratio, 1.0)
    assert got == pytest.approx(lam, abs=2e-4)   # округление в таблице
    assert safe_life.safe_life_years(got, 94.3) == pytest.approx(years, abs=0.05)


def test_table2_ratio_below_one_is_not_clamped_in_table():
    # Табл. 2, № 10: D_км / D_скм = 0,843 → λ_пр = 0,1049, t = 10,7 — в таблице без нормирования до 1.
    lam = safe_life.failure_rate(RATES, 0.843, 1.0, clamp_km=False)
    assert lam == pytest.approx(0.1049, abs=1e-4)
    assert safe_life.safe_life_years(lam, 94.3) == pytest.approx(10.7, abs=0.05)
    # По тексту статьи (с. 194) отношение < 1 принимается равным 1.
    assert safe_life.failure_rate(RATES, 0.843, 1.0) == pytest.approx(0.1083, abs=1e-4)


@pytest.mark.parametrize("km_ratio,krn_ratio,lam,years", [
    (3.245, 6.969, 0.4166, 2.5),    # табл. 3, № 1
    (0.964, 2.556, 0.1752, 6.0),    # № 2
    (1.401, 1.0, 0.1174, 8.9),      # № 12
])
def test_table3_scc(km_ratio, krn_ratio, lam, years):
    got = safe_life.failure_rate(RATES, km_ratio, 1.0, krn_ratio, 1.0, clamp_km=False)
    assert got == pytest.approx(lam, rel=0.01)
    assert safe_life.safe_life_years(got, 100.7) == pytest.approx(years, abs=0.1)


def test_damage_formulas():
    assert safe_life.pipe_damage(0.3, 0.6) == pytest.approx(0.5)
    assert safe_life.pipeline_damage([0.5, 0.7, 0.3], 10) == pytest.approx(0.15)
    assert safe_life.sample_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        safe_life.safe_life_years(0, 94.3)


def test_config_compute_and_p9_boundary():
    res = {r.object_id: r for r in safe_life.compute(safe_life.settings())}
    assert res["PL-03"].formula == "(7)" and res["PL-01"].formula == "(3)"
    assert res["PL-03"].years < config.thresholds()["П9"]["min_years"] < res["PL-01"].years


def test_cli_writes_manual_values(tmp_path, capsys):
    db = tmp_path / "t.sqlite"
    assert safe_life.main(["--db", str(db)]) == 0
    assert "П9" in capsys.readouterr().out
    m = load_measurements(storage.connect(db))
    rows = m[(m.stream == "pipelines") & (m.parameter == "safe_life_years")]
    assert set(rows.object_id) == {"PL-01", "PL-02", "PL-03"}
