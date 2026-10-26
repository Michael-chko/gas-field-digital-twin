"""Гидравлика шлейфов (Григорьев и соавторы, 2026): уравнения (1) и (2)."""

import math

import pytest

from field_twin import hydraulics as h

# Свойства фаз — условные (вода и газ при ~5 МПа), в статье примера расчёта нет.
F = h.Fluids(rho_liq=1000.0, rho_gas=40.0, mu_liq=1e-3, mu_gas=1.3e-5)
D = 0.3
Q_LIQ = 1e-4      # м³/с


def test_v_star_by_formula():
    q_mix = 0.5
    w = 4 * q_mix / (math.pi * D ** 2)
    re2, fr = w * 40 * D / 1.3e-5, w ** 2 / (9.81 * D)
    expected = (121.36 - 0.25 * (1e-3 / 1.3e-5) ** -0.6) * (re2 * fr) ** (-1 / 3) * (40 / 960) ** -0.5 \
        * (Q_LIQ / q_mix) ** 0.115
    assert h.v_star(F, Q_LIQ, q_mix, D) == pytest.approx(expected)


def test_v_star_decreases_with_velocity_and_regime():
    vs = [h.v_star(F, Q_LIQ, q, D) for q in (0.05, 0.2, 0.8)]
    assert vs[0] > vs[1] > vs[2]
    assert h.regime(0.9) == "annular" and h.regime(1.1) == "slug_or_stratified"


def test_critical_velocity_gives_v_star_one():
    wc = h.critical_velocity(F, Q_LIQ, D)
    q = wc * math.pi * D ** 2 / 4
    assert h.v_star(F, Q_LIQ, q, D) == pytest.approx(1.0, abs=1e-6)
    # больше жидкости — выше критическая скорость (β₁ входит с положительной степенью)
    assert h.critical_velocity(F, 10 * Q_LIQ, D) > wc


def test_pressure_gradient_and_adaptation():
    theta = math.radians(2)
    g0 = h.pressure_gradient(200, 1000, theta, 0.02, 5, D)
    # E < 1 увеличивает потери на трение
    assert h.pressure_gradient(200, 1000, theta, 0.02, 5, D, e=0.9) > g0
    assert h.fit_e(90, 100) == pytest.approx(0.9)
    # K восстанавливается из гидростатической составляющей
    hydro = (200 / 1000) ** 1.3 * 1000 * 9.81 * math.sin(theta)
    assert h.fit_k(hydro, 200, 1000, theta) == pytest.approx(1.3)


def test_validated_range():
    assert h.in_validated_range(5) and not h.in_validated_range(9)
    with pytest.raises(ValueError):
        h.v_star(h.Fluids(40, 1000, 1e-3, 1e-5), Q_LIQ, 0.5, D)
