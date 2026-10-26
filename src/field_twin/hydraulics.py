"""Гидравлика газожидкостного потока в шлейфах: критическая скорость выноса жидкости и потери
давления с коэффициентами адаптации E и K.

Источник: Григорьев Б.А., Изюмченко Д.В., Гужов И.А., Федулов Д.М. Анализ необходимости изменения
требований к проведению гидравлических расчетов промысловых трубопроводов и минимально
допустимой скорости газа в шлейфах // Вести газовой науки. 2026. № 2 (66). С. 209–216.

Что взято из статьи:
- модифицированное критериальное уравнение (1) методики ООО «Газпром ВНИИГАЗ» (с. 212):
  V* = (121,36 − 0,25·μ̄^−0,6)·(Re₂·Fr_см)^(−1/3)·(ρ₂/(ρ₁ − ρ₂))^(−1/2)·β₁^0,115,
  μ̄ = μ₁/μ₂; β₁ = Q₁/Q_см; Re₂ = w_см·ρ₂·D/μ₂; Fr_см = w_см²/(g·D); w_см = 4·Q_см/(π·D²).
  V* ≤ 1 — кольцевой режим (жидкость выносится), V* > 1 — пробковый или расслоенный (накопление);
- область: давление 1–10 МПа и содержание воды 1–20 г/м³ (сравнение с TACITE, OLGAS), 0,5–8 МПа
  (промысловые данные), с. 213;
- вместо фиксированного диапазона 2…6 м/с — минимально допустимая (критическая) скорость для
  конкретных условий (с. 214): здесь — скорость смеси, при которой V* = 1;
- уравнение движения (2) с адаптационными коэффициентами (с. 214):
  −dP/dL = (ρ_см/ρ_ж)^K·ρ_ж·g·sinθ + (1/E²)·λ_см·ρ_см·w_см²/(2D);
  E определяется в режимах выноса жидкости (ΔP_тр ≫ ΔP_ст), K — в режимах накопления
  (ΔP_тр ≪ ΔP_ст); «фактические значения коэффициентов находятся из соотношения фактической
  и расчётной производительностей трубопровода при идентичных граничных условиях».

Что решено в прототипе:
- критическая скорость ищется при неизменном расходе жидкости Q₁ (β₁ меняется вместе с Q_см);
- K выражен из уравнения (2) при пренебрежимо малом трении — это алгебра, в статье формулы нет;
- формула истинного содержания жидкости φ₁ (с. 213) не реализована: в ней β₂, которое в статье
  не определено.
Свойства фаз (плотности, вязкости), расход жидкости и диаметр в АСУ обычно не измеряются —
они задаются из PVT-расчётов и паспорта шлейфа.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81

# Область применимости уравнения (1) по статье, МПа
PRESSURE_RANGE_MPA = (0.5, 8.0)


@dataclass
class Fluids:
    rho_liq: float      # ρ₁, кг/м³
    rho_gas: float      # ρ₂, кг/м³
    mu_liq: float       # μ₁, Па·с
    mu_gas: float       # μ₂, Па·с


def mixture_velocity(q_mix_m3s: float, d_m: float) -> float:
    """w_см = 4·Q_см / (π·D²), м/с."""
    return 4 * q_mix_m3s / (math.pi * d_m ** 2)


def v_star(f: Fluids, q_liq_m3s: float, q_mix_m3s: float, d_m: float) -> float:
    """Критериальное уравнение (1). q — объёмные расходы жидкости и смеси при рабочих условиях, м³/с."""
    if not (f.rho_liq > f.rho_gas > 0) or q_mix_m3s <= 0 or d_m <= 0 or q_liq_m3s <= 0:
        raise ValueError("Нужны ρ₁ > ρ₂ > 0 и положительные расходы и диаметр")
    w = mixture_velocity(q_mix_m3s, d_m)
    mu_bar = f.mu_liq / f.mu_gas
    beta1 = q_liq_m3s / q_mix_m3s
    re2 = w * f.rho_gas * d_m / f.mu_gas
    fr = w ** 2 / (G * d_m)
    return ((121.36 - 0.25 * mu_bar ** -0.6) * (re2 * fr) ** (-1 / 3)
            * (f.rho_gas / (f.rho_liq - f.rho_gas)) ** -0.5 * beta1 ** 0.115)


def regime(vs: float) -> str:
    """V* ≤ 1 — кольцевой (вынос жидкости); V* > 1 — пробковый или расслоенный (накопление)."""
    return "annular" if vs <= 1 else "slug_or_stratified"


def critical_velocity(f: Fluids, q_liq_m3s: float, d_m: float, w_max: float = 100.0) -> float:
    """Скорость смеси, при которой V* = 1 (переход к выносу жидкости), м/с, при заданном расходе жидкости.

    V* убывает с ростом скорости смеси, поэтому корень ищется делением отрезка.
    """
    area = math.pi * d_m ** 2 / 4

    def vs(w: float) -> float:
        return v_star(f, q_liq_m3s, max(w * area, q_liq_m3s * (1 + 1e-9)), d_m)

    lo, hi = q_liq_m3s / area * (1 + 1e-6), w_max
    if vs(hi) > 1:
        raise ValueError(f"V* > 1 даже при {w_max} м/с")
    if vs(lo) <= 1:
        return lo
    for _ in range(200):
        mid = (lo + hi) / 2
        if vs(mid) > 1:
            lo = mid
        else:
            hi = mid
    return hi


def pressure_gradient(rho_mix: float, rho_liq: float, theta_rad: float, lambda_mix: float, w_mix: float,
                      d_m: float, e: float = 1.0, k: float = 1.0) -> float:
    """Уравнение (2): −dP/dL, Па/м. E и K — адаптационные коэффициенты (1 — без адаптации)."""
    hydro = (rho_mix / rho_liq) ** k * rho_liq * G * math.sin(theta_rad)
    friction = lambda_mix * rho_mix * w_mix ** 2 / (2 * d_m) / e ** 2
    return hydro + friction


def fit_e(q_fact: float, q_calc: float) -> float:
    """E — отношение фактической и расчётной производительностей в режиме выноса жидкости (с. 214)."""
    if q_calc <= 0:
        raise ValueError("Расчётная производительность должна быть положительной")
    return q_fact / q_calc


def fit_k(grad_hydro_fact: float, rho_mix: float, rho_liq: float, theta_rad: float) -> float:
    """K из уравнения (2) в режиме накопления (трение пренебрежимо мало): решение прототипа."""
    base = rho_liq * G * math.sin(theta_rad)
    if base <= 0 or grad_hydro_fact <= 0 or not (0 < rho_mix < rho_liq):
        raise ValueError("Нужен подъёмный участок, положительный перепад и ρ_см < ρ_ж")
    return math.log(grad_hydro_fact / base) / math.log(rho_mix / rho_liq)


def in_validated_range(pressure_mpa: float) -> bool:
    """Давление в области, где уравнение (1) подтверждено промысловыми данными (0,5–8 МПа)."""
    return PRESSURE_RANGE_MPA[0] <= pressure_mpa <= PRESSURE_RANGE_MPA[1]
