"""Срок безопасной эксплуатации газопровода по частоте отказов (правило П9).

Источник: Харионовский В.В., Велиюлин И.И., Городниченко В.И. Прогнозирование сроков безопасной
эксплуатации газопроводов на основе анализа частоты отказов // Вести газовой науки. 2026.
№ 2 (66). С. 191–200. Апробация — 57 газопроводов ООО «Газпром трансгаз Югорск» (ВТД 2016–2017 гг.).

Формулы статьи:
- (1) t_сбэ = −ln(p_б) · 1000 / (λ_пр · L_ср), p_б = 0,9 (СТО Газпром 2-3.5-252-2008); L_ср —
  средняя протяжённость газопроводов сплошной выборки, км; λ_пр — отказов на 1000 км в год;
- (3) по коррозионному состоянию: λ_пр = λ_шнп + λ_крн + λ_км · D_км / D_скм;
- (7) для газопроводов с КРН: λ_пр = λ_шнп + λ_крн · D_крн / D_скрн + λ_км · D_км / D_скм;
- (4), (8) D_скм, D_скрн — средние по сплошной выборке; (5), (9) D = Σ d_max / L;
  (6), (10) d = ξ / ξ_р — повреждаемость трубы как отношение глубины дефекта к критической.
- Отношения повреждаемостей меньше 1 принимаются равными 1 (с. 194).

Расхождение внутри статьи: в табл. 2 и 3 отношение D_км / D_скм меньше 1 в расчёт входит как есть
(например, табл. 2, № 10: 0,843 → λ_пр = 0,1049), а текст требует округлять его до 1. Поэтому
нормирование задаётся отдельно для D_км и D_крн (clamp_km, clamp_krn); по умолчанию — как в тексте.

Критическая глубина ξ_р (по СТО Газпром 2-2.3-112-2007 и статье Ряховских) здесь не считается:
на вход подаются готовые повреждаемости труб d или повреждаемости газопровода D по данным ВТД.
Данные ВТД — не из АСУ ТП: значение срока вносится ручным вводом (М12) в поток pipelines.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from field_twin import config

P_FAILURE_FREE = 0.9            # p_б — вероятность безотказной работы (с. 191)


def settings(config_dir=None) -> dict[str, Any]:
    return config.load_yaml("models.yaml", config_dir)["safe_life"]


def safe_life_years(lambda_pred: float, l_avg_km: float, p_b: float = P_FAILURE_FREE) -> float:
    """Формула (1): срок безопасной эксплуатации, лет."""
    if lambda_pred <= 0 or l_avg_km <= 0:
        raise ValueError("Частота отказов и средняя протяжённость должны быть положительными")
    return -math.log(p_b) * 1000 / (lambda_pred * l_avg_km)


def pipe_damage(depth_rel: float, depth_rel_critical: float) -> float:
    """Формулы (6), (10): повреждаемость трубы d = ξ / ξ_р."""
    if depth_rel_critical <= 0:
        raise ValueError("Критическая относительная глубина должна быть положительной")
    return depth_rel / depth_rel_critical


def pipeline_damage(max_pipe_damages: Iterable[float], length_km: float) -> float:
    """Формулы (5), (9): повреждаемость 1 км газопровода D = Σ d_max(i) / L."""
    if length_km <= 0:
        raise ValueError("Протяжённость газопровода должна быть положительной")
    return sum(max_pipe_damages) / length_km


def sample_mean(values: Iterable[float]) -> float:
    """Формулы (4), (8): среднее по сплошной выборке."""
    v = list(values)
    if not v:
        raise ValueError("Сплошная выборка пуста")
    return sum(v) / len(v)


def _ratio(d: float, d_avg: float, clamp: bool) -> float:
    r = d / d_avg
    return max(r, 1.0) if clamp else r


def failure_rate(rates: dict[str, float], d_km: float, d_km_avg: float, d_krn: float | None = None,
                 d_krn_avg: float | None = None, clamp_km: bool = True, clamp_krn: bool = True) -> float:
    """Формулы (3) и (7): прогнозируемая частота отказов λ_пр на 1000 км в год.

    rates — λ_шнп, λ_крн, λ_км (ключи shnp, krn, km). Если заданы D_крн и D_скрн — формула (7)
    для газопроводов с КРН, иначе (3).
    """
    km = rates["km"] * _ratio(d_km, d_km_avg, clamp_km)
    if d_krn is not None and d_krn_avg:
        krn = rates["krn"] * _ratio(d_krn, d_krn_avg, clamp_krn)
    else:
        krn = rates["krn"]
    return rates["shnp"] + krn + km


@dataclass
class SafeLife:
    object_id: str
    lambda_pred: float
    years: float
    formula: str


def compute(s: dict[str, Any]) -> list[SafeLife]:
    """Срок по каждому газопроводу из раздела safe_life конфигурации."""
    rates, samp = s["failure_rates"], s["sample"]
    out = []
    for p in s.get("pipelines") or []:
        krn = p.get("d_krn") is not None
        d_km_avg = samp.get("d_km_avg_scc", samp["d_km_avg"]) if krn else samp["d_km_avg"]
        lam = failure_rate(rates, p["d_km"], d_km_avg, p.get("d_krn"), samp.get("d_krn_avg") if krn else None,
                           clamp_km=s.get("clamp_km", True), clamp_krn=s.get("clamp_krn", True))
        l_avg = samp["l_avg_km_scc"] if krn and samp.get("l_avg_km_scc") else samp["l_avg_km"]
        out.append(SafeLife(p["object_id"], lam, safe_life_years(lam, l_avg, s.get("p_b", P_FAILURE_FREE)),
                            "(7)" if krn else "(3)"))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Срок безопасной эксплуатации газопроводов (Харионовский и соавторы)")
    p.add_argument("--db", default=None, help="записать сроки ручным вводом в поток pipelines")
    p.add_argument("--user", default="safe_life", help="кто вносит значения")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    s = settings(args.config)
    res = compute(s)
    th = config.thresholds(args.config)["П9"]["min_years"]
    for r in res:
        flag = f"  < {th} лет — П9" if r.years < th else ""
        print(f"{r.object_id}: λ_пр = {r.lambda_pred:.4f}, t_сбэ = {r.years:.1f} лет (формула {r.formula}){flag}")
    if args.db:
        from field_twin import storage
        from field_twin.gateway.manual_input import ManualInput
        from field_twin.ingest import Ingestor
        mi = ManualInput(Ingestor(storage.connect(args.db)))
        now = datetime.now(timezone.utc)
        for r in res:
            mi.add("pipelines", r.object_id, "safe_life_years", round(r.years, 2), now, args.user)
        print(f"Записано значений: {len(res)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
