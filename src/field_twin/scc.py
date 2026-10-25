"""Модель стресс-коррозионной повреждённости (КРН) участков газопроводов.

Источник: Мирзоев А.М., Иващенко М.С., Маршаков А.И. Нейросетевая модель стресс-коррозионной
повреждённости участков линейной части магистральных газопроводов // Вести газовой науки. 2016.
№ 3 (27). С. 108–112. Данные ООО «Газпром трансгаз Югорск», МГ Пунга — Вуктыл — Ухта I.

Что взято из статьи:
- 9 факторов КРН (табл. 1): 7 по грунту, расстояние от компрессорной станции, результаты
  геотехнической диагностики (водотоки, непроектное положение, грунтовые воды);
- многослойный перцептрон: 10 входов, 1 скрытый слой из 13 нейронов с сигмоидой, 2 выхода
  с линейной активацией; выход — наличие или отсутствие КРН глубиной 0,3 мм — 1,0t;
- выборка 368 участков, тестовая и перекрёстная по 55 примеров, деление случайное;
- матрица ошибок с долей верных по классам (табл. 2) — ориентиры 80,43 / 81,52 / 80,98 %;
- классы риска прототипа программы (рис. 4);
- применимость на другом участке — шурфы (около 10), модель адекватна при совпадении ≥ 75 %.

Что решено в прототипе и чего в статье нет:
- 10 входов на 9 факторов: последний фактор разложен на два входа (водотоки; непроектное
  положение / грунтовые воды);
- данных статьи нет — модель обучается на СИНТЕТИКЕ. Направления влияния факторов в синтетике
  выбраны по общим представлениям о КРН (низкое сопротивление грунта, карбонаты, наводороживание,
  близость к КС, обводнённость повышают риск), а не взяты из статьи. Метрики на синтетике
  проверяют только работу кода. Каждый прогноз модели, обученной на синтетике, помечен;
- статья — о МАГИСТРАЛЬНЫХ газопроводах. Для ПРОМЫСЛОВЫХ (другие диаметры и давления, влажный
  газ, расстояние «от КС» теряет смысл) модель вне области применимости: прогноз помечается
  «перенос не проверен», пока шурфы на этом газопроводе его не подтвердят;
- применимость проверяется по каждому газопроводу отдельно; до 10 шурфов прогноз
  предварительный, при совпадении < 75 % модель для газопровода признаётся неприменимой и
  уведомлений по нему не даёт.

Входы обновляются по графику обследований (ручной ввод М12), а не из АСУ ТП в реальном времени.
"""

from __future__ import annotations

import argparse
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from field_twin import config
from field_twin.config import ROOT
from field_twin.gateway.tag_dictionary import TagDictionary

SYNTHETIC = "synthetic"


def settings(config_dir=None) -> dict[str, Any]:
    return config.load_yaml("models.yaml", config_dir)["scc"]


# ---------- синтетика ----------

def synthetic_factors(n: int, rng: np.random.Generator, pipeline_class: str = "trunk") -> pd.DataFrame:
    """Факторы КРН. Распределения — синтетика; для промыслового газопровода расстояние меряется
    от ДКС/УКПГ и короче, грунты те же."""
    dist = rng.uniform(0, 120_000, n) if pipeline_class == "trunk" else rng.uniform(0, 15_000, n)
    return pd.DataFrame({
        "soil_resistivity": np.exp(rng.normal(np.log(50), 0.9, n)),
        "soil_ph": np.clip(rng.normal(6.8, 0.8, n), 4, 9.5),
        "sulfides": np.exp(rng.normal(np.log(1e-3), 0.8, n)),
        "carbonates": np.exp(rng.normal(np.log(1e-2), 0.8, n)),
        "hydrogen_permeation": np.exp(rng.normal(np.log(5), 0.6, n)),
        "o2_limit_current": np.exp(rng.normal(np.log(30), 0.5, n)),
        "anodic_current": np.exp(rng.normal(np.log(10), 0.6, n)),
        "distance_from_cs": dist,
        "watercourse": (rng.random(n) < 0.25).astype(float),
        "off_design_groundwater": (rng.random(n) < 0.3).astype(float),
    })


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def synthetic_scc_latent(f: pd.DataFrame, pipeline_class: str = "trunk") -> np.ndarray:
    """Скрытая склонность к КРН — синтетика. Направления по общим представлениям о КРН, НЕ из статьи.
    Для промыслового газопровода связь другая (влияние расстояния нет, иные веса) — чтобы
    проверка применимости по шурфам показывала, что перенос не очевиден."""
    if pipeline_class == "trunk":
        return (-1.0 * _z(np.log(f["soil_resistivity"])) - 0.5 * np.abs(f["soil_ph"] - 6.5)
                + 0.7 * _z(np.log(f["carbonates"])) + 0.4 * _z(np.log(f["sulfides"]))
                + 0.9 * _z(np.log(f["hydrogen_permeation"])) + 0.3 * _z(np.log(f["o2_limit_current"]))
                + 0.5 * _z(np.log(f["anodic_current"])) - 0.8 * f["distance_from_cs"] / 50_000
                + 0.8 * f["watercourse"] + 0.6 * f["off_design_groundwater"]).to_numpy()
    return (0.9 * _z(np.log(f["sulfides"])) - 0.3 * _z(np.log(f["soil_resistivity"]))
            + 1.2 * f["off_design_groundwater"] - 0.2 * _z(np.log(f["hydrogen_permeation"]))).to_numpy()


def synthetic_dataset(n: int = 368, seed: int = 0, pipeline_class: str = "trunk", noise: float = 0.8,
                      balanced: bool = True) -> pd.DataFrame:
    """Синтетическая выборка «факторы → КРН есть/нет». balanced — поровну участков с КРН и без
    (в статье 184 и 184)."""
    rng = np.random.default_rng(seed)
    f = synthetic_factors(n, rng, pipeline_class)
    latent = synthetic_scc_latent(f, pipeline_class) + rng.normal(0, noise, n)
    cut = np.median(latent) if balanced else np.quantile(latent, 0.7)
    f["scc"] = (latent > cut).astype(int)
    return f


# ---------- модель ----------

@dataclass
class SccMetrics:
    no_scc_total: int
    no_scc_correct: int
    scc_total: int
    scc_correct: int

    @property
    def no_scc_pct(self):
        return 100 * self.no_scc_correct / self.no_scc_total if self.no_scc_total else 0.0

    @property
    def scc_pct(self):
        return 100 * self.scc_correct / self.scc_total if self.scc_total else 0.0

    @property
    def total_pct(self):
        n = self.no_scc_total + self.scc_total
        return 100 * (self.no_scc_correct + self.scc_correct) / n if n else 0.0

    def to_text(self, reference: dict[str, float] | None = None) -> str:
        ref = reference or {}
        return "\n".join([
            "Матрица ошибок (как табл. 2 статьи):",
            f"  участков без КРН: {self.no_scc_total}, верно {self.no_scc_correct} ({self.no_scc_pct:.2f} %)"
            + (f" — статья {ref['no_scc_pct']} %" if "no_scc_pct" in ref else ""),
            f"  участков с КРН:   {self.scc_total}, верно {self.scc_correct} ({self.scc_pct:.2f} %)"
            + (f" — статья {ref['scc_pct']} %" if "scc_pct" in ref else ""),
            f"  всего верно: {self.total_pct:.2f} %" + (f" — статья {ref['total_pct']} %" if "total_pct" in ref else ""),
        ])


@dataclass
class SccModel:
    net: MLPRegressor
    scaler: StandardScaler
    inputs: list[str]
    log_inputs: list[str]
    trained_on: str                          # synthetic | real
    domain_class: str                        # trunk — область исходной модели
    metrics: SccMetrics | None = None
    validation_metrics: SccMetrics | None = None
    trained_at: str = ""
    settings: dict[str, Any] = field(default_factory=dict)

    def _x(self, f: pd.DataFrame) -> np.ndarray:
        x = f[self.inputs].astype(float).copy()
        for c in self.log_inputs:
            x[c] = np.log(np.maximum(x[c], 1e-12))
        return self.scaler.transform(x.to_numpy())

    def probability(self, f: pd.DataFrame) -> np.ndarray:
        """Вероятность КРН по двум линейным выходам сети: доля выхода «КРН есть»."""
        out = np.clip(np.asarray(self.net.predict(self._x(f))).reshape(len(f), 2), 0, None)
        s = out.sum(axis=1)
        return np.where(s > 0, out[:, 1] / np.where(s > 0, s, 1), 0.5)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "SccModel":
        return joblib.load(path)


def _metrics(y, p) -> SccMetrics:
    y, pred = np.asarray(y).astype(int), (np.asarray(p) >= 0.5).astype(int)
    return SccMetrics(int((y == 0).sum()), int(((y == 0) & (pred == 0)).sum()),
                      int((y == 1).sum()), int(((y == 1) & (pred == 1)).sum()))


def train(data: pd.DataFrame, s: dict[str, Any], trained_on: str) -> SccModel:
    """Обучение по схеме статьи: случайное деление, тестовая и перекрёстная выборки по 55 примеров."""
    if trained_on not in (SYNTHETIC, "real"):
        raise ValueError("trained_on: synthetic или real")
    rng = np.random.default_rng(s.get("seed", 0))
    idx = rng.permutation(len(data))
    n_te, n_va = s["test_examples"], s["validation_examples"]
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
    x = data[s["inputs"]].astype(float).copy()
    for c in s["log_inputs"]:
        x[c] = np.log(np.maximum(x[c], 1e-12))
    scaler = StandardScaler().fit(x.iloc[tr].to_numpy())
    y = data["scc"].to_numpy().astype(int)
    onehot = np.stack([1 - y, y], axis=1).astype(float)
    best = None
    # Перекрёстная выборка — для выбора лучшего из нескольких запусков (в статье число нейронов
    # подбиралось по минимуму ошибки; здесь архитектура фиксирована, перебираются начальные веса).
    for seed in range(5):
        net = MLPRegressor(hidden_layer_sizes=tuple(s["hidden_layers"]), activation=s["activation"],
                           solver=s.get("solver", "lbfgs"), max_iter=2000, alpha=s.get("alpha", 1e-3), random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.fit(scaler.transform(x.iloc[tr].to_numpy()), onehot[tr])
        m = SccModel(net, scaler, s["inputs"], s["log_inputs"], trained_on, s.get("domain_class", "trunk"),
                     settings=s)
        vm = _metrics(y[va], m.probability(data.iloc[va]))
        if best is None or vm.total_pct > best.validation_metrics.total_pct:
            m.validation_metrics = vm
            best = m
    best.metrics = _metrics(y[te], best.probability(data.iloc[te]))
    best.trained_at = datetime.now(timezone.utc).isoformat()
    return best


def risk_class(p: float, classes: list[dict[str, Any]]) -> str:
    name = classes[0]["name"]
    for c in classes:
        if p >= c["from"]:
            name = c["name"]
    return name


# ---------- шурфы и применимость ----------

def add_pit(conn: sqlite3.Connection, tags: TagDictionary, segment_id: str, scc_found: bool, user_name: str,
            dug_at: datetime, depth_mm: float | None = None, predicted_probability: float | None = None,
            comment: str | None = None) -> int:
    sig = tags.get("pipeline_segments", segment_id, "soil_ph")
    if sig is None:
        raise ValueError(f"Участка {segment_id} нет в словаре параметров")
    if not user_name:
        raise ValueError("Не указан, кто вносит результат шурфования")
    if dug_at.tzinfo is None:
        raise ValueError("Дата шурфования — с часовым поясом")
    cur = conn.execute(
        """INSERT INTO scc_pits(segment_id, pipeline, dug_at, scc_found, depth_mm, predicted_probability, user_name, comment)
           VALUES (?,?,?,?,?,?,?,?)""",
        (segment_id, sig.object_attrs["pipeline"], dug_at.isoformat(), int(scc_found), depth_mm,
         predicted_probability, user_name, comment))
    conn.commit()
    return int(cur.lastrowid)


@dataclass
class Applicability:
    pipeline: str
    pipeline_class: str
    pits: int
    matches: int
    status: str        # unverified | verified | rejected

    @property
    def match_share(self) -> float | None:
        return self.matches / self.pits if self.pits else None


def applicability(conn: sqlite3.Connection, s: dict[str, Any], tags: TagDictionary) -> dict[str, Applicability]:
    """Совпадение прогноза (на момент шурфования) с шурфами по каждому газопроводу (статья: ≥ 75 % при ~10 шурфах)."""
    rule = s["applicability"]
    out = {}
    pipes = {}
    for sig in tags.stream_signals("pipeline_segments"):
        pipes[sig.object_attrs["pipeline"]] = sig.object_attrs.get("pipeline_class", "field")
    for pipe, cls in pipes.items():
        rows = conn.execute("SELECT scc_found, predicted_probability FROM scc_pits WHERE pipeline = ? "
                            "AND predicted_probability IS NOT NULL", (pipe,)).fetchall()
        matches = sum(int((p >= 0.5) == bool(f)) for f, p in rows)
        if len(rows) < rule["min_pits"]:
            status = "unverified"
        else:
            status = "verified" if matches / len(rows) >= rule["min_match"] else "rejected"
        out[pipe] = Applicability(pipe, cls, len(rows), matches, status)
    return out


# ---------- прогноз по участкам двойника и правило П17 ----------

@dataclass
class SegmentRisk:
    segment_id: str
    pipeline: str
    pipeline_class: str
    probability: float
    risk_class: str
    measured_at: pd.Timestamp
    in_domain: bool                      # магистральный — область исходной модели
    applicability: str
    synthetic_model: bool

    def caveats(self, min_pits: int) -> list[str]:
        out = []
        if self.synthetic_model:
            out.append("модель обучена на синтетических данных — прогноз не для решений")
        if not self.in_domain:
            out.append("промысловый газопровод вне области исходной модели (статья — о магистральных)")
        if self.applicability == "unverified":
            out.append(f"применимость на газопроводе не подтверждена (нужно не менее {min_pits} шурфов, совпадение ≥ 75 %)")
        return out


def segment_risks(data: pd.DataFrame, model: SccModel, tags: TagDictionary, s: dict[str, Any],
                  appl: dict[str, Applicability]) -> list[SegmentRisk]:
    seg = data[(data["stream"] == s["stream"]) & data["value_clean"].notna()]
    if seg.empty:
        return []
    last = seg.sort_values("ts").groupby(["object_id", "parameter"]).tail(1)
    wide = last.pivot_table(index="object_id", columns="parameter", values="value_clean", aggfunc="last")
    when = last.groupby("object_id")["ts"].max()
    wide = wide.dropna(subset=[c for c in s["inputs"] if c in wide.columns])
    if wide.empty or any(c not in wide.columns for c in s["inputs"]):
        return []
    probs = model.probability(wide)
    out = []
    for (obj, p) in zip(wide.index, probs):
        sig = tags.get(s["stream"], obj, "soil_ph")
        attrs = sig.object_attrs if sig else {}
        pipe, cls = attrs.get("pipeline", "?"), attrs.get("pipeline_class", "field")
        a = appl.get(pipe)
        out.append(SegmentRisk(obj, pipe, cls, float(p), risk_class(float(p), s["risk_classes"]), when[obj],
                               cls == model.domain_class, a.status if a else "unverified",
                               model.trained_on == SYNTHETIC))
    return out


def check_p17(risks: list[SegmentRisk], th: dict[str, Any], s: dict[str, Any], now: pd.Timestamp):
    """П17 (сверх ТЗ): риск КРН участка с класса «высокий». Для газопровода, где модель признана
    неприменимой по шурфам, уведомлений нет. Оговорки — в тексте и в payload."""
    from field_twin.rules import RuleEvent, recipient

    lim = th["П17"]["risk_from"]
    events = []
    for r in risks:
        if r.applicability == "rejected" or r.probability < lim:
            continue
        cav = r.caveats(s["applicability"]["min_pits"])
        prefix = "ПРЕДВАРИТЕЛЬНО: " if cav else ""
        events.append(RuleEvent(
            "П17", now, "pipeline_segments", r.segment_id,
            f"{prefix}Участок {r.segment_id} ({r.pipeline}): риск КРН {r.probability:.2f} — класс «{r.risk_class}». "
            f"Запланировать обследование в шурфе." + (f" Оговорки: {'; '.join(cav)}." if cav else ""),
            recipient("П17", th),
            {"parameter": "scc_risk", "probability": r.probability, "risk_class": r.risk_class,
             "pipeline": r.pipeline, "pipeline_class": r.pipeline_class, "in_domain": r.in_domain,
             "applicability": r.applicability, "synthetic_model": r.synthetic_model, "caveats": cav}))
    return events


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Модель КРН участков газопроводов (Мирзоев, Иващенко, Маршаков, 2016)")
    p.add_argument("--synthetic", action="store_true", help="обучить на синтетике объёма статьи (368 участков)")
    p.add_argument("--csv", default=None, help="реальная выборка: столбцы факторов и scc (0/1)")
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    s = settings(args.config)
    if args.csv:
        data, src = pd.read_csv(args.csv), "real"
    elif args.synthetic:
        data, src = synthetic_dataset(s["samples"], s.get("seed", 0)), SYNTHETIC
    else:
        p.error("укажите --synthetic или --csv")
    m = train(data, s, src)
    print(f"Модель КРН: 10 входов, скрытый слой {s['hidden_layers']} (сигмоида), 2 линейных выхода; "
          f"обучена на {'СИНТЕТИКЕ' if src == SYNTHETIC else 'реальных данных'}")
    print("Тестовая выборка (55 примеров):")
    print(m.metrics.to_text(s.get("reference")))
    if src == SYNTHETIC:
        print("Метрики на синтетике проверяют работу кода, а не качество прогноза на газопроводах.")
    out = Path(args.out or ROOT / s["model_path"])
    m.save(out)
    print(f"Модель сохранена: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
