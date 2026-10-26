"""М3. Классификатор гидратов в абсорбере (Сафронова).

- вход: параметры абсорбера, их изменение за последние 2 минуты (1 шаг, приставка d1) и отклонение
  от среднего за 200 предыдущих значений (400 минут, приставка d) — Сафронова, 2026, с. 112–113;
  если есть расход метанола по точкам — добавляется в признаки (оценка целостности данных);
- модель — классификатор CatBoost; Optuna подбирает depth, subsample, colsample_bylevel,
  min_data_in_leaf и learning_rate по минимуму среднеквадратической ошибки (там же, с. 115);
- выход каждые около 2 минут: 1 — признак гидрата, 0 — нет;
- обучение на периодах, которые пользователь разметил как гидратообразование, и на причинах
  аномалий из журнала (М8);
- после обучения — матрица ошибок, точность, полнота, F1-мера и доля правильных ответов.
  Значения из статьи — ориентир, а не критерий приёмки; матрица ошибок обязательна, потому что
  в статье доля правильных ответов не сходится с точностью и полнотой.

Модель работает отдельно от управления режимом и ничего не меняет в оборудовании.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from field_twin import config, storage
from field_twin.config import ROOT, parse_interval

POSITIVE_REASON = "hydrate"


def settings(config_dir=None) -> dict[str, Any]:
    return config.load_yaml("models.yaml", config_dir)["hydrates"]


# ---------- признаки ----------

def build_features(data: pd.DataFrame, s: dict[str, Any]) -> pd.DataFrame:
    """Таблица признаков: строка = момент × абсорбер.

    data — очищенные данные М2 (столбцы stream, object_id, parameter, ts, value_clean).
    """
    step = parse_interval(s["step"])
    plant = s["plant_object"]
    ab = data[data["stream"] == s["stream"]]
    wide = ab.pivot_table(index="ts", columns=["object_id", "parameter"], values="value_clean", aggfunc="last")
    meth = data[data["stream"] == s.get("methanol_stream", "")]
    meth_wide = meth.pivot_table(index="ts", columns="object_id", values="value_clean", aggfunc="last") if len(meth) else None
    grid = pd.date_range(wide.index.min().floor(step), wide.index.max(), freq=step) if len(wide) else pd.DatetimeIndex([])
    frames = []
    for obj in s["objects"]:
        cols = {}
        if obj in wide.columns.get_level_values(0):
            for p in wide[obj].columns:
                cols[p] = wide[(obj, p)]
        if plant in wide.columns.get_level_values(0):
            for p in wide[plant].columns:
                cols[f"plant.{p}"] = wide[(plant, p)]
        if meth_wide is not None:
            for m in meth_wide.columns:
                cols[f"methanol.{m}"] = meth_wide[m]
        if not cols:
            continue
        base = pd.DataFrame(cols).reindex(grid)
        feats = {c: base[c] for c in base.columns}
        for lag in s["lags"]:
            for c in base.columns:
                if lag == 1:
                    # изменение за последний шаг (d1)
                    feats[f"{c}.d{lag}"] = base[c] - base[c].shift(1)
                else:
                    # относительно среднего за lag предыдущих значений (dQ_gas у Сафроновой, с. 113)
                    feats[f"{c}.d{lag}"] = base[c] - base[c].shift(1).rolling(lag, min_periods=1).mean()
        f = pd.DataFrame(feats)
        f.insert(0, "object_id", obj)
        f.index.name = "ts"
        frames.append(f.reset_index())
    if not frames:
        return pd.DataFrame(columns=["ts", "object_id"])
    return pd.concat(frames, ignore_index=True)


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in ("ts", "object_id", "label")]


# ---------- разметка ----------

def labels_from_periods(feats: pd.DataFrame, periods: list[dict[str, Any]]) -> pd.Series:
    """1 внутри периодов гидратообразования, 0 вне их."""
    y = pd.Series(0, index=feats.index)
    for p in periods:
        if p.get("reason_code", POSITIVE_REASON) != POSITIVE_REASON:
            continue
        m = feats["ts"].between(pd.Timestamp(p["start_utc"]), pd.Timestamp(p["end_utc"]))
        if p.get("object_id"):
            m &= feats["object_id"] == p["object_id"]
        y[m] = 1
    return y


def labels_from_journal(conn: sqlite3.Connection, feats: pd.DataFrame, window: str = "20min") -> tuple[pd.Series, pd.Series]:
    """Причины аномалий из журнала: подтверждённое П1 с причиной «гидраты» — 1 за окно правила,
    отклонённое — 0. Возвращает (метки, маска размеченных строк)."""
    y = pd.Series(0, index=feats.index)
    known = pd.Series(False, index=feats.index)
    rows = conn.execute(
        """SELECT e.ts_utc, e.object_id, d.decision, d.reason_code FROM journal_decisions d
           JOIN journal_events e ON e.id = d.event_id WHERE e.code = 'П1'""").fetchall()
    for ts, obj, decision, reason in rows:
        end = pd.Timestamp(ts)
        m = feats["ts"].between(end - pd.Timedelta(window), end) & (feats["object_id"] == obj)
        known |= m
        if decision == "confirm" and reason == POSITIVE_REASON:
            y[m] = 1
    return y, known


def periods_from_truth(truth: dict[str, Any]) -> list[dict[str, Any]]:
    """Эпизоды гидратов генератора как размеченные периоды (для синтетики)."""
    return [{"object_id": d["object_id"], "start_utc": d["start"], "end_utc": d["end"], "reason_code": POSITIVE_REASON}
            for d in truth["defects"] if d["defect"] == "hydrate_episode"]


# ---------- обучение ----------

@dataclass
class Metrics:
    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tn + self.fp + self.fn + self.tp
        return (self.tp + self.tn) / n if n else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"confusion_matrix": [[self.tn, self.fp], [self.fn, self.tp]], "precision": self.precision,
                "recall": self.recall, "f1": self.f1, "accuracy": self.accuracy}

    def to_text(self, reference: dict[str, float] | None = None) -> str:
        ref = reference or {}

        def r(k):
            return f" (статья: {ref[k]})" if k in ref else ""
        return "\n".join([
            "Матрица ошибок (строки — факт, столбцы — прогноз):",
            f"              прогноз 0  прогноз 1",
            f"   факт 0   {self.tn:>9}  {self.fp:>9}",
            f"   факт 1   {self.fn:>9}  {self.tp:>9}",
            f"Точность (precision): {self.precision:.3f}{r('precision')}",
            f"Полнота (recall):     {self.recall:.3f}{r('recall')}",
            f"F1-мера:              {self.f1:.3f}{r('f1')}",
            f"Доля правильных (accuracy): {self.accuracy:.3f}{r('accuracy')}",
            "Значения статьи — ориентир, а не критерий приёмки.",
        ])


def metrics(y_true, y_pred) -> Metrics:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return Metrics(int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum()),
                   int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum()))


@dataclass
class HydrateModel:
    model: Any
    features: list[str]
    threshold: float
    metrics: Metrics | None = None
    params: dict[str, Any] = field(default_factory=dict)
    trained_at: str = ""
    split: str = ""

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Выход модели каждые ~2 мин: 1 — признак гидрата, 0 — нет."""
        x = feats.reindex(columns=self.features)
        proba = self.model.predict_proba(x)[:, 1] if len(x) else np.array([])
        return pd.DataFrame({"ts": feats["ts"].to_numpy(), "object_id": feats["object_id"].to_numpy(),
                             "proba": proba, "label": (proba >= self.threshold).astype(int)})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        meta = {"features": self.features, "threshold": self.threshold, "params": self.params,
                "trained_at": self.trained_at, "split": self.split,
                "metrics": self.metrics.to_dict() if self.metrics else None}
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HydrateModel":
        from catboost import CatBoostClassifier
        path = Path(path)
        m = CatBoostClassifier()
        m.load_model(str(path))
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        met = meta.get("metrics")
        cm = met["confusion_matrix"] if met else None
        return cls(m, meta["features"], meta["threshold"],
                   Metrics(cm[0][0], cm[0][1], cm[1][0], cm[1][1]) if cm else None,
                   meta.get("params", {}), meta.get("trained_at", ""), meta.get("split", ""))


def _split(feats: pd.DataFrame, y: pd.Series, test_share: float) -> tuple[np.ndarray, np.ndarray, str]:
    """Деление по времени: конец ряда — на проверку. Если в одной из частей нет гидратов —
    стратифицированное случайное деление (с пометкой)."""
    order = np.argsort(feats["ts"].to_numpy(), kind="stable")
    cut = int(len(order) * (1 - test_share))
    tr, te = order[:cut], order[cut:]
    if y.iloc[tr].sum() > 0 and y.iloc[te].sum() > 0:
        return tr, te, "time"
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(y)), test_size=test_share, stratify=y, random_state=0)
    return tr, te, "stratified_random"


def train(feats: pd.DataFrame, y: pd.Series, s: dict[str, Any], trials: int | None = None) -> HydrateModel:
    import optuna
    from catboost import CatBoostClassifier

    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Для обучения нужны размеченные периоды и с гидратами, и без них")
    cols = feature_columns(feats)
    tr, te, split = _split(feats, y, s.get("test_share", 0.3))
    x, yv = feats[cols], y.to_numpy()
    # Валидация для Optuna — конец обучающей части по времени.
    tr_sorted = tr[np.argsort(feats["ts"].to_numpy()[tr], kind="stable")]
    vcut = int(len(tr_sorted) * 0.8)
    fit_idx, val_idx = tr_sorted[:vcut], tr_sorted[vcut:]
    if yv[val_idx].sum() == 0 or yv[fit_idx].sum() == 0:
        fit_idx, val_idx = tr, tr
    seed = s.get("seed", 42)

    # Подбираемые параметры и критерий — Сафронова, с. 115; диапазоны, число деревьев и фиксированные
    # bootstrap_type / grow_policy (без них CatBoost не принимает subsample и min_data_in_leaf) —
    # решения прототипа.
    fixed = {"iterations": s.get("iterations", 300), "bootstrap_type": "Bernoulli", "grow_policy": "Depthwise"}

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        }
        m = CatBoostClassifier(**params, **fixed, random_seed=seed, verbose=False, allow_writing_files=False,
                               auto_class_weights="Balanced")
        m.fit(x.iloc[fit_idx], yv[fit_idx])
        # минимальная среднеквадратическая ошибка вероятности на валидации
        return float(np.mean((m.predict_proba(x.iloc[val_idx])[:, 1] - yv[val_idx]) ** 2))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials or s.get("optuna_trials", 15))
    best = {**study.best_params, **fixed}
    model = CatBoostClassifier(**best, random_seed=seed, verbose=False, allow_writing_files=False,
                               auto_class_weights="Balanced")
    model.fit(x.iloc[tr], yv[tr])
    hm = HydrateModel(model, cols, s["threshold"], params=best, split=split,
                      trained_at=datetime.now(timezone.utc).isoformat())
    hm.metrics = metrics(yv[te], hm.predict(feats.iloc[te])["label"])
    return hm


def main(argv: list[str] | None = None) -> int:
    from field_twin import quality

    p = argparse.ArgumentParser(description="М3. Обучение классификатора гидратов")
    p.add_argument("--db", default="data/twin.sqlite")
    p.add_argument("--truth", default=None, help="truth.json генератора: эпизоды как размеченные периоды")
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--out", default=None, help="файл модели (по умолчанию из config/models.yaml)")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    s = settings(args.config)
    conn = storage.connect(args.db)
    data = quality.QualityPipeline(config_dir=args.config).run(quality.load_measurements(conn)).data
    feats = build_features(data, s)
    from field_twin.journal import Journal
    periods = Journal(conn, config_dir=args.config).period_labels()
    if args.truth:
        periods += periods_from_truth(json.loads(Path(args.truth).read_text(encoding="utf-8")))
    y = labels_from_periods(feats, periods)
    yj, known = labels_from_journal(conn, feats)
    y[known & (yj == 1)] = 1
    print(f"Строк: {len(feats)}; с гидратами: {int(y.sum())}; признаков: {len(feature_columns(feats))}")
    hm = train(feats, y, s, args.trials)
    print(f"Деление выборки: {hm.split}; параметры CatBoost: {hm.params}")
    print(hm.metrics.to_text(s.get("reference_metrics")))
    out = Path(args.out or ROOT / s["model_path"])
    hm.save(out)
    print(f"Модель сохранена: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
