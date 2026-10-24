"""М4. Модели дожимной компрессорной станции и внутрипромыслового транспорта (Чудин)
с адаптацией на базовом периоде и связывающим алгоритмом (Еникеев, Топольников).

- нейросеть ДКС: давления на входе и выходе компрессора и расход → мощность и скорость вращения;
  5 скрытых слоёв; метод оптимизации Adam; ориентир средней ошибки по мощности около 1,9 %;
- нейросеть внутрипромыслового транспорта: 10 входов → дебиты скважин; 1 скрытый слой из
  4 нейронов; Adam; ориентир ошибки прогноза на 74 дня — не более 2 %;
- деление выборки 60 / 40 %, суточные данные, дообучение на новых данных (Чудин);
- адаптация: на базовом периоде подбираются поправочные коэффициенты, чтобы расчёт совпал
  с замерами, затем они фиксируются на время прогноза (Еникеев, Топольников);
- связывающий алгоритм: модели звеньев запускаются последовательно и обмениваются данными
  на каждом шаге расчёта (Еникеев, Топольников);
- расхождение модели с замерами выше порога — правило П15 (Григорьев и соавторы).
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from field_twin import config
from field_twin.config import ROOT, parse_interval


def settings(config_dir=None) -> dict[str, Any]:
    return config.load_yaml("models.yaml", config_dir)["chudin"]


def daily_table(data: pd.DataFrame, stream: str, step: str = "1d") -> pd.DataFrame:
    """Суточные средние годных значений: столбцы «объект.параметр», индекс — сутки (UTC)."""
    d = data[(data["stream"] == stream) & data["value_clean"].notna()]
    if "f_downtime" in d:
        d = d[~d["f_downtime"]]
    if "f_outlier" in d:
        d = d[~d["f_outlier"]]
    if d.empty:
        return pd.DataFrame()
    d = d.assign(col=d["object_id"] + "." + d["parameter"], day=d["ts"].dt.floor(parse_interval(step)))
    return d.pivot_table(index="day", columns="col", values="value_clean", aggfunc="mean").sort_index()


def mean_relative_error_pct(y_true, y_pred) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return np.mean(np.abs(y_pred - y_true) / np.abs(y_true), axis=0) * 100


@dataclass
class NetModel:
    """Нейросеть звена с масштабированием, адаптацией и дообучением."""
    name: str
    inputs: list[str]
    outputs: list[str]
    net: MLPRegressor
    sx: StandardScaler
    sy: StandardScaler
    corrections: np.ndarray | None = None      # поправочные коэффициенты адаптации (по выходам)
    adapted_on: tuple[str, str] | None = None
    test_error_pct: dict[str, float] = field(default_factory=dict)
    train_rows: int = 0
    test_rows: int = 0

    def predict_raw(self, x: pd.DataFrame) -> np.ndarray:
        y = self.net.predict(self.sx.transform(x[self.inputs].to_numpy(float)))
        return self.sy.inverse_transform(np.asarray(y).reshape(len(x), -1))

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Расчёт; если модель адаптирована — с зафиксированными поправочными коэффициентами."""
        y = self.predict_raw(x)
        if self.corrections is not None:
            y = y * self.corrections
        return pd.DataFrame(y, index=x.index, columns=self.outputs)

    def adapt(self, base: pd.DataFrame) -> np.ndarray:
        """Подбор поправочных коэффициентов на базовом периоде (Еникеев, Топольников):
        k = Σ(замер·расчёт) / Σ(расчёт²) по каждому выходу — расчёт совпадает с замерами в среднем.
        Коэффициенты фиксируются и в прогнозе не меняются."""
        y = self.predict_raw(base)
        m = base[self.outputs].to_numpy(float)
        self.corrections = (m * y).sum(axis=0) / (y * y).sum(axis=0)
        self.adapted_on = (str(base.index.min()), str(base.index.max()))
        return self.corrections.copy()

    def retrain(self, new: pd.DataFrame, epochs: int = 200) -> None:
        """Дообучение на новых данных (Чудин): продолжение обучения Adam с текущих весов.
        Масштабирование не пересчитывается, чтобы не сдвигать уже выученное."""
        x = self.sx.transform(new[self.inputs].to_numpy(float))
        y = self.sy.transform(new[self.outputs].to_numpy(float))
        y = y.ravel() if y.shape[1] == 1 else y
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for _ in range(epochs):
                self.net.partial_fit(x, y)


def _fit(name: str, table: pd.DataFrame, inputs: list[str], outputs: list[str], cfg: dict[str, Any],
         train_share: float, seed: int) -> NetModel:
    t = table[inputs + outputs].dropna()
    if len(t) < 10:
        raise ValueError(f"{name}: мало суточных данных для обучения ({len(t)})")
    cut = int(len(t) * train_share)               # деление по времени: первые 60 % — обучение
    tr, te = t.iloc[:cut], t.iloc[cut:]
    sx = StandardScaler().fit(tr[inputs].to_numpy(float))
    sy = StandardScaler().fit(tr[outputs].to_numpy(float))
    net = MLPRegressor(hidden_layer_sizes=tuple(cfg["hidden_layers"]), activation=cfg.get("activation", "relu"),
                       solver="adam", max_iter=cfg.get("max_iter", 3000), random_state=seed,
                       early_stopping=False, tol=1e-6, n_iter_no_change=50)
    ytr = sy.transform(tr[outputs].to_numpy(float))
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)   # обучение ограничено max_iter
        net.fit(sx.transform(tr[inputs].to_numpy(float)), ytr.ravel() if ytr.shape[1] == 1 else ytr)
    m = NetModel(name, inputs, outputs, net, sx, sy, train_rows=len(tr), test_rows=len(te))
    if len(te):
        err = mean_relative_error_pct(te[outputs], m.predict(te))
        m.test_error_pct = {o: float(e) for o, e in zip(outputs, err)}
    return m


def transport_columns(s: dict[str, Any]) -> tuple[list[str], list[str]]:
    t = s["transport"]
    plume = t["plume_object"]
    inputs = [f"{plume}.Q_plume_total", f"{plume}.P_zpa_inlet"]
    inputs += [f"{w}.P_buffer" for w in t["wells"]] + [f"{w}.T_buffer" for w in t["wells"]]
    outputs = [f"{w}.Q_well" for w in t["wells"]]
    return inputs, outputs


def dks_columns(s: dict[str, Any]) -> tuple[list[str], list[str]]:
    d = s["dks"]
    return [f"{d['object_id']}.{p}" for p in d["inputs"]], [f"{d['object_id']}.{p}" for p in d["outputs"]]


@dataclass
class ChudinModels:
    dks: NetModel
    transport: NetModel
    settings: dict[str, Any]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ChudinModels":
        return joblib.load(path)

    def report(self) -> str:
        lines = ["Модели Чудина (М4): средняя относительная ошибка на проверочной части (40 %)"]
        for m, ref in ((self.dks, self.settings["dks"].get("reference_error_pct", {})),
                       (self.transport, self.settings["transport"].get("reference_error_pct", {}))):
            lines.append(f"  {m.name}: обучение {m.train_rows} сут, проверка {m.test_rows} сут; "
                         f"слои {list(m.net.hidden_layer_sizes)}, Adam")
            for o, e in m.test_error_pct.items():
                r = ref.get(o.split(".", 1)[1])
                lines.append(f"    {o}: {e:.2f} %" + (f" (ориентир Чудина {r} %)" if r is not None else ""))
            if m.corrections is not None:
                lines.append(f"    поправочные коэффициенты: {np.round(m.corrections, 4).tolist()} "
                             f"(базовый период {m.adapted_on[0]} — {m.adapted_on[1]})")
        lines.append("  Ориентиры статьи — не критерии приёмки.")
        return "\n".join(lines)


def train(dks_table: pd.DataFrame, cluster_table: pd.DataFrame, s: dict[str, Any]) -> ChudinModels:
    seed = s.get("seed", 0)
    di, do = dks_columns(s)
    ti, to = transport_columns(s)
    return ChudinModels(
        _fit("ДКС", dks_table, di, do, s["dks"], s["train_share"], seed),
        _fit("Внутрипромысловый транспорт", cluster_table, ti, to, s["transport"], s["train_share"], seed),
        copy.deepcopy(s),
    )


def run_linked(models: ChudinModels, cluster: pd.DataFrame, dks: pd.DataFrame) -> pd.DataFrame:
    """Связывающий алгоритм (Еникеев, Топольников): на каждом шаге модели звеньев запускаются
    последовательно — сначала транспорт (дебиты скважин), затем ДКС, которая получает от транспорта
    расход (с долей куста dks_flow_factor, тыс. м³/сут → тыс. м³/ч) и берёт давления из замеров."""
    s = models.settings
    obj = s["dks"]["object_id"]
    ti, to = transport_columns(s)
    rows = []
    for t in cluster.index.intersection(dks.index):
        x_tr = cluster.loc[[t], ti]
        wells = models.transport.predict(x_tr)                     # звено 1
        q_dks = float(wells.sum(axis=1).iloc[0]) * s["dks_flow_factor"] / 24.0
        x_dks = pd.DataFrame({f"{obj}.P_in": [dks.at[t, f"{obj}.P_in"]], f"{obj}.P_out": [dks.at[t, f"{obj}.P_out"]],
                              f"{obj}.Q": [q_dks]}, index=[t])
        out = models.dks.predict(x_dks)                             # звено 2 — данные звена 1
        rows.append({"ts": t, **wells.iloc[0].to_dict(), "dks_Q_from_transport": q_dks, **out.iloc[0].to_dict()})
    return pd.DataFrame(rows).set_index("ts") if rows else pd.DataFrame()


def model_errors(models: ChudinModels, dks_table: pd.DataFrame, cluster_table: pd.DataFrame,
                 window_days: int = 7) -> list[dict[str, Any]]:
    """Ошибка моделей относительно замеров за последние window_days суток (для П15)."""
    out = []
    for m, table in ((models.dks, dks_table), (models.transport, cluster_table)):
        t = table.reindex(columns=m.inputs + m.outputs).dropna()
        if t.empty:
            continue
        t = t[t.index > t.index.max() - pd.Timedelta(days=window_days)]
        err = mean_relative_error_pct(t[m.outputs], m.predict(t))
        for o, e in zip(m.outputs, err):
            out.append({"model": m.name, "output": o, "error_pct": float(e), "days": len(t), "end": t.index.max()})
    return out


def main(argv: list[str] | None = None) -> int:
    from field_twin import quality, storage
    from field_twin.simulator import chudin_daily

    p = argparse.ArgumentParser(description="М4. Обучение нейросетей Чудина и адаптация")
    p.add_argument("--db", default=None, help="база двойника: суточные средние из очищенных данных")
    p.add_argument("--synthetic-days", type=int, default=None, help="обучить на синтетике заданной длины")
    p.add_argument("--adapt-days", type=int, default=30, help="длина базового периода адаптации, суток (конец ряда)")
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    s = settings(args.config)
    if args.synthetic_days:
        data = chudin_daily(args.synthetic_days)
    elif args.db:
        data = quality.QualityPipeline(config_dir=args.config).run(quality.load_measurements(storage.connect(args.db))).data
    else:
        p.error("укажите --db или --synthetic-days")
    dt, ct = daily_table(data, "dks", s["step"]), daily_table(data, "cluster", s["step"])
    models = train(dt, ct, s)
    if args.adapt_days:
        models.dks.adapt(dt.dropna().iloc[-args.adapt_days:])
        models.transport.adapt(ct.dropna().iloc[-args.adapt_days:])
    print(models.report())
    out = Path(args.out or ROOT / s["model_path"])
    models.save(out)
    print(f"Модели сохранены: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
