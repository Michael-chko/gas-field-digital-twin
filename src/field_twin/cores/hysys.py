"""М9. Адаптер Aspen HYSYS: заводская часть двойника (подготовка ПГ, сжижение, аммиак, карбамид).

Связь — через COM-автоматизацию HYSYS (только Windows, нужна лицензия HYSYS и пакет pywin32):
двойник записывает в модель последние очищенные значения промысла (расход, давление, температура
газа на заводе), запускает расчёт и читает результаты — производительность по СПГ, NH₃, карбамиду
и мощности компрессоров. Соответствие сигналов и объектов модели — config/hysys.yaml.

Рекомендации идут по общему пути ядер: П12 → главный инженер → журнал. В АСУ ТП ничего не пишется.

Имена свойств COM (MaterialStreams.Item(...).MassFlow.GetValue("tonne/h"), Solver.CanSolve)
соответствуют распространённым версиям HYSYS; на своей версии их нужно проверить (docs/hysys.md).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol

from field_twin import config
from field_twin.cores.base import Change, CoreAdapter, CoreInput, CoreResult, Recommendation


class HysysBackend(Protocol):
    def open(self, path: str, visible: bool) -> None: ...
    def hold(self) -> None: ...
    def set(self, collection: str, name: str, prop: str, value: float, unit: str) -> None: ...
    def solve(self, timeout_s: float) -> None: ...
    def get(self, collection: str, name: str, prop: str, unit: str) -> float: ...
    def close(self) -> None: ...


class ComBackend:
    """Доступ к HYSYS через COM (win32com)."""

    def open(self, path: str, visible: bool) -> None:
        try:
            import win32com.client  # pywin32: только Windows
        except ImportError as e:
            raise RuntimeError("Связь с HYSYS работает только под Windows с пакетом pywin32 и установленным "
                               "Aspen HYSYS (pip install pywin32)") from e
        self.app = win32com.client.Dispatch("HYSYS.Application")
        self.case = self.app.SimulationCases.Open(str(path))
        self.case.Visible = bool(visible)
        self.fs = self.case.Flowsheet

    def _obj(self, collection: str, name: str):
        return getattr(self.fs, collection).Item(name)

    def hold(self) -> None:
        self.case.Solver.CanSolve = False      # не пересчитывать после каждого изменения

    def set(self, collection, name, prop, value, unit) -> None:
        getattr(self._obj(collection, name), prop).SetValue(float(value), unit)

    def solve(self, timeout_s: float) -> None:
        solver = self.case.Solver
        solver.CanSolve = True
        t0 = time.monotonic()
        while getattr(solver, "IsSolving", False):
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError("HYSYS не завершил расчёт за отведённое время")
            time.sleep(0.2)

    def get(self, collection, name, prop, unit) -> float:
        return float(getattr(self._obj(collection, name), prop).GetValue(unit))

    def close(self) -> None:
        if getattr(self, "case", None) is not None:
            self.case.Close()
            self.case = None


class FakeBackend:
    """Модель-заменитель для проверки адаптера без HYSYS: хранит значения и считает выходы
    функцией model(values) -> {(collection, name, prop): value}."""

    def __init__(self, model: Callable[[dict], dict] | None = None):
        self.values: dict[tuple[str, str, str], float] = {}
        self.model = model
        self.solved = 0
        self.held = False
        self.opened = None

    def open(self, path, visible):
        self.opened = path

    def hold(self):
        self.held = True

    def set(self, collection, name, prop, value, unit):
        self.values[(collection, name, prop)] = float(value)

    def solve(self, timeout_s):
        self.held = False
        self.solved += 1
        if self.model:
            self.values.update(self.model(self.values))

    def get(self, collection, name, prop, unit):
        return self.values[(collection, name, prop)]

    def close(self):
        self.opened = None


class HysysCore(CoreAdapter):
    name = "hysys"

    def __init__(self, options: dict[str, Any] | None = None, backend: HysysBackend | None = None,
                 settings: dict[str, Any] | None = None):
        super().__init__(options)
        self.s = settings or config.load_yaml("hysys.yaml", (options or {}).get("config_dir"))
        self.backend = backend or ComBackend()
        self.connected = False

    def connect(self) -> None:
        self.backend.open(self.s["case_path"], self.s.get("visible", False))
        self.connected = True

    def calculate(self, inp: CoreInput) -> CoreResult:
        if not self.connected:
            self.connect()
        res = CoreResult(self.name, inp.now)
        missing = []
        self.backend.hold()
        for m in self.s.get("inputs", []):
            v = inp.get(m["signal"])
            if v is None:
                missing.append(m["signal"])
                continue
            self.backend.set(m["collection"], m["name"], m["property"], v * m.get("scale", 1), m["unit"])
        self.backend.solve(self.s.get("solve_timeout_s", 120))
        for key, m in (self.s.get("outputs") or {}).items():
            res.calculated[key] = self.backend.get(m["collection"], m["name"], m["property"], m["unit"])
        power = {k: self.backend.get(m["collection"], m["name"], m["property"], m["unit"])
                 for k, m in (self.s.get("compressor_power") or {}).items()}
        for k, v in power.items():
            res.calculated[f"power_kw.{k}"] = v
        lng = res.calculated.get("lng_t_h")
        if power and lng:
            res.calculated["specific_energy_kwh_t"] = sum(power.values()) / lng
        if missing:
            res.calculated["missing_inputs"] = float(len(missing))
        self._recommend(res, power)
        return res

    def _recommend(self, res: CoreResult, power: dict[str, float]) -> None:
        rule = (self.s.get("recommendations") or {}).get("specific_energy") or {}
        base, lim = rule.get("baseline_kwh_t"), rule.get("max_increase_pct")
        se = res.calculated.get("specific_energy_kwh_t")
        if base is None or lim is None or se is None:
            return
        growth = (se - base) / base * 100
        if growth <= lim:
            return
        top = sorted(power.items(), key=lambda kv: -kv[1])[:2]
        res.recommendations.append(Recommendation(
            key="hysys:specific_energy",
            title="Растёт удельный расход энергии на тонну СПГ",
            message=(f"Удельный расход энергии {se:.0f} кВт·ч/т выше базового {base:g} на {growth:.1f} % — "
                     f"перераспределить нагрузку между компрессорами хладагента "
                     f"(наиболее нагружены {', '.join(k for k, _ in top)}); по расчёту модели HYSYS"),
            object_id="LNG",
            changes=[Change(f"power_kw.{k}", v, None, "кВт") for k, v in top],
            expected_effect={"specific_energy_kwh_t": base}))

    def close(self) -> None:
        if self.connected:
            self.backend.close()
            self.connected = False
