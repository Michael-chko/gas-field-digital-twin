"""М9. Заглушка расчётного ядра на синтетических данных.

Вместо расчёта установки подготовки (как у АЭРОСИМ) даёт простые рекомендации по текущим
замерам, чтобы путь «ядро → П12 → главный инженер → журнал» работал до подключения настоящего
ядра. Пороги — синтетика (config/cores.yaml).
"""

from __future__ import annotations

from field_twin.cores.base import Change, CoreAdapter, CoreInput, CoreResult, Recommendation


class StubCore(CoreAdapter):
    name = "stub"

    def calculate(self, inp: CoreInput) -> CoreResult:
        res = CoreResult(self.name, inp.now)
        q1, q2 = inp.get("absorber.A-1.Q_gas"), inp.get("absorber.A-2.Q_gas")
        if q1 and q2:
            total = q1 + q2
            res.calculated["absorber.total_Q_gas"] = total
            imb = abs(q1 - q2) / (total / 2) * 100
            res.calculated["absorber.imbalance_pct"] = imb
            if imb > self.options.get("imbalance_pct", 3):
                hi, lo = ("A-1", "A-2") if q1 > q2 else ("A-2", "A-1")
                res.recommendations.append(Recommendation(
                    key=f"redistribute:{hi}->{lo}",
                    title="Перераспределить поток газа между абсорберами",
                    message=f"Расход через {hi} выше, чем через {lo}, на {imb:.1f} %: выровнять нагрузку "
                            f"до {total / 2:.0f} тыс. м³/ч на аппарат",
                    object_id=hi,
                    changes=[Change(f"absorber.{hi}.Q_gas", max(q1, q2), total / 2, "тыс. м³/ч"),
                             Change(f"absorber.{lo}.Q_gas", min(q1, q2), total / 2, "тыс. м³/ч")],
                    expected_effect={"imbalance_pct": 0.0}))
        level, meth = inp.get("absorber.A-2.L_cube"), inp.get("methanol.M-2.Q_methanol")
        nominal = self.options.get("nominal_level", 280.0)
        if level is not None and meth is not None and level < nominal * (1 - self.options.get("level_drop_pct", 10) / 100):
            res.recommendations.append(Recommendation(
                key="methanol:A-2",
                title="Увеличить подачу метанола перед абсорбером А-2",
                message=f"Уровень в кубе А-2 {level:.0f} мм ниже рабочего — увеличить расход метанола "
                        f"в точке М-2 на 10 % (Сафронова: подача ингибитора перед абсорбером А-2)",
                object_id="A-2",
                changes=[Change("methanol.M-2.Q_methanol", meth, meth * 1.1, "кг/ч")]))
        return res
