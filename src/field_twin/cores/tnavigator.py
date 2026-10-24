"""М9. Адаптер tNavigator (ИРМ; Кулагин и соавторы).

Ядро пласта и сети сбора; опция Network; расчёты управляются скриптами на Python. Температуры
в сети сбора у Кулагина и соавторов пришлось досчитывать отдельным скриптом.

Пустой класс: методы описаны, реализация — после уточнения условий лицензии и программного
интерфейса («Открытые вопросы»).
"""

from __future__ import annotations

from field_twin.cores.base import CoreAdapter, CoreInput, CoreResult

TODO = "tNavigator: условия лицензии и программный интерфейс не уточнены — см. «Открытые вопросы» ТЗ"


class TNavigatorCore(CoreAdapter):
    name = "tnavigator"

    def connect(self) -> None:
        """Запустить tNavigator с интегрированной моделью (пласт + Network)."""
        raise NotImplementedError(TODO)

    def is_available(self) -> bool:
        return False

    def run_script(self, script: str, params: dict) -> dict:
        """Выполнить управляющий скрипт на Python внутри tNavigator и вернуть результаты."""
        raise NotImplementedError(TODO)

    def calculate(self, inp: CoreInput) -> CoreResult:
        """Обновить граничные условия по замерам, пересчитать пласт и сеть сбора,
        вернуть давления и дебиты по узлам и варианты распределения отборов."""
        raise NotImplementedError(TODO)
