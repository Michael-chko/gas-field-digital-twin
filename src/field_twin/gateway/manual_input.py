"""М12. Ручной ввод данных, которых нет в АСУ: ручные замеры электрохимической защиты,
лабораторные анализы, результаты исследований скважин (Куник и соавторы; Запевалов и соавторы).

Значение сохраняется с пометкой «ручной» (origin = manual) и проходит те же проверки приёма
и качества, что и данные АСУ.
"""

from __future__ import annotations

from datetime import datetime

from field_twin.ingest import IngestReport, Ingestor


class ManualInput:
    def __init__(self, ingestor: Ingestor):
        self.ingestor = ingestor

    def add(self, stream: str, object_id: str, parameter: str, value: float | str, ts: datetime,
            user_name: str, quality: str = "good") -> IngestReport:
        if not user_name:
            raise ValueError("Не указан, кто вводит значение")
        if ts.tzinfo is None:
            raise ValueError("Время замера нужно указать с часовым поясом")
        raw = value if isinstance(value, str) else repr(float(value))
        return self.ingestor.ingest_records(
            [{"stream": stream, "object_id": object_id, "parameter": parameter, "ts": ts, "value_raw": raw,
              "quality": quality}], source=f"manual:{user_name}", origin="manual")
