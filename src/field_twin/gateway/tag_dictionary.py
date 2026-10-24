"""Словарь параметров (М12; Куник и соавторы; Аксютин и соавторы).

Каждый сигнал привязан к объекту двойника: скважине, шлейфу, аппарату.
Компактное описание из config/tags.yaml разворачивается в список сигналов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from field_twin.config import load_yaml, parse_interval


@dataclass(frozen=True)
class Signal:
    stream: str
    object_id: str
    parameter: str
    unit: str
    kind: str
    title: str
    interval: timedelta | None
    tag: str
    setpoints: dict[str, Any] | None = None
    object_attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.stream, self.object_id, self.parameter)


def _objects(raw: list[Any]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append((str(item["id"]), dict(item.get("attrs") or {})))
        else:
            out.append((str(item), {}))
    return out


class TagDictionary:
    def __init__(self, data: dict[str, Any]):
        self.streams: dict[str, dict[str, Any]] = data.get("streams") or {}
        self.signals: dict[tuple[str, str, str], Signal] = {}
        for stream, sdef in self.streams.items():
            interval = parse_interval(sdef.get("interval"))
            default_objects = _objects(sdef.get("objects") or [])
            for param, pdef in (sdef.get("parameters") or {}).items():
                objs = _objects(pdef["objects"]) if "objects" in pdef else default_objects
                for obj_id, attrs in objs:
                    sig = Signal(
                        stream=stream,
                        object_id=obj_id,
                        parameter=param,
                        unit=str(pdef.get("unit", "")),
                        kind=str(pdef.get("kind", "other")),
                        title=str(pdef.get("title", param)),
                        interval=interval,
                        # Имя сигнала в АСУ: tags[объект] → tag → условное «поток.объект.параметр».
                        tag=(pdef.get("tags") or {}).get(obj_id) or pdef.get("tag") or f"{stream}.{obj_id}.{param}",
                        setpoints=(pdef.get("setpoints_by_object") or {}).get(obj_id) or pdef.get("setpoints"),
                        object_attrs=attrs,
                    )
                    self.signals[sig.key] = sig

    @classmethod
    def load(cls, config_dir: Path | str | None = None) -> "TagDictionary":
        return cls(load_yaml("tags.yaml", config_dir))

    @classmethod
    def from_file(cls, path: Path | str) -> "TagDictionary":
        """Словарь другой АСУ (переход на другой сервер — смена адреса и словаря)."""
        import yaml
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def get(self, stream: str, object_id: str, parameter: str) -> Signal | None:
        return self.signals.get((stream, object_id, parameter))

    def by_tag(self, tag: str) -> Signal | None:
        if not hasattr(self, "_by_tag"):
            self._by_tag = {s.tag: s for s in self.signals.values()}
        return self._by_tag.get(tag)

    def without_setpoints(self) -> list[Signal]:
        """Сигналы, для которых владелец АСУ ещё не задал уставки (ТЗ, М12)."""
        return [s for s in self.signals.values() if not s.setpoints]

    def stream_signals(self, stream: str) -> list[Signal]:
        return [s for s in self.signals.values() if s.stream == stream]

    def stream_interval(self, stream: str) -> timedelta | None:
        return parse_interval(self.streams[stream].get("interval"))

    def __len__(self) -> int:
        return len(self.signals)
