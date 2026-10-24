"""Имитатор АСУ ТП промысла: сервер OPC UA с сигналами из словаря параметров (М12).

- узлы — по словарю параметров: Objects/Field/<имя сигнала АСУ>; значение с признаком качества
  и меткой времени источника;
- история по каждому узлу (HistoryRead) и отдельная архивная база (archive.py);
- технологическая сигнализация по уставкам словаря — события OPC UA (alarms.py);
- узлы доступны только на чтение: двойник не может ничего изменить в «АСУ».

Наполняется генератором синтетических данных: записи CSV длинного формата подаются в порядке
времени (вручную через publish или проигрыванием файла).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from asyncua import Server, ua

from asu_simulator.alarms import Alarm, AlarmEngine
from asu_simulator.archive import AsuArchive
from field_twin.gateway.tag_dictionary import TagDictionary

NAMESPACE = "urn:field-twin:asu-simulator"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
QUALITY_CODES = {"good": ua.StatusCodes.Good, "uncertain": ua.StatusCodes.Uncertain, "bad": ua.StatusCodes.Bad}

logging.getLogger("asyncua").setLevel(logging.ERROR)


@dataclass
class Sample:
    tag: str
    ts: datetime
    value: float | None
    quality: str = "good"


class AsuSimulator:
    def __init__(self, url: str, tags: TagDictionary, namespace: str = NAMESPACE,
                 extra_tags: Iterable[str] = (), archive: AsuArchive | None = None):
        """extra_tags — узлы, которых нет в словаре двойника (проверка «сигнал без привязки»)."""
        self.url, self.tags, self.namespace = url, tags, namespace
        self.extra_tags = list(extra_tags)
        self.archive = archive or AsuArchive()
        self.alarms = AlarmEngine({s.tag: s.setpoints for s in tags.signals.values()})
        self.nodes: dict[str, object] = {}
        self.server: Server | None = None
        self.alarm_log: list[Alarm] = []

    async def start(self) -> None:
        self.server = Server()
        await self.server.init()
        self.server.set_endpoint(self.url)
        self.server.set_server_name("Имитатор АСУ ТП промысла")
        idx = await self.server.register_namespace(self.namespace)
        field = await self.server.nodes.objects.add_object(idx, "Field")
        for tag in [s.tag for s in self.tags.signals.values()] + self.extra_tags:
            node = await field.add_variable(ua.NodeId(tag, idx), tag, ua.Variant(0.0, ua.VariantType.Double))
            # до первого значения узел честно сообщает, что данных ещё нет; метка времени нужна архиву
            await node.write_value(ua.DataValue(ua.Variant(0.0, ua.VariantType.Double),
                                                StatusCode=ua.StatusCode(ua.StatusCodes.BadWaitingForInitialData),
                                                SourceTimestamp=EPOCH, ServerTimestamp=EPOCH))
            await self.server.historize_node_data_change(node, period=None, count=0)
            self.nodes[tag] = node          # переменные не делаются записываемыми: только чтение
        self.evgen = await self.server.get_event_generator()
        await self.server.start()

    async def stop(self) -> None:
        if self.server is not None:
            await self.server.stop()
            self.server = None

    async def publish(self, samples: Iterable[Sample]) -> list[Alarm]:
        """Записать значения в узлы (как их обновил бы контроллер) и сгенерировать сигнализацию."""
        raised = []
        for smp in samples:
            node = self.nodes.get(smp.tag)
            if node is None:
                continue
            code = ua.StatusCode(QUALITY_CODES.get(smp.quality, ua.StatusCodes.Bad))
            val = ua.Variant(float("nan") if smp.value is None else float(smp.value), ua.VariantType.Double)
            await node.write_value(ua.DataValue(val, StatusCode=code, SourceTimestamp=smp.ts, ServerTimestamp=smp.ts))
            self.archive.write(smp.tag, smp.ts, smp.value, smp.quality)
            alarm = self.alarms.evaluate(smp.tag, smp.value)
            if alarm:
                self.evgen.event.Message = ua.LocalizedText(alarm.message)
                self.evgen.event.Severity = alarm.severity
                self.evgen.event.SourceName = alarm.tag
                await self.evgen.trigger(time_attr=smp.ts)
                self.alarm_log.append(alarm)
                raised.append(alarm)
        self.archive.commit()
        return raised


def samples_from_csv(files: Iterable[str | Path], tags: TagDictionary) -> list[Sample]:
    """Записи CSV генератора → значения для узлов (по словарю: поток.объект.параметр → имя в АСУ)."""
    out = []
    for f in files:
        with open(f, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                sig = tags.get(r["stream"], r["object_id"], r["parameter"])
                try:
                    ts = datetime.fromisoformat(r["timestamp"])
                except (ValueError, TypeError):
                    continue
                if sig is None or ts.tzinfo is None:
                    continue
                try:
                    value = float(r["value"]) if r["value"] else None
                except ValueError:
                    value = None
                out.append(Sample(sig.tag, ts.astimezone(timezone.utc), value, r.get("quality") or "good"))
    out.sort(key=lambda s: s.ts)
    return out


async def serve(url: str, files: list[str], speed: float, tags: TagDictionary) -> None:
    sim = AsuSimulator(url, tags)
    await sim.start()
    samples = samples_from_csv(files, tags)
    print(f"Имитатор АСУ ТП: {url}; сигналов {len(sim.nodes)}; записей для проигрывания {len(samples)}")
    try:
        prev = None
        for s in samples:
            if prev is not None and speed > 0:
                await asyncio.sleep(max(0.0, (s.ts - prev).total_seconds() / speed))
            prev = s.ts
            for a in await sim.publish([s]):
                print(f"  сигнализация: {a.message}")
        print("Проигрывание закончено; сервер работает, Ctrl+C — выход")
        while True:
            await asyncio.sleep(3600)
    finally:
        await sim.stop()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Имитатор АСУ ТП: сервер OPC UA, архив, сигнализация")
    p.add_argument("files", nargs="*", help="CSV генератора для проигрывания")
    p.add_argument("--url", default="opc.tcp://127.0.0.1:4840/asu")
    p.add_argument("--speed", type=float, default=60.0, help="во сколько раз быстрее реального времени")
    p.add_argument("--dictionary", default=None, help="файл словаря параметров (по умолчанию config/tags.yaml)")
    args = p.parse_args(argv)
    tags = TagDictionary.from_file(args.dictionary) if args.dictionary else TagDictionary.load()
    try:
        asyncio.run(serve(args.url, args.files, args.speed, tags))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
