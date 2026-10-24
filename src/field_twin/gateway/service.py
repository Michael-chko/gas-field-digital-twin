"""М12. Служба шлюза: источники АСУ → буфер → база двойника; сигнализация АСУ → журнал.

- два независимых источника (основной и резервный сервер сбора): при отказе одного приём
  продолжается со второго (Куник и соавторы);
- на каждом опросе из архива сервера (HistoryRead) дочитывается всё с последней принятой метки
  времени — и между опросами, и после обрыва связи; повторы отсекает буфер, поэтому досылка идёт
  без потерь и дублей;
- технологическая сигнализация АСУ пишется в журнал отдельно от уведомлений двойника (source = asu);
- в отчёт о приёме попадают сигналы источника без привязки к объекту двойника и сигналы
  с незаполненными уставками.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from field_twin.gateway.buffer import DeliveryBuffer
from field_twin.gateway.opcua_reader import BindingReport, OpcUaReader
from field_twin.ingest import IngestReport, Ingestor
from field_twin.journal import Journal


@dataclass
class PollResult:
    source: str | None
    received: int = 0
    new: int = 0
    delivered: int = 0
    backfilled: int = 0
    failover: bool = False
    delivery_error: str | None = None
    report: IngestReport | None = None


class GatewayService:
    def __init__(self, readers: list[OpcUaReader], buffer: DeliveryBuffer, ingestor: Ingestor,
                 journal: Journal | None = None, backfill: bool = True):
        self.readers, self.buffer, self.ingestor, self.journal = readers, buffer, ingestor, journal
        self.backfill = backfill
        self.active: OpcUaReader | None = None
        self.binding: BindingReport | None = None
        self.last_ts: datetime | None = None
        self.sink_ok = True                     # можно имитировать обрыв связи с центром
        self.alarms: list[dict[str, Any]] = []
        self._report_binding = True
        self._sub = None

    async def connect(self, start: int = 0) -> OpcUaReader:
        errors = []
        order = self.readers[start:] + self.readers[:start]
        for r in order:
            try:
                if not r.connected:
                    await r.connect()
                self.active = r
                self.binding = await r.verify()
                return r
            except Exception as e:   # noqa: BLE001 — источник недоступен, пробуем следующий
                errors.append(f"{r.name}: {type(e).__name__}")
                await r.disconnect()
        self.active = None
        raise ConnectionError("Нет доступных источников АСУ: " + "; ".join(errors))

    async def subscribe_alarms(self) -> None:
        if self.active is None:
            await self.connect()
        self._sub = await self.active.subscribe(lambda r: None, self._on_alarm)

    def _on_alarm(self, a: dict[str, Any]) -> None:
        self.alarms.append(a)
        if self.journal is not None:
            sig = self.active.tags.by_tag(a["tag"]) if self.active else None
            self.journal.log_event("asu_alarm", code="АСУ", source="asu", message=a["message"], ts=a["ts"],
                                   stream=sig.stream if sig else None, object_id=sig.object_id if sig else None,
                                   payload={"tag": a["tag"], "severity": a["severity"], "source": a["source"]})

    async def poll(self) -> PollResult:
        """Опрос: всё, что накопилось в архиве источника с последней принятой метки времени
        (без потерь между опросами), плюс текущие значения. Повторы отсекает буфер."""
        failover = False
        if self.active is None or not self.active.connected:
            await self.connect()
        try:
            recs = await self._read(self.active)
        except Exception:   # noqa: BLE001 — обрыв: переключаемся на другой источник
            failed = self.active
            await failed.disconnect()
            idx = self.readers.index(failed)
            await self.connect((idx + 1) % len(self.readers))
            failover = self.active is not failed
            recs = await self._read(self.active)
        res = PollResult(self.active.name, received=len(recs), failover=failover)
        res.backfilled = sum(1 for r in recs if r.get("_from_history"))
        for r in recs:
            r.pop("_from_history", None)
        res.new = self.buffer.put(recs, origin="asu")
        if recs:
            newest = max(r["ts"] for r in recs)
            self.last_ts = newest if self.last_ts is None else max(self.last_ts, newest)
        try:
            res.delivered = self.buffer.flush(lambda batch: self._sink(batch, res))
        except ConnectionError as e:
            res.delivery_error = str(e)
        return res

    async def _read(self, reader: OpcUaReader) -> list[dict[str, Any]]:
        recs = []
        if self.backfill:
            since = self.last_ts or datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
            hist = await reader.read_history(since, datetime.now(timezone.utc) + timedelta(days=3650))
            for r in hist:
                r["_from_history"] = True
            recs += hist
        return recs + await reader.read_current()

    def _sink(self, batch: list[dict[str, Any]], res: PollResult) -> None:
        if not self.sink_ok:
            raise ConnectionError("нет связи с центром — записи остаются в буфере")
        b = self.binding or BindingReport()
        rep = self.ingestor.ingest_records(
            batch, source=f"asu:{self.active.name if self.active else '?'}", origin="asu",
            unbound=b.unbound_on_server if self._report_binding else None,
            without_setpoints=b.without_setpoints if self._report_binding else None)
        self._report_binding = False
        res.report = rep

    async def close(self) -> None:
        for r in self.readers:
            await r.disconnect()


def main(argv: list[str] | None = None) -> int:
    from field_twin import config, storage
    from field_twin.cycle import CycleRunner
    from field_twin.gateway.tag_dictionary import TagDictionary

    p = argparse.ArgumentParser(description="Шлюз М12: чтение АСУ ТП по OPC UA в базу двойника")
    p.add_argument("--db", default="data/twin.sqlite")
    p.add_argument("--dictionary", default=None, help="файл словаря параметров подключаемой АСУ")
    p.add_argument("--url", action="append", default=None, help="адрес сервера OPC UA (можно два: основной и резервный)")
    p.add_argument("--namespace", default=None)
    p.add_argument("--cycles", action="store_true", help="после каждого опроса запускать цикл двойника")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    gw = config.load_yaml("gateway.yaml", args.config)
    tags = TagDictionary.from_file(args.dictionary) if args.dictionary else TagDictionary.load(args.config)
    sources = gw["opcua"]["sources"]
    if args.url:
        sources = [{"name": f"src{i + 1}", "url": u, "namespace_uri": args.namespace or sources[0]["namespace_uri"]}
                   for i, u in enumerate(args.url)]
    readers = [OpcUaReader(s["url"], tags, s["namespace_uri"], s["name"], gw["opcua"].get("username"),
                           gw["opcua"].get("password")) for s in sources]
    conn = storage.connect(args.db)
    svc = GatewayService(readers, DeliveryBuffer(gw["buffer_path"]), Ingestor(conn, tags=tags, config_dir=args.config),
                         Journal(conn, config_dir=args.config), gw["opcua"].get("backfill_history", True))
    runner = CycleRunner(conn, config_dir=args.config) if args.cycles else None

    async def loop():
        await svc.connect()
        print(f"Подключено к {svc.active.name} ({svc.active.url})")
        if svc.binding.unbound_on_server:
            print(f"  сигналы АСУ без привязки: {len(svc.binding.unbound_on_server)}")
        print(f"  сигналов с незаполненными уставками: {len(svc.binding.without_setpoints)}")
        await svc.subscribe_alarms()
        while True:
            try:
                r = await svc.poll()
                print(f"{datetime.now():%H:%M:%S} {r.source}: получено {r.received}, новых {r.new}, "
                      f"доставлено {r.delivered}, из архива {r.backfilled}" + (" — ПЕРЕКЛЮЧЕНИЕ" if r.failover else ""))
                if runner:
                    runner.step()
            except ConnectionError as e:
                print(f"{datetime.now():%H:%M:%S} {e}")
            await asyncio.sleep(gw["opcua"].get("poll_seconds", 5))

    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
