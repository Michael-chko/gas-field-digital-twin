"""М12. Чтение данных АСУ ТП по OPC UA — только чтение (Куник и соавторы; Сафронова).

Двойник подключается к верхнему уровню АСУ через стандартный интерфейс и берёт данные в реальном
времени и из архива. У адаптера нет методов записи: между сетью АСУ и сетью двойника стоит
межсетевой экран, решения вводит человек в штатную АСУ.

Какие узлы читать, решает словарь параметров: имя сигнала АСУ → объект двойника. Переход на
другой сервер OPC UA — смена адреса, пространства имён и словаря, без правки кода.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from asyncua import Client, ua

from field_twin.gateway.tag_dictionary import TagDictionary

logging.getLogger("asyncua").setLevel(logging.ERROR)


def quality_of(status) -> str:
    code = status.value if hasattr(status, "value") else int(status)
    if code & 0x80000000:
        return "bad"
    if code & 0x40000000:
        return "uncertain"
    return "good"


@dataclass
class BindingReport:
    """Сверка узлов сервера со словарём параметров."""
    unbound_on_server: list[str] = field(default_factory=list)   # на сервере есть, в словаре нет
    missing_on_server: list[str] = field(default_factory=list)   # в словаре есть, на сервере нет
    without_setpoints: list[str] = field(default_factory=list)   # уставки в словаре не заполнены


class OpcUaReader:
    """Клиент OPC UA двойника. Методов записи нет намеренно."""

    def __init__(self, url: str, tags: TagDictionary, namespace_uri: str, name: str = "opcua",
                 username: str | None = None, password: str | None = None, timeout: float = 4.0):
        self.url, self.tags, self.namespace_uri, self.name = url, tags, namespace_uri, name
        self.username, self.password, self.timeout = username, password, timeout
        self.client: Client | None = None
        self.ns: int | None = None
        self.last_ts: datetime | None = None

    async def connect(self) -> None:
        c = Client(self.url, timeout=self.timeout)
        if self.username:
            c.set_user(self.username)
            c.set_password(self.password or "")
        await c.connect()
        self.client = c
        self.ns = await c.get_namespace_index(self.namespace_uri)

    async def disconnect(self) -> None:
        if self.client is not None:
            try:
                await self.client.disconnect()
            finally:
                self.client = None

    @property
    def connected(self) -> bool:
        return self.client is not None

    def _node(self, tag: str):
        return self.client.get_node(ua.NodeId(tag, self.ns))

    def _record(self, tag: str, dv) -> dict[str, Any] | None:
        sig = self.tags.by_tag(tag)
        if sig is None:
            return None
        v = dv.Value.Value if dv.Value is not None else None
        value = None if v is None or v != v else float(v)
        ts = dv.SourceTimestamp or dv.ServerTimestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return {"stream": sig.stream, "object_id": sig.object_id, "parameter": sig.parameter, "ts": ts,
                "value": value, "unit": sig.unit, "quality": quality_of(dv.StatusCode),
                "source": self.name, "tag": tag}

    async def verify(self) -> BindingReport:
        """Каждый сигнал сервера должен быть привязан к объекту двойника; сверка в обе стороны."""
        field_obj = await self.client.nodes.objects.get_child([f"{self.ns}:Field"])
        server_tags = set()
        for ch in await field_obj.get_children():
            server_tags.add((await ch.read_browse_name()).Name)
        dict_tags = {s.tag for s in self.tags.signals.values()}
        return BindingReport(sorted(server_tags - dict_tags), sorted(dict_tags - server_tags),
                             sorted(s.tag for s in self.tags.without_setpoints() if s.tag in server_tags))

    async def read_current(self, tags: list[str] | None = None) -> list[dict[str, Any]]:
        tags = tags or [s.tag for s in self.tags.signals.values()]
        nodes = [self._node(t) for t in tags]
        dvs = await self.client.read_attributes(nodes, ua.AttributeIds.Value)
        out = []
        for t, dv in zip(tags, dvs):
            if dv.StatusCode.value == ua.StatusCodes.BadWaitingForInitialData:
                continue                    # данных по узлу ещё не было
            r = self._record(t, dv)
            if r is not None and dv.SourceTimestamp is not None:
                out.append(r)
        if out:
            self.last_ts = max(r["ts"] for r in out)
        return out

    async def read_history(self, start: datetime, end: datetime, tags: list[str] | None = None) -> list[dict[str, Any]]:
        """Архив за период (HistoryRead) — для обучения моделей и досылки после обрыва связи."""
        tags = tags or [s.tag for s in self.tags.signals.values()]
        out = []
        for t in tags:
            try:
                hist = await self._node(t).read_raw_history(start, end)
            except ua.UaStatusCodeError:
                continue
            for dv in hist:
                if dv.StatusCode.value == ua.StatusCodes.BadWaitingForInitialData:
                    continue
                r = self._record(t, dv)
                if r is not None:
                    out.append(r)
        out.sort(key=lambda r: r["ts"])
        if out:
            self.last_ts = max(self.last_ts or out[-1]["ts"], out[-1]["ts"])
        return out

    async def subscribe(self, on_record: Callable[[dict], None], on_alarm: Callable[[dict], None] | None = None,
                        period_ms: int = 500):
        """Подписка на изменения значений и на технологическую сигнализацию."""
        reader = self

        class Handler:
            def datachange_notification(self, node, val, data):
                tag = node.nodeid.Identifier
                r = reader._record(tag, data.monitored_item.Value)
                if r is not None:
                    on_record(r)

            def event_notification(self, event):
                if on_alarm:
                    ts = getattr(event, "Time", None) or datetime.now(timezone.utc)
                    on_alarm({"ts": ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
                              "tag": str(getattr(event, "SourceName", "") or ""),
                              "message": event.Message.Text if event.Message else "",
                              "severity": int(getattr(event, "Severity", 0) or 0), "source": reader.name})

        sub = await self.client.create_subscription(period_ms, Handler())
        await sub.subscribe_data_change([self._node(s.tag) for s in self.tags.signals.values()])
        await sub.subscribe_events()
        return sub
