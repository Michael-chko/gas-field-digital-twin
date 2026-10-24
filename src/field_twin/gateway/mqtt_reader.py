"""М12. Приём событий MQTT от удалённых объектов (кусты, шельф) — только подписка.

Тема: <prefix>/<stream>/<object_id>/<parameter>; сообщение JSON:
{"v": значение, "q": "good|uncertain|bad", "t": "время ISO 8601 с поясом", "state": "run|stop|repair"}.
Формат сообщения MQTT не стандартизован — он задаётся здесь и в словаре параметров.
Публиковать двойник ничего не может: методов публикации у адаптера нет.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Callable

import paho.mqtt.client as mqtt

from field_twin.gateway.tag_dictionary import TagDictionary


class MqttReader:
    def __init__(self, host: str, port: int, tags: TagDictionary, prefix: str = "field", qos: int = 1,
                 name: str = "mqtt"):
        self.host, self.port, self.tags, self.prefix, self.qos, self.name = host, port, tags, prefix, qos, name
        self.unbound: set[str] = set()
        self.errors: list[str] = []
        self._client: mqtt.Client | None = None
        self._lock = threading.Lock()
        self._on_record: Callable[[dict], None] | None = None

    def parse(self, topic: str, payload: bytes) -> dict[str, Any] | None:
        parts = topic.split("/")
        if len(parts) != 4 or parts[0] != self.prefix:
            self.unbound.add(topic)
            return None
        _, stream, obj, param = parts
        sig = self.tags.get(stream, obj, param)
        if sig is None:
            self.unbound.add(topic)        # сигнал без привязки в словаре
            return None
        try:
            d = json.loads(payload)
            ts = datetime.fromisoformat(d["t"])
            if ts.tzinfo is None:
                raise ValueError("время без часового пояса")
            v = d.get("v")
            return {"stream": stream, "object_id": obj, "parameter": param, "ts": ts.astimezone(timezone.utc),
                    "value": None if v is None else float(v), "unit": sig.unit, "quality": d.get("q", "good"),
                    "equipment_state": d.get("state"), "source": self.name, "tag": sig.tag}
        except (ValueError, KeyError, TypeError) as e:
            self.errors.append(f"{topic}: {e}")
            return None

    def start(self, on_record: Callable[[dict], None]) -> None:
        self._on_record = on_record
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"field-twin-{self.name}")

        def on_message(client, userdata, msg):
            r = self.parse(msg.topic, msg.payload)
            if r is not None:
                with self._lock:
                    self._on_record(r)

        def on_connect(client, userdata, flags, reason, props):
            client.subscribe(f"{self.prefix}/#", qos=self.qos)

        c.on_message = on_message
        c.on_connect = on_connect
        c.connect(self.host, self.port)
        c.loop_start()
        self._client = c

    def stop(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
