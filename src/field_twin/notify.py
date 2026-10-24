"""М7 (часть этапа 4). Правило П1 и теневой режим (Сафронова; Аксютин и соавторы).

П1: предупреждение о гидратах со звуком — только при 3 и более положительных значениях модели М3
за 20 минут. В предупреждении — действия из статьи: проверить уносы из абсорбера А-2, при
необходимости подать метанол в линию орошения или увеличить его расход перед абсорбером А-2.

Теневой режим: модели и правила работают, звуковой сигнал не подаётся, сигналы сравниваются
с фактом постфактум (Сафронова).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from field_twin import config

P1_ACTIONS = ("Проверить уносы из абсорбера А-2; при необходимости подать метанол в линию орошения "
              "или увеличить его расход перед абсорбером А-2 (Сафронова)")


@dataclass
class Notification:
    code: str
    ts: pd.Timestamp
    object_id: str
    message: str
    recipient: str
    sound: bool
    shadow: bool
    first_positive: pd.Timestamp
    positives: int
    actions: str = P1_ACTIONS
    stream: str = "absorber"
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def delay(self) -> pd.Timedelta:
        """Время от первого срабатывания модели до предупреждения (ТЗ, «Ограничения данных»)."""
        return self.ts - self.first_positive


def evaluate_p1(predictions: pd.DataFrame, thresholds: dict[str, Any] | None = None,
                shadow: bool = False) -> list[Notification]:
    """predictions: ts, object_id, label (выход М3). Предупреждение — в момент, когда в окне
    [t − 20 мин, t] набралось 3 положительных значения; повторно — только после того, как окно
    опустело ниже порога."""
    rule = (thresholds or config.thresholds())["П1"]
    need, window = int(rule["min_positive"]), pd.Timedelta(minutes=rule["window_minutes"])
    out: list[Notification] = []
    for obj, g in predictions.sort_values("ts").groupby("object_id"):
        pos = list(g.loc[g["label"] == 1, "ts"])
        active = False
        streak_start = None
        for i, t in enumerate(pos):
            in_win = [p for p in pos[: i + 1] if p >= t - window]
            if len(in_win) < need:
                active = False
                streak_start = None if len(in_win) == 1 else streak_start
                if len(in_win) == 1:
                    streak_start = t
                continue
            if streak_start is None:
                streak_start = in_win[0]
            if not active:
                active = True
                out.append(Notification(
                    "П1", t, obj,
                    f"Гидраты в абсорбере {obj}: {len(in_win)} положительных значений модели за "
                    f"{rule['window_minutes']} мин. {P1_ACTIONS}",
                    rule["recipient"], sound=not shadow, shadow=shadow, first_positive=streak_start,
                    positives=len(in_win)))
        # окно опустело — правило снова взводится
    return out


def log_notifications(journal, notes: list[Notification]) -> list[int]:
    ids = []
    for n in notes:
        ids.append(journal.log_event(
            "notification", code=n.code, message=n.message, ts=n.ts.to_pydatetime(), stream=n.stream,
            object_id=n.object_id, shadow=n.shadow,
            payload={"recipient": n.recipient, "sound": n.sound, "actions": n.actions, "positives": n.positives,
                     "first_positive": n.first_positive.isoformat(), "delay_s": n.delay.total_seconds(), **n.payload}))
    return ids


def log_model_outputs(journal, predictions: pd.DataFrame, shadow: bool = False, code: str = "М3") -> int:
    """Каждое положительное срабатывание модели — в журнал (М8)."""
    pos = predictions[predictions["label"] == 1]
    journal.log_many([
        {"kind": "model", "code": code, "ts": t.to_pydatetime(), "stream": "absorber", "object_id": o,
         "message": f"М3: признак гидрата, вероятность {p:.2f}", "payload": {"proba": float(p)}, "shadow": shadow}
        for t, o, p in zip(pos["ts"], pos["object_id"], pos["proba"])
    ])
    return len(pos)


def shadow_comparison(notes: list[Notification], periods: list[dict[str, Any]]) -> dict[str, Any]:
    """Теневой режим: сравнение предупреждений с фактом постфактум."""
    hits, false_alarms = [], []
    found = set()
    for n in notes:
        match = [i for i, p in enumerate(periods)
                 if (not p.get("object_id") or p["object_id"] == n.object_id)
                 and pd.Timestamp(p["start_utc"]) <= n.ts <= pd.Timestamp(p["end_utc"])]
        if match:
            hits.append(n)
            found.update(match)
        else:
            false_alarms.append(n)
    return {"warnings": len(notes), "hits": len(hits), "false_alarms": len(false_alarms),
            "missed_episodes": len(periods) - len(found),
            "delays_s": [(n.ts - pd.Timestamp(periods[0]["start_utc"])).total_seconds() for n in hits] if periods else []}
