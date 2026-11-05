"""Typed, bounded in-process pub/sub message transport."""

from __future__ import annotations

import threading
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..contracts import freeze_json, utc_now


@dataclass(frozen=True, slots=True)
class MessageEnvelope:
    topic: str
    sender: str
    payload: Mapping[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=utc_now)
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.topic.strip() or not self.sender.strip():
            raise ValueError("topic and sender must be non-empty")
        object.__setattr__(self, "payload", freeze_json(self.payload, path="message.payload"))


class InMemoryMessageBus:
    """Synchronous local transport; handlers execute outside the bus lock."""

    def __init__(self, *, history_limit: int = 500) -> None:
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError("history_limit must be positive")
        self._lock = threading.RLock()
        self._history: deque[MessageEnvelope] = deque(maxlen=history_limit)
        self._subscribers: dict[int, tuple[str, Callable[[MessageEnvelope], None]]] = {}
        self._next_id = 1
        self._failures: deque[str] = deque(maxlen=100)

    def subscribe(self, topic: str, handler: Callable[[MessageEnvelope], None]) -> Callable[[], None]:
        if not topic.strip() or not callable(handler):
            raise ValueError("topic must be non-empty and handler callable")
        with self._lock:
            key = self._next_id
            self._next_id += 1
            self._subscribers[key] = (topic, handler)
        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.pop(key, None)
        return unsubscribe

    def publish(self, message: MessageEnvelope) -> MessageEnvelope:
        if not isinstance(message, MessageEnvelope):
            raise TypeError("message must be MessageEnvelope")
        with self._lock:
            self._history.append(message)
            targets = tuple(self._subscribers.values())
        for topic, handler in targets:
            if topic != message.topic:
                continue
            try:
                handler(message)
            except Exception as exc:
                with self._lock:
                    self._failures.append(f"{type(exc).__name__}: {exc}")
        return message

    @property
    def history(self) -> tuple[MessageEnvelope, ...]:
        with self._lock:
            return tuple(self._history)

    @property
    def delivery_failures(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._failures)
