"""Bounded, ordered in-process event bus with isolated subscribers."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..contracts import RuntimeEvent, freeze_json, utc_now

EventHandler = Callable[[RuntimeEvent], None]


@dataclass(frozen=True, slots=True)
class EventDeliveryFailure:
    sequence: int
    handler_name: str
    error: str


class InMemoryEventBus:
    def __init__(self, *, max_history: int = 1000, max_failures: int = 100) -> None:
        if type(max_history) is not int or max_history < 1:
            raise ValueError("max_history must be positive")
        if type(max_failures) is not int or max_failures < 1:
            raise ValueError("max_failures must be positive")
        self._lock = threading.RLock()
        self._history: deque[RuntimeEvent] = deque(maxlen=max_history)
        self._failures: deque[EventDeliveryFailure] = deque(maxlen=max_failures)
        self._subscribers: dict[int, tuple[str | None, EventHandler]] = {}
        self._next_id = 1
        self._sequence = 0

    def subscribe(self, handler: EventHandler, *, event_type: str | None = None) -> Callable[[], None]:
        if not callable(handler):
            raise TypeError("handler must be callable")
        if event_type is not None and (not isinstance(event_type, str) or not event_type.strip()):
            raise ValueError("event_type must be non-empty or None")
        with self._lock:
            key = self._next_id
            self._next_id += 1
            self._subscribers[key] = (event_type, handler)

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.pop(key, None)
        return unsubscribe

    def publish(self, event_type: str, task_id: str, details: Mapping[str, Any] | None = None) -> RuntimeEvent:
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be non-empty")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be non-empty")
        frozen = freeze_json(details or {}, path="event.details")
        with self._lock:
            self._sequence += 1
            event = RuntimeEvent(self._sequence, event_type, task_id, utc_now(), frozen)
            self._history.append(event)
            subscribers = tuple(self._subscribers.values())
        for filter_type, handler in subscribers:
            if filter_type is not None and filter_type != event_type:
                continue
            try:
                handler(event)
            except Exception as exc:
                failure = EventDeliveryFailure(
                    event.sequence, getattr(handler, "__qualname__", repr(handler)),
                    f"{type(exc).__name__}: {exc}",
                )
                with self._lock:
                    self._failures.append(failure)
        return event

    @property
    def history(self) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            return tuple(self._history)

    @property
    def delivery_failures(self) -> tuple[EventDeliveryFailure, ...]:
        with self._lock:
            return tuple(self._failures)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)
