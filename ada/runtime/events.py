"""Bounded in-process event publication with subscriber isolation."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .contracts import RuntimeEvent, freeze_json, utc_now


EventHandler = Callable[[RuntimeEvent], None]


@dataclass(frozen=True, slots=True)
class EventDeliveryFailure:
    sequence: int
    handler_name: str
    error: str


class InMemoryEventBus:
    """Synchronous event bus; handler failures never block other subscribers."""

    def __init__(self, *, max_history: int = 1000, max_failures: int = 100) -> None:
        if type(max_history) is not int or max_history < 1:
            raise ValueError("max_history must be a positive integer")
        if type(max_failures) is not int or max_failures < 1:
            raise ValueError("max_failures must be a positive integer")
        self._lock = threading.RLock()
        self._history: deque[RuntimeEvent] = deque(maxlen=max_history)
        self._failures: deque[EventDeliveryFailure] = deque(maxlen=max_failures)
        self._subscriptions: dict[int, tuple[str | None, EventHandler]] = {}
        self._next_subscription = 1
        self._sequence = 0

    def subscribe(
        self, handler: EventHandler, *, event_type: str | None = None
    ) -> Callable[[], None]:
        if not callable(handler):
            raise TypeError("handler must be callable")
        if event_type is not None and (
            not isinstance(event_type, str) or not event_type.strip()
        ):
            raise ValueError("event_type must be a non-empty string or None")
        with self._lock:
            subscription_id = self._next_subscription
            self._next_subscription += 1
            self._subscriptions[subscription_id] = (event_type, handler)

        def unsubscribe() -> None:
            with self._lock:
                self._subscriptions.pop(subscription_id, None)

        return unsubscribe

    def publish(
        self, event_type: str, task_id: str, details: dict[str, Any] | None = None
    ) -> RuntimeEvent:
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        frozen_details = freeze_json(details or {}, path="event.details")
        with self._lock:
            self._sequence += 1
            event = RuntimeEvent(
                sequence=self._sequence,
                event_type=event_type,
                task_id=task_id,
                occurred_at=utc_now(),
                details=frozen_details,
            )
            self._history.append(event)
            handlers = tuple(self._subscriptions.values())

        for filter_type, handler in handlers:
            if filter_type is not None and filter_type != event_type:
                continue
            try:
                handler(event)
            except Exception as exc:
                failure = EventDeliveryFailure(
                    sequence=event.sequence,
                    handler_name=getattr(handler, "__qualname__", repr(handler)),
                    error=f"{type(exc).__name__}: {exc}",
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
            return len(self._subscriptions)
