"""Small thread-safe coordination primitives without global state."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from collections.abc import Iterator


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> bool:
        was_set = self._event.is_set()
        self._event.set()
        return not was_set

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            from ..contracts import TaskCancelledError
            raise TaskCancelledError("operation cancelled")


class CapacityLimiter:
    """A bounded semaphore with a context-manager acquisition API."""

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._semaphore = threading.BoundedSemaphore(capacity)

    @contextmanager
    def slot(self, *, timeout: float | None = None) -> Iterator[None]:
        acquired = self._semaphore.acquire(timeout=timeout) if timeout is not None else self._semaphore.acquire()
        if not acquired:
            raise TimeoutError("capacity slot acquisition timed out")
        try:
            yield
        finally:
            self._semaphore.release()
