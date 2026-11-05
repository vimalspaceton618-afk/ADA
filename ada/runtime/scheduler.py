"""Deterministic bounded FIFO scheduler."""

from __future__ import annotations

import threading
from collections import deque

from .contracts import QueueFullError, TaskSpec


class FIFOScheduler:
    """A thread-safe in-memory FIFO queue with explicit backpressure."""

    def __init__(self, *, maxsize: int = 1024) -> None:
        if type(maxsize) is not int or maxsize < 1:
            raise ValueError("maxsize must be a positive integer")
        self._maxsize = maxsize
        self._queue: deque[TaskSpec] = deque()
        self._lock = threading.RLock()

    @property
    def capacity(self) -> int:
        return self._maxsize

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._queue)

    def put(self, task: TaskSpec) -> None:
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        with self._lock:
            if len(self._queue) >= self._maxsize:
                raise QueueFullError(
                    f"runtime queue is full ({self._maxsize} pending tasks)"
                )
            self._queue.append(task)

    def get_nowait(self) -> TaskSpec | None:
        with self._lock:
            return self._queue.popleft() if self._queue else None

    def remove(self, task_id: str) -> bool:
        """Remove a queued task while preserving the relative order of others."""
        with self._lock:
            for index, task in enumerate(self._queue):
                if task.task_id == task_id:
                    del self._queue[index]
                    return True
            return False

    def contains(self, task_id: str) -> bool:
        with self._lock:
            return any(task.task_id == task_id for task in self._queue)
