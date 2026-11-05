"""Explicit handler registry with duplicate-registration protection."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from ..contracts import HandlerNotFoundError

TaskHandler = Callable[[Mapping[str, Any], Any], Any]


class HandlerRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handlers: dict[str, TaskHandler] = {}

    def register(self, task_type: str, handler: TaskHandler, *, replace: bool = False) -> None:
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("task_type must be non-empty")
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            if task_type in self._handlers and not replace:
                raise ValueError(f"handler already registered: {task_type}")
            self._handlers[task_type] = handler

    def resolve(self, task_type: str) -> TaskHandler:
        with self._lock:
            handler = self._handlers.get(task_type)
        if handler is None:
            raise HandlerNotFoundError(f"no handler registered for task type: {task_type}")
        return handler

    def contains(self, task_type: str) -> bool:
        with self._lock:
            return task_type in self._handlers

    def unregister(self, task_type: str) -> bool:
        with self._lock:
            return self._handlers.pop(task_type, None) is not None

    @property
    def task_types(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._handlers))
