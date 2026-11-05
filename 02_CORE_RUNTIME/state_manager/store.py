"""Thread-safe task state machine and immutable snapshots."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

from ..contracts import (
    DuplicateTaskError, InvalidTransitionError, TaskSnapshot, TaskSpec,
    TaskStatus, UnknownTaskError, freeze_json, utc_now,
)

_ALLOWED = {
    TaskStatus.CREATED: {TaskStatus.QUEUED},
    TaskStatus.QUEUED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT},
    TaskStatus.SUCCEEDED: set(), TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(), TaskStatus.TIMED_OUT: set(),
}


class RuntimeStateStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskSnapshot] = {}

    def create(self, task: TaskSpec) -> TaskSnapshot:
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        now = utc_now()
        item = TaskSnapshot(task, TaskStatus.CREATED, now, now)
        with self._lock:
            if task.task_id in self._tasks:
                raise DuplicateTaskError(f"duplicate task id: {task.task_id}")
            self._tasks[task.task_id] = item
        return item

    def discard_created(self, task_id: str) -> None:
        with self._lock:
            item = self._require(task_id)
            if item.status is not TaskStatus.CREATED:
                raise InvalidTransitionError("only CREATED tasks can be rolled back")
            del self._tasks[task_id]

    def transition(self, task_id: str, target: TaskStatus, *, result: Any = None, error: str | None = None) -> TaskSnapshot:
        if not isinstance(target, TaskStatus):
            raise TypeError("target must be TaskStatus")
        if error is not None and not isinstance(error, str):
            raise TypeError("error must be str or None")
        with self._lock:
            current = self._require(task_id)
            if target not in _ALLOWED[current.status]:
                raise InvalidTransitionError(f"illegal transition {current.status.value} -> {target.value}")
            frozen_result = freeze_json(result, path="task.result") if target is TaskStatus.SUCCEEDED else current.result
            now = utc_now()
            updated = replace(
                current, status=target, updated_at=now, version=current.version + 1,
                started_at=now if target is TaskStatus.RUNNING else current.started_at,
                finished_at=now if target.is_terminal else current.finished_at,
                result=frozen_result, error=error if target.is_terminal else current.error,
            )
            self._tasks[task_id] = updated
            return updated

    def increment_attempt(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            current = self._require(task_id)
            if current.status is not TaskStatus.RUNNING:
                raise InvalidTransitionError("attempt count can change only while RUNNING")
            updated = replace(current, attempt=current.attempt + 1, version=current.version + 1, updated_at=utc_now())
            self._tasks[task_id] = updated
            return updated

    def get(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            return self._require(task_id)

    def contains(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._tasks

    def all(self) -> tuple[TaskSnapshot, ...]:
        with self._lock:
            return tuple(self._tasks.values())

    def _require(self, task_id: str) -> TaskSnapshot:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise UnknownTaskError(f"unknown task: {task_id}") from exc
