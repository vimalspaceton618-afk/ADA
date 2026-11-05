"""Thread-safe task lifecycle state store."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from .contracts import (
    DuplicateTaskError,
    InvalidTransitionError,
    TaskSnapshot,
    TaskSpec,
    TaskStatus,
    UnknownTaskError,
    freeze_json,
    utc_now,
)


_ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset({TaskStatus.QUEUED}),
    TaskStatus.QUEUED: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED}),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
        }
    ),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.TIMED_OUT: frozenset(),
}


class RuntimeStateStore:
    """Stores immutable snapshots and enforces the lifecycle transition graph."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskSnapshot] = {}

    def create(self, task: TaskSpec) -> TaskSnapshot:
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        now = utc_now()
        snapshot = TaskSnapshot(
            task=task,
            status=TaskStatus.CREATED,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            if task.task_id in self._tasks:
                raise DuplicateTaskError(f"task already exists: {task.task_id}")
            self._tasks[task.task_id] = snapshot
        return snapshot

    def discard_created(self, task_id: str) -> None:
        """Rollback an unaccepted task; only CREATED records may be discarded."""
        with self._lock:
            snapshot = self._require(task_id)
            if snapshot.status is not TaskStatus.CREATED:
                raise InvalidTransitionError(
                    "only a CREATED task may be discarded during submission rollback"
                )
            del self._tasks[task_id]

    def transition(
        self,
        task_id: str,
        target: TaskStatus,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> TaskSnapshot:
        if not isinstance(target, TaskStatus):
            raise TypeError("target must be a TaskStatus")
        with self._lock:
            current = self._require(task_id)
            if target not in _ALLOWED_TRANSITIONS[current.status]:
                raise InvalidTransitionError(
                    f"cannot transition task {task_id} from "
                    f"{current.status.value} to {target.value}"
                )
            if target is TaskStatus.SUCCEEDED:
                frozen_result = freeze_json(result, path="task.result")
            else:
                frozen_result = current.result
            if error is not None and not isinstance(error, str):
                raise TypeError("error must be a string or None")
            now = utc_now()
            updated = replace(
                current,
                status=target,
                updated_at=now,
                version=current.version + 1,
                started_at=now if target is TaskStatus.RUNNING else current.started_at,
                finished_at=now if target.is_terminal else current.finished_at,
                result=frozen_result,
                error=error if target is TaskStatus.FAILED else current.error,
            )
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

    def by_status(self, *statuses: TaskStatus) -> tuple[TaskSnapshot, ...]:
        wanted = set(statuses)
        with self._lock:
            return tuple(
                snapshot
                for snapshot in self._tasks.values()
                if snapshot.status in wanted
            )

    def _require(self, task_id: str) -> TaskSnapshot:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise UnknownTaskError(f"unknown task: {task_id}") from exc
