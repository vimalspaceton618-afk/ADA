"""Typed, immutable contracts for the ADA Core Runtime."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def freeze_json(value: Any, *, path: str = "value") -> Any:
    """Validate and recursively freeze a JSON-compatible value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            frozen[key] = freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only JSON-compatible values; "
        f"got {type(value).__name__}"
    )


def thaw_json(value: Any) -> Any:
    """Return ordinary dict/list containers from a frozen JSON value."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


class TaskStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
        }


class RuntimeErrorBase(Exception):
    """Base class for expected ADA runtime errors."""


class InvalidTransitionError(RuntimeErrorBase):
    """Raised when a task lifecycle transition is not permitted."""


class DuplicateTaskError(RuntimeErrorBase):
    """Raised when a task identifier is submitted more than once."""


class UnknownTaskError(RuntimeErrorBase):
    """Raised when a task identifier is not known to the state store."""


class QueueFullError(RuntimeErrorBase):
    """Raised when the bounded scheduler cannot accept another task."""


class HandlerNotFoundError(RuntimeErrorBase):
    """Raised when no handler is registered for a task type."""


class TaskCancelledError(RuntimeErrorBase):
    """Raised when a task observes a cooperative cancellation request."""


class TaskTimedOutError(RuntimeErrorBase):
    """Raised when a task observes that its execution deadline has passed."""


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Immutable task input. Payloads must be JSON-compatible."""

    task_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, str) or not self.task_type.strip():
            raise ValueError("task_type must be a non-empty string")
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        if self.timeout_seconds is not None:
            timeout = self.timeout_seconds
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout_seconds must be a positive finite number")
            if not math.isfinite(float(timeout)) or timeout <= 0:
                raise ValueError("timeout_seconds must be a positive finite number")
            object.__setattr__(self, "timeout_seconds", float(timeout))
        frozen_payload = freeze_json(self.payload, path="payload")
        object.__setattr__(self, "payload", frozen_payload)


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    """Read-only view of a task and its current lifecycle state."""

    task: TaskSpec
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    version: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: Any = None
    error: str | None = None

    @property
    def task_id(self) -> str:
        """Convenience identifier for this task snapshot."""
        return self.task.task_id

    @property
    def task_type(self) -> str:
        """Task handler key associated with this snapshot."""
        return self.task.task_type


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """Immutable event emitted after a successful state transition."""

    sequence: int
    event_type: str
    task_id: str
    occurred_at: datetime
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("event sequence must be positive")
        if not self.event_type.strip() or not self.task_id.strip():
            raise ValueError("event_type and task_id must be non-empty")
        object.__setattr__(
            self, "details", freeze_json(self.details, path="event.details")
        )
