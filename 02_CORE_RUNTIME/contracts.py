"""Shared immutable contracts and errors for the ADA Core Runtime."""

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
    """Validate and deeply freeze JSON-compatible data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            result[key] = freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, path=f"{path}[{i}]") for i, item in enumerate(value))
    raise TypeError(f"{path} must contain JSON-compatible values, got {type(value).__name__}")


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
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED, self.TIMED_OUT}


class RuntimeErrorBase(Exception):
    """Base class for runtime contract and execution errors."""


class InvalidTransitionError(RuntimeErrorBase): pass
class DuplicateTaskError(RuntimeErrorBase): pass
class UnknownTaskError(RuntimeErrorBase): pass
class QueueFullError(RuntimeErrorBase): pass
class HandlerNotFoundError(RuntimeErrorBase): pass
class TaskCancelledError(RuntimeErrorBase): pass
class TaskTimedOutError(RuntimeErrorBase): pass
class RetryExhaustedError(RuntimeErrorBase): pass


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeout_seconds: float | None = None
    max_attempts: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, str) or not self.task_type.strip():
            raise ValueError("task_type must be non-empty")
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
                raise TypeError("timeout_seconds must be a positive finite number")
            if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be a positive finite number")
            object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        object.__setattr__(self, "payload", freeze_json(self.payload, path="payload"))


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    task: TaskSpec
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    version: int = 0
    attempt: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: Any = None
    error: str | None = None

    @property
    def task_id(self) -> str: return self.task.task_id

    @property
    def task_type(self) -> str: return self.task.task_type


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    sequence: int
    event_type: str
    task_id: str
    occurred_at: datetime
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 1 or not self.event_type.strip() or not self.task_id.strip():
            raise ValueError("event sequence, type and task id must be valid")
        object.__setattr__(self, "details", freeze_json(self.details, path="event.details"))
