"""CPU-first ADA runtime orchestration.

Handlers are synchronous callables with signature handler(payload, context).
Cancellation and timeout are cooperative: long-running handlers must call
context.checkpoint() at safe interruption points.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import (
    DuplicateTaskError,
    HandlerNotFoundError,
    InvalidTransitionError,
    TaskCancelledError,
    TaskSnapshot,
    TaskSpec,
    TaskStatus,
    TaskTimedOutError,
    UnknownTaskError,
    freeze_json,
)
from .events import InMemoryEventBus
from .scheduler import FIFOScheduler
from .state import RuntimeStateStore


TaskHandler = Callable[[Mapping[str, Any], "TaskContext"], Any]


class TaskContext:
    """Cooperative cancellation/deadline boundary supplied to each handler."""

    def __init__(
        self,
        task_id: str,
        cancellation_event: threading.Event,
        timeout_seconds: float | None,
    ) -> None:
        self.task_id = task_id
        self._cancellation_event = cancellation_event
        self._deadline = (
            time.monotonic() + timeout_seconds
            if timeout_seconds is not None
            else None
        )

    @property
    def cancellation_requested(self) -> bool:
        return self._cancellation_event.is_set()

    def checkpoint(self) -> None:
        """Raise a typed exception when cancellation or timeout is observed."""
        if self._cancellation_event.is_set():
            raise TaskCancelledError(f"task {self.task_id} was cancelled")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TaskTimedOutError(f"task {self.task_id} exceeded its time limit")


class Runtime:
    """Bounded task runtime with explicit lifecycle and synchronous dispatch.

    This implementation intentionally has no background worker, process pool,
    distributed queue, or model-specific dependency. The caller controls when
    queued work is executed by invoking run_next().
    """

    def __init__(
        self,
        *,
        queue_capacity: int = 1024,
        event_bus: InMemoryEventBus | None = None,
        state_store: RuntimeStateStore | None = None,
    ) -> None:
        self.scheduler = FIFOScheduler(maxsize=queue_capacity)
        self.events = event_bus or InMemoryEventBus()
        self.state = state_store or RuntimeStateStore()
        self._handlers: dict[str, TaskHandler] = {}
        self._tokens: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def register_handler(
        self, task_type: str, handler: TaskHandler, *, replace: bool = False
    ) -> None:
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("task_type must be a non-empty string")
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            if task_type in self._handlers and not replace:
                raise ValueError(f"handler already registered for task type: {task_type}")
            self._handlers[task_type] = handler

    def submit(self, task: TaskSpec) -> TaskSnapshot:
        """Accept a task into the bounded FIFO queue or reject it atomically."""
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        with self._lock:
            if self.state.contains(task.task_id):
                raise DuplicateTaskError(f"task already exists: {task.task_id}")
            self.state.create(task)
            token = threading.Event()
            self._tokens[task.task_id] = token
            try:
                self.scheduler.put(task)
            except Exception:
                self._tokens.pop(task.task_id, None)
                self.state.discard_created(task.task_id)
                raise
            queued = self.state.transition(task.task_id, TaskStatus.QUEUED)
        self.events.publish(
            "task.queued",
            task.task_id,
            {"task_type": task.task_type, "queue_depth": self.scheduler.depth},
        )
        return queued

    def cancel(self, task_id: str) -> bool:
        """Request cancellation; queued tasks terminate immediately.

        A running handler must observe the request through context.checkpoint().
        Returns False when the task is already terminal or is unknown.
        """
        event_type: str | None = None
        with self._lock:
            try:
                snapshot = self.state.get(task_id)
            except UnknownTaskError:
                return False
            if snapshot.status.is_terminal:
                return False
            token = self._tokens.get(task_id)
            if token is None:
                return False
            token.set()
            if snapshot.status is TaskStatus.QUEUED:
                if not self.scheduler.remove(task_id):
                    # Another run_next call can only remove a task while holding
                    # this same runtime lock, so this indicates an invariant fault.
                    raise InvalidTransitionError(
                        f"queued task {task_id} is missing from the scheduler"
                    )
                self.state.transition(task_id, TaskStatus.CANCELLED)
                self._tokens.pop(task_id, None)
                event_type = "task.cancelled"
            elif snapshot.status is TaskStatus.RUNNING:
                event_type = "task.cancellation_requested"
            else:
                return False
        self.events.publish(event_type, task_id, {"source": "runtime.cancel"})
        return True

    def run_next(self) -> TaskSnapshot | None:
        """Execute one FIFO task; return None when the queue is empty."""
        with self._lock:
            task = self.scheduler.get_nowait()
            if task is None:
                return None
            queued_snapshot = self.state.get(task.task_id)
            if queued_snapshot.status is not TaskStatus.QUEUED:
                raise InvalidTransitionError(
                    f"scheduler contained task {task.task_id} in state "
                    f"{queued_snapshot.status.value}"
                )
            running = self.state.transition(task.task_id, TaskStatus.RUNNING)
            token = self._tokens[task.task_id]
            handler = self._handlers.get(task.task_type)
            context = TaskContext(task.task_id, token, task.timeout_seconds)

        self.events.publish(
            "task.running",
            task.task_id,
            {"task_type": task.task_type, "version": running.version},
        )

        target_status = TaskStatus.SUCCEEDED
        result: Any = None
        error: str | None = None
        try:
            if handler is None:
                raise HandlerNotFoundError(
                    f"no handler registered for task type: {task.task_type}"
                )
            context.checkpoint()
            result = handler(task.payload, context)
            context.checkpoint()
            # Validate output before attempting the state transition so an
            # unsupported object becomes an ordinary FAILED task, not a stuck one.
            result = freeze_json(result, path="task.result")
        except TaskCancelledError as exc:
            target_status = TaskStatus.CANCELLED
            error = str(exc)
        except TaskTimedOutError as exc:
            target_status = TaskStatus.TIMED_OUT
            error = str(exc)
        except Exception as exc:
            target_status = TaskStatus.FAILED
            error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            # Close the race between a handler's last checkpoint and a concurrent
            # cancellation request. The lock makes the terminal transition the
            # single winner against cancel().
            if target_status is TaskStatus.SUCCEEDED:
                try:
                    context.checkpoint()
                except TaskCancelledError as exc:
                    target_status = TaskStatus.CANCELLED
                    error = str(exc)
                except TaskTimedOutError as exc:
                    target_status = TaskStatus.TIMED_OUT
                    error = str(exc)
            finished = self.state.transition(
                task.task_id,
                target_status,
                result=result if target_status is TaskStatus.SUCCEEDED else None,
                error=error,
            )
            self._tokens.pop(task.task_id, None)

        self.events.publish(
            f"task.{target_status.value}",
            task.task_id,
            {
                "task_type": task.task_type,
                "version": finished.version,
                "error": error,
            },
        )
        return finished

    def run_until_idle(self, *, max_tasks: int | None = None) -> tuple[TaskSnapshot, ...]:
        """Drain currently queued work, optionally limiting the number executed."""
        if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 0):
            raise ValueError("max_tasks must be a non-negative integer or None")
        completed: list[TaskSnapshot] = []
        while max_tasks is None or len(completed) < max_tasks:
            snapshot = self.run_next()
            if snapshot is None:
                break
            completed.append(snapshot)
        return tuple(completed)

    def get_task(self, task_id: str) -> TaskSnapshot:
        return self.state.get(task_id)
