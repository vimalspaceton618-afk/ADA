"""Top-level CPU-first ADA Core Runtime orchestration."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

from .contracts import (
    DuplicateTaskError, HandlerNotFoundError, InvalidTransitionError,
    TaskCancelledError, TaskSnapshot, TaskSpec, TaskStatus, TaskTimedOutError,
    UnknownTaskError, freeze_json,
)
from .task_scheduler.fifo import FIFOScheduler
from .state_manager.store import RuntimeStateStore
from .event_system.bus import InMemoryEventBus
from .execution_engine.registry import HandlerRegistry
from .fault_recovery.policy import RetryPolicy
from .runtime_monitor.health import RuntimeHealthSnapshot, RuntimeMonitor


class TaskContext:
    """Cooperative interruption boundary exposed to task handlers."""

    def __init__(self, task_id: str, cancel_event: threading.Event, timeout: float | None) -> None:
        self.task_id = task_id
        self._cancel_event = cancel_event
        self._deadline = time.monotonic() + timeout if timeout is not None else None

    @property
    def cancellation_requested(self) -> bool:
        return self._cancel_event.is_set()

    def checkpoint(self) -> None:
        if self._cancel_event.is_set():
            raise TaskCancelledError(f"task {self.task_id} was cancelled")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TaskTimedOutError(f"task {self.task_id} exceeded its deadline")


class Runtime:
    """Bounded FIFO runtime with explicit lifecycle, events and retry policy.

    Dispatch is caller-driven via run_next()/run_until_idle(). No hidden worker
    thread is started. Running-task cancellation and deadlines are cooperative.
    """

    def __init__(
        self, *, queue_capacity: int = 1024,
        event_bus: InMemoryEventBus | None = None,
        state_store: RuntimeStateStore | None = None,
        retry_policies: Mapping[str, RetryPolicy] | None = None,
    ) -> None:
        self.scheduler = FIFOScheduler(maxsize=queue_capacity)
        self.events = event_bus or InMemoryEventBus()
        self.state = state_store or RuntimeStateStore()
        self.handlers = HandlerRegistry()
        self._tokens: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._event_lock = threading.RLock()
        self._retry_policies = dict(retry_policies or {})
        if any(not isinstance(policy, RetryPolicy) for policy in self._retry_policies.values()):
            raise TypeError("retry_policies values must be RetryPolicy instances")

    def register_handler(self, task_type: str, handler: Any, *, replace: bool = False) -> None:
        self.handlers.register(task_type, handler, replace=replace)

    def set_retry_policy(self, task_type: str, policy: RetryPolicy) -> None:
        if not isinstance(policy, RetryPolicy):
            raise TypeError("policy must be RetryPolicy")
        with self._lock:
            self._retry_policies[task_type] = policy

    def submit(self, task: TaskSpec) -> TaskSnapshot:
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be TaskSpec")
        with self._event_lock:
            with self._lock:
                if self.state.contains(task.task_id):
                    raise DuplicateTaskError(f"task already exists: {task.task_id}")
                self.state.create(task)
                self._tokens[task.task_id] = threading.Event()
                try:
                    self.scheduler.put(task)
                except Exception:
                    self._tokens.pop(task.task_id, None)
                    self.state.discard_created(task.task_id)
                    raise
                queued = self.state.transition(task.task_id, TaskStatus.QUEUED)
                depth = self.scheduler.depth
            self.events.publish("task.queued", task.task_id, {"task_type": task.task_type, "queue_depth": depth})
        return queued

    def cancel(self, task_id: str) -> bool:
        with self._event_lock:
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
                        raise InvalidTransitionError(f"queued task missing from scheduler: {task_id}")
                    self.state.transition(task_id, TaskStatus.CANCELLED, error="cancelled before execution")
                    self._tokens.pop(task_id, None)
                    event_type = "task.cancelled"
                elif snapshot.status is TaskStatus.RUNNING:
                    event_type = "task.cancellation_requested"
                else:
                    return False
            self.events.publish(event_type, task_id, {"source": "runtime.cancel"})
        return True

    def run_next(self) -> TaskSnapshot | None:
        with self._event_lock:
            with self._lock:
                task = self.scheduler.get_nowait()
                if task is None:
                    return None
                current = self.state.get(task.task_id)
                if current.status is not TaskStatus.QUEUED:
                    raise InvalidTransitionError(f"expected QUEUED, got {current.status.value}")
                running = self.state.transition(task.task_id, TaskStatus.RUNNING)
                token = self._tokens[task.task_id]
                context = TaskContext(task.task_id, token, task.timeout_seconds)
                policy = self._retry_policies.get(task.task_type, RetryPolicy(max_attempts=task.max_attempts))
            self.events.publish("task.running", task.task_id, {"task_type": task.task_type, "version": running.version})

        target = TaskStatus.SUCCEEDED
        result: Any = None
        error: str | None = None
        attempts = 0
        while True:
            attempts += 1
            with self._lock:
                self.state.increment_attempt(task.task_id)
            try:
                context.checkpoint()
                handler = self.handlers.resolve(task.task_type)
                result = freeze_json(handler(task.payload, context), path="task.result")
                context.checkpoint()
                break
            except TaskCancelledError as exc:
                target, error = TaskStatus.CANCELLED, str(exc)
                break
            except TaskTimedOutError as exc:
                target, error = TaskStatus.TIMED_OUT, str(exc)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                decision = policy.decide(completed_attempts=attempts, error=exc)
                self.events.publish("task.attempt_failed", task.task_id, {
                    "attempt": attempts, "retry": decision.retry, "delay_seconds": decision.delay_seconds,
                    "error": error,
                })
                if not decision.retry:
                    target = TaskStatus.FAILED
                    break
                if token.wait(decision.delay_seconds):
                    target, error = TaskStatus.CANCELLED, f"task {task.task_id} cancelled during retry backoff"
                    break
                context.checkpoint()

        with self._event_lock:
            with self._lock:
                if target is TaskStatus.SUCCEEDED:
                    try:
                        context.checkpoint()
                    except TaskCancelledError as exc:
                        target, error = TaskStatus.CANCELLED, str(exc)
                    except TaskTimedOutError as exc:
                        target, error = TaskStatus.TIMED_OUT, str(exc)
                finished = self.state.transition(
                    task.task_id, target,
                    result=result if target is TaskStatus.SUCCEEDED else None,
                    error=error,
                )
                self._tokens.pop(task.task_id, None)
            self.events.publish(f"task.{target.value}", task.task_id, {
                "task_type": task.task_type, "attempts": attempts, "version": finished.version, "error": error,
            })
        return finished

    def run_until_idle(self, *, max_tasks: int | None = None) -> tuple[TaskSnapshot, ...]:
        if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 0):
            raise ValueError("max_tasks must be a non-negative integer or None")
        results = []
        while max_tasks is None or len(results) < max_tasks:
            item = self.run_next()
            if item is None:
                break
            results.append(item)
        return tuple(results)

    def get_task(self, task_id: str) -> TaskSnapshot:
        return self.state.get(task_id)

    def health_snapshot(self) -> RuntimeHealthSnapshot:
        return RuntimeMonitor.snapshot(
            self.state.all(), queue_depth=self.scheduler.depth,
            queue_capacity=self.scheduler.capacity, retained_events=len(self.events.history),
            event_delivery_failures=len(self.events.delivery_failures),
        )
