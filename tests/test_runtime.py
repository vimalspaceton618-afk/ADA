from __future__ import annotations

import threading
import time

import pytest

from ada.runtime import (
    DuplicateTaskError,
    InMemoryEventBus,
    InvalidTransitionError,
    QueueFullError,
    Runtime,
    RuntimeStateStore,
    TaskSpec,
    TaskStatus,
)


def test_task_payload_is_deeply_immutable() -> None:
    task = TaskSpec(task_type="echo", payload={"nested": {"items": [1, 2]}})
    with pytest.raises(TypeError):
        task.payload["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        task.payload["nested"]["new"] = "value"  # type: ignore[index]
    assert task.payload["nested"]["items"] == (1, 2)


def test_task_spec_rejects_non_json_payload_and_bad_timeout() -> None:
    with pytest.raises(TypeError, match="JSON-compatible"):
        TaskSpec(task_type="bad", payload={"value": object()})
    with pytest.raises(ValueError, match="positive finite"):
        TaskSpec(task_type="bad", timeout_seconds=float("inf"))
    with pytest.raises(TypeError, match="positive finite"):
        TaskSpec(task_type="bad", timeout_seconds=True)


def test_state_store_rejects_illegal_and_repeated_terminal_transitions() -> None:
    store = RuntimeStateStore()
    store.create(TaskSpec(task_type="echo", task_id="state-1"))
    with pytest.raises(InvalidTransitionError):
        store.transition("state-1", TaskStatus.RUNNING)
    store.transition("state-1", TaskStatus.QUEUED)
    store.transition("state-1", TaskStatus.RUNNING)
    store.transition("state-1", TaskStatus.SUCCEEDED, result={"ok": True})
    with pytest.raises(InvalidTransitionError):
        store.transition("state-1", TaskStatus.FAILED, error="late failure")


def test_queue_capacity_rejects_without_leaving_partial_task() -> None:
    runtime = Runtime(queue_capacity=1)
    first = runtime.submit(TaskSpec(task_type="echo", task_id="queue-1"))
    assert first.status is TaskStatus.QUEUED

    with pytest.raises(QueueFullError):
        runtime.submit(TaskSpec(task_type="echo", task_id="queue-2"))

    assert runtime.scheduler.depth == 1
    assert not runtime.state.contains("queue-2")
    assert runtime.get_task("queue-1").status is TaskStatus.QUEUED


def test_fifo_execution_and_success_event_trace() -> None:
    runtime = Runtime()
    seen: list[int] = []
    events: list[str] = []
    runtime.register_handler(
        "collect", lambda payload, context: seen.append(payload["value"]) or payload["value"]
    )
    runtime.events.subscribe(lambda event: events.append(event.event_type))

    runtime.submit(TaskSpec("collect", {"value": 10}, task_id="fifo-1"))
    runtime.submit(TaskSpec("collect", {"value": 20}, task_id="fifo-2"))
    completed = runtime.run_until_idle()

    assert seen == [10, 20]
    assert [task.task_id for task in completed] == ["fifo-1", "fifo-2"]
    assert all(task.status is TaskStatus.SUCCEEDED for task in completed)
    assert completed[0].result == 10
    assert events == [
        "task.queued",
        "task.queued",
        "task.running",
        "task.succeeded",
        "task.running",
        "task.succeeded",
    ]
    assert runtime.scheduler.depth == 0


def test_duplicate_task_id_is_rejected() -> None:
    runtime = Runtime()
    task = TaskSpec("echo", task_id="duplicate")
    runtime.submit(task)
    with pytest.raises(DuplicateTaskError):
        runtime.submit(task)


def test_handler_failure_is_recorded_and_does_not_block_next_task() -> None:
    runtime = Runtime()

    def fail(payload, context):
        raise RuntimeError("simulated failure")

    runtime.register_handler("fail", fail)
    runtime.register_handler("ok", lambda payload, context: {"done": True})
    runtime.submit(TaskSpec("fail", task_id="failure"))
    runtime.submit(TaskSpec("ok", task_id="success"))

    first = runtime.run_next()
    second = runtime.run_next()

    assert first is not None and first.status is TaskStatus.FAILED
    assert first.error == "RuntimeError: simulated failure"
    assert second is not None and second.status is TaskStatus.SUCCEEDED
    assert second.result["done"] is True


def test_missing_handler_becomes_failed_task() -> None:
    runtime = Runtime()
    runtime.submit(TaskSpec("unregistered", task_id="missing-handler"))
    snapshot = runtime.run_next()
    assert snapshot is not None
    assert snapshot.status is TaskStatus.FAILED
    assert snapshot.error is not None
    assert "HandlerNotFoundError" in snapshot.error


def test_non_json_handler_result_becomes_failed_task() -> None:
    runtime = Runtime()
    runtime.register_handler("bad-result", lambda payload, context: object())
    runtime.submit(TaskSpec("bad-result", task_id="invalid-result"))
    snapshot = runtime.run_next()
    assert snapshot is not None
    assert snapshot.status is TaskStatus.FAILED
    assert "JSON-compatible" in (snapshot.error or "")


def test_queued_task_can_be_cancelled_without_execution() -> None:
    runtime = Runtime()
    runtime.register_handler("echo", lambda payload, context: payload)
    runtime.submit(TaskSpec("echo", task_id="cancel-queued"))

    assert runtime.cancel("cancel-queued") is True
    assert runtime.get_task("cancel-queued").status is TaskStatus.CANCELLED
    assert runtime.scheduler.depth == 0
    assert runtime.run_next() is None
    assert runtime.cancel("cancel-queued") is False


def test_running_task_observes_cooperative_cancellation() -> None:
    runtime = Runtime()
    started = threading.Event()
    release = threading.Event()

    def cooperative_handler(payload, context):
        started.set()
        assert release.wait(timeout=2)
        context.checkpoint()
        return {"unexpected": "completion"}

    runtime.register_handler("cooperative", cooperative_handler)
    runtime.submit(TaskSpec("cooperative", task_id="cancel-running"))
    results = []
    worker = threading.Thread(target=lambda: results.append(runtime.run_next()))
    worker.start()
    assert started.wait(timeout=1)
    assert runtime.cancel("cancel-running") is True
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert results[0] is not None
    assert results[0].status is TaskStatus.CANCELLED


def test_timeout_is_reported_as_distinct_terminal_state() -> None:
    runtime = Runtime()
    runtime.register_handler("slow", lambda payload, context: time.sleep(0.02) or "late")
    runtime.submit(TaskSpec("slow", task_id="timeout", timeout_seconds=0.001))
    snapshot = runtime.run_next()
    assert snapshot is not None
    assert snapshot.status is TaskStatus.TIMED_OUT
    assert "exceeded its time limit" in (snapshot.error or "")


def test_event_bus_is_bounded_immutable_and_isolates_subscriber_errors() -> None:
    bus = InMemoryEventBus(max_history=2)
    received = []

    def bad_handler(event):
        raise RuntimeError("subscriber failed")

    bus.subscribe(bad_handler)
    bus.subscribe(received.append)
    first = bus.publish("task.queued", "event-1", {"nested": {"value": 1}})
    bus.publish("task.running", "event-1")
    bus.publish("task.succeeded", "event-1")

    assert len(bus.history) == 2
    assert len(bus.delivery_failures) == 3
    assert len(received) == 3
    with pytest.raises(TypeError):
        first.details["nested"]["value"] = 2  # type: ignore[index]


def test_runtime_drain_limit_and_empty_queue() -> None:
    runtime = Runtime()
    runtime.register_handler("echo", lambda payload, context: payload)
    for index in range(3):
        runtime.submit(TaskSpec("echo", {"index": index}, task_id=f"limit-{index}"))

    first = runtime.run_until_idle(max_tasks=2)
    assert len(first) == 2
    assert runtime.scheduler.depth == 1
    last = runtime.run_until_idle()
    assert len(last) == 1
    assert runtime.run_next() is None
    with pytest.raises(ValueError, match="max_tasks"):
        runtime.run_until_idle(max_tasks=-1)
