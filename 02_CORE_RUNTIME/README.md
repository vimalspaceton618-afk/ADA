# ADA Core Runtime

This directory defines the architectural boundary and subsystem map for ADA's
CPU-first runtime. The importable implementation lives in ada/runtime/ because
a directory beginning with a numeric prefix is not a valid Python package name.

## Implemented foundation

| Runtime responsibility | Implementation | Status |
|---|---|---|
| Task contracts and lifecycle | ada/runtime/contracts.py, state.py | Implemented |
| FIFO dispatch and bounded queue | ada/runtime/scheduler.py | Implemented |
| Synchronous handler execution | ada/runtime/runtime.py | Implemented |
| Cooperative cancellation and deadlines | TaskContext in runtime.py | Implemented |
| Immutable lifecycle events | ada/runtime/events.py | Implemented |
| In-process event publication | InMemoryEventBus in events.py | Implemented |
| Health summary and counters | ada/runtime/monitor.py | Implemented |
| Process isolation / worker supervision | process_manager/ | Deferred |
| Retry orchestration and backoff | fault_recovery/ | Deferred |
| Async/distributed scheduling | concurrency/, scheduler extensions | Deferred |

## Minimal usage

    from ada.runtime import Runtime, TaskSpec

    runtime = Runtime(queue_capacity=128)

    def summarize(payload, context):
        context.checkpoint()
        return {"summary": f"items={len(payload['items'])}"}

    runtime.register_handler("summarize", summarize)
    accepted = runtime.submit(
        TaskSpec(
            task_type="summarize",
            payload={"items": ["a", "b", "c"]},
            task_id="summary-001",
            timeout_seconds=5.0,
        )
    )
    completed = runtime.run_next()

submit() accepts work into a bounded in-memory FIFO queue. run_next() is
explicitly caller-driven; this initial runtime does not start background workers.

## Lifecycle

    CREATED -> QUEUED -> RUNNING -> SUCCEEDED
                              |-> FAILED
                              |-> CANCELLED
                              |-> TIMED_OUT

Every accepted task has one terminal state. Invalid transitions are rejected.
The state store returns immutable snapshots; task inputs, event details, and
successful results are restricted to JSON-compatible values and deeply frozen.

## Cancellation and timeouts

Queued tasks can be cancelled immediately. Running tasks receive a thread-safe
cancellation request and must call context.checkpoint() to observe it.
Timeouts use the same cooperative checkpoint boundary. A synchronous Python
handler blocked in a non-cooperative operation cannot be forcibly interrupted;
the runtime reports timeout after control returns or when a checkpoint is hit.
Use process isolation in a later phase for hard termination boundaries.

## Event delivery

Events are immutable, monotonically sequenced, and retained in a bounded
in-memory history. Subscribers run synchronously outside the bus lock. A
subscriber exception is recorded and isolated so it does not prevent delivery
to other subscribers or corrupt task state. This is an in-process mechanism,
not a durable message broker.

## Verification

The runtime tests cover lifecycle legality, immutable contracts, FIFO order,
queue backpressure, task failure isolation, missing handlers, invalid handler
results, queued/running cancellation, timeout distinction, bounded event
history, subscriber isolation, and bounded draining.

Run locally:

    python -m pip install -e ".[dev]"
    python -m pytest -q

## Deliberate non-goals

This foundation does not claim distributed execution, process supervision,
hard-kill timeouts, asynchronous dispatch, GPU scheduling, model loading,
Mamba/ADA-TLA execution, or production-scale throughput. Neural computation
and inference kernels remain separate from runtime orchestration.
