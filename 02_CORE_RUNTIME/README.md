# ADA Core Runtime

02_CORE_RUNTIME/ is the canonical source tree for ADA's CPU-first orchestration runtime. The stable Python import surface is ada.runtime; its facade delegates to implementations in this directory. Runtime orchestration remains independent of neural layers, model weights, and inference kernels.

## Architecture

| Subsystem | Source | Responsibility |
|---|---|---|
| Contracts | contracts.py | Immutable task/event records, JSON-safe payloads, lifecycle states and typed errors |
| Task scheduler | task_scheduler/fifo.py | Bounded FIFO queue, deterministic dequeue, backpressure |
| State manager | state_manager/store.py | Thread-safe task state machine, versioned snapshots, attempt tracking |
| Execution engine | execution_engine/registry.py | Handler registration, lookup and duplicate protection |
| Event system | event_system/bus.py | Sequenced lifecycle events, bounded history, subscriber isolation |
| Communication bus | communication_bus/messages.py | Typed component messages, topic pub/sub, correlation IDs |
| Concurrency | concurrency/primitives.py | Cancellation token and bounded capacity limiter |
| Fault recovery | fault_recovery/policy.py | Explicit retry classification and bounded exponential backoff |
| Process manager | process_manager/supervisor.py | Owned subprocess lifecycle, graceful stop then kill, executable allowlist |
| Runtime monitor | runtime_monitor/health.py | Immutable queue, lifecycle, timing and event health snapshots |
| Orchestrator | runtime.py | Submission, dispatch, cancellation, retries and terminal-state coordination |

## Quick start

    from ada.runtime import Runtime, TaskSpec

    runtime = Runtime(queue_capacity=128)

    def summarize(payload, context):
        context.checkpoint()
        return {"count": len(payload["items"])}

    runtime.register_handler("summarize", summarize)
    runtime.submit(
        TaskSpec(
            task_type="summarize",
            payload={"items": ["a", "b", "c"]},
            task_id="summary-001",
            timeout_seconds=10.0,
        )
    )
    completed = runtime.run_next()
    print(completed.status, completed.result)

submit() validates and queues a task. run_next() executes exactly one task; run_until_idle() drains queued work. Dispatch is caller-driven: constructing a runtime does not create background workers.

## Lifecycle contract

    CREATED -> QUEUED -> RUNNING -> SUCCEEDED
                              |-> FAILED
                              |-> CANCELLED
                              |-> TIMED_OUT

Accepted tasks are transitioned through a guarded state machine. Terminal states cannot be reopened. Task payloads, successful results, event details and message payloads accept JSON-compatible data and are deeply frozen.

## Cancellation and deadlines

Queued tasks can be cancelled immediately. A running handler receives a TaskContext; it must call context.checkpoint() at safe interruption points to observe cancellation or deadline expiry. A synchronous handler blocked in a non-cooperative call cannot be interrupted by Python thread cancellation. Use a supervised subprocess boundary for work requiring hard termination.

## Retry semantics

Retries are disabled unless a task type is given an explicit RetryPolicy with a non-empty tuple of retryable exception classes. Retry attempts are bounded and backoff is capped. Cancellation and timeout are terminal outcomes and are not retried. There is no implicit retry of arbitrary failures.

## Process supervision and security

ProcessSupervisor only controls processes it starts. It never invokes a shell. It requires an explicit allowed_executables allowlist; an empty allowlist denies all process launches. Environment overrides are explicit. Stop sends a graceful termination request, waits for the configured grace period, and then kills the owned process if necessary.

The supervisor is a low-level component and is not automatically invoked by the task runtime. A later process-backed execution adapter must define serialization, output capture, resource limits, and worker isolation before it is used for task handlers.

## Communication and events

Lifecycle events and component messages are separate channels:
- InMemoryEventBus publishes runtime lifecycle events with monotonically increasing sequence numbers.
- InMemoryMessageBus transports typed messages by topic and supports correlation IDs.

Both are bounded, in-memory, synchronous transports. Subscriber failures are retained as diagnostics and isolated from other subscribers. Neither is a durable broker.

## Public import compatibility

Use ada.runtime for normal Python imports. The package facade loads the canonical implementation from this directory using importlib, allowing the repository's numbered architecture folder to remain the source-of-truth location. pyproject.toml includes both package trees in built distributions.

## Operational boundaries

This implementation is CPU-first and has no mandatory ML dependency. It does not claim distributed scheduling, durable messaging, async worker pools, GPU scheduling, model execution, production throughput, or hard real-time guarantees. Neural computation belongs under 09_NEURAL_SUBSTRATE; inference adapters belong under 18_INFERENCE.
