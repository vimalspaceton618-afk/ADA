# ADA Core Runtime — Implementation Record

## Objective

Implement a CPU-first runtime foundation in the architectural source tree 02_CORE_RUNTIME. Runtime orchestration is separate from neural model layers, model weights, and inference kernels.

## Implemented source map

| Subsystem | Canonical source | Implementation |
|---|---|---|
| Contracts | contracts.py | Immutable task/event contracts, lifecycle enum, typed errors, JSON-safe deep freezing |
| Task scheduler | task_scheduler/fifo.py | Thread-safe bounded FIFO, dequeue, cancellation removal, backpressure |
| State manager | state_manager/store.py | Versioned snapshots, guarded lifecycle transitions, execution-attempt accounting |
| Execution engine | execution_engine/registry.py | Handler registry, duplicate protection, explicit resolution |
| Event system | event_system/bus.py | Sequenced events, bounded history, subscription filters, isolated delivery failures |
| Communication bus | communication_bus/messages.py | Typed message envelopes, topic pub/sub, correlation identifiers |
| Concurrency | concurrency/primitives.py | Cancellation token and bounded semaphore capacity limiter |
| Fault recovery | fault_recovery/policy.py | Explicit retryable exception classes, bounded attempts, capped exponential backoff |
| Process manager | process_manager/supervisor.py | Owned subprocess tracking, allowlisted executables, graceful termination and forced stop |
| Runtime monitor | runtime_monitor/health.py | Queue health, task-state counts, average execution duration, event failures |
| Orchestrator | runtime.py | Submission, FIFO dispatch, lifecycle coordination, cooperative cancellation/deadlines, opt-in retries |

## Public API and package layout

- Canonical implementation: 02_CORE_RUNTIME/
- Stable import surface: ada.runtime
- ada/runtime/ contains compatibility facades that load canonical modules through importlib.
- pyproject.toml includes both package trees in built distributions.
- The numeric architecture directory is intentionally preserved as the source location; normal application code should import from ada.runtime.

## Runtime behavior

Task lifecycle:

    CREATED -> QUEUED -> RUNNING -> SUCCEEDED
                              |-> FAILED
                              |-> CANCELLED
                              |-> TIMED_OUT

The runtime is caller-driven. submit() enqueues work; run_next() executes one task; run_until_idle() drains work. No background worker is silently created.

Task inputs, successful outputs, events, and component messages accept JSON-compatible values and are deeply frozen. Invalid state transitions and duplicate task IDs are rejected.

Cancellation and timeout behavior is cooperative for running handlers. Handlers must call TaskContext.checkpoint() at safe points. A non-cooperative blocking call cannot be forcibly interrupted inside a Python thread.

Retries are disabled by default. A task type must be configured with a RetryPolicy that explicitly names retryable exception classes. Backoff is bounded and cancellation is observed during the wait.

ProcessSupervisor starts no process by itself. Callers must supply an explicit executable allowlist. It never uses shell=True and can stop only processes it owns. It is not yet wired as an automatic task execution backend.

## Upstream provenance and licensing gate

The implementation in this phase is an interface/behavior implementation authored for ADA; it does not vendor Mamba source, model weights, tokenizer assets, or third-party kernel code.

Before adapting upstream source in a later phase:
- Inspect the exact repository LICENSE and file-level headers.
- Record upstream project, URL, revision, SPDX license, and modified files.
- Preserve copyright and attribution notices.
- Verify compatibility with ADA's repository license and redistribution plan.
- Prefer a clean-room implementation if provenance or license compatibility is unclear.

## Verification status

The implementation files have been committed directly to main. Per the implementation instruction for this phase, no basic test run was requested or performed for this file-writing pass. Do not treat the existence of prior runtime CI runs as verification of this newer canonical-source migration.

## Explicit non-goals

- No distributed scheduler or multi-node execution.
- No durable event/message broker.
- No asynchronous worker pool or background dispatcher.
- No automatic process-backed task handler adapter.
- No GPU/CUDA/Triton scheduling or model execution.
- No production throughput or hard real-time guarantees.
- Neural computation remains under 09_NEURAL_SUBSTRATE; inference adapters remain under 18_INFERENCE.
