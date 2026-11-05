# ADA Core Runtime — Implementation Plan

## Objective

Turn `02_CORE_RUNTIME` from an architectural directory scaffold into a tested, Python-first runtime foundation. This phase deliberately does not implement the ADA neural architecture or claim production-scale scheduling, distributed execution, or GPU acceleration.

## Engineering principles

- Keep runtime orchestration separate from neural model layers and inference kernels.
- Fail closed: reject invalid tasks, transitions, and configuration rather than silently guessing.
- Make lifecycle transitions explicit and observable.
- Keep the first implementation CPU-compatible and testable on Python 3.11+.
- Use narrow interfaces so later backends (asyncio, process pools, distributed workers, PyTorch, CUDA) can be added without changing core contracts.
- Avoid mandatory heavyweight ML dependencies in the runtime foundation.

## Proposed subsystem boundaries

| Subsystem | Responsibility | Initial implementation |
|---|---|---|
| `task_scheduler` | Queueing and dispatch policy | Deterministic in-memory FIFO scheduler contract |
| `execution_engine` | Execute submitted work | Typed task runner with cancellation and timeout boundaries |
| `state_manager` | Runtime/task state | Explicit state machine and transition validation |
| `event_system` | Lifecycle events | Immutable event records and bounded in-process event dispatch |
| `communication_bus` | Component messages | Typed envelopes and in-process pub/sub |
| `concurrency` | Coordination primitives | Async-safe locks/semaphores only where required |
| `process_manager` | Worker lifecycle | Deferred until process isolation is required |
| `fault_recovery` | Retry/failure policy | Bounded retry policy with explicit retryable errors |
| `runtime_monitor` | Health and diagnostics | Structured counters/timing hooks; no vendor lock-in |

## Hugging Face / upstream reference register

These are reference implementations and model/config artifacts to inspect. They are not runtime dependencies by default and are not to be copied wholesale.

1. `state-spaces/mamba-130m-hf` — https://huggingface.co/state-spaces/mamba-130m-hf  
   Its model card describes Transformers-compatible loading/generation and optional `causal-conv1d` / `mamba-ssm` optimized kernels. Its config illustrates Mamba-1 dimensions and cache-related settings. The card contains an apparent model-name inconsistency in its prose; verify source files rather than treating the description as authoritative.
2. `state-spaces/mamba2-130m` — https://huggingface.co/state-spaces/mamba2-130m  
   Its config records a Mamba2 layer selection and explicit attention-layer indices. It is a configuration reference, not a source-code implementation.
3. `kernels-community/mamba-ssm` — https://huggingface.co/kernels-community/mamba-ssm  
   Hub card lists entry points such as `selective_scan_fn`, `selective_state_update`, `mamba_chunk_scan_combined`, and `Mamba2`; card states no benchmark is available. It is a kernel reference only; inspect source and license before considering integration.

## Licensing and provenance gate

Before copying or adapting any upstream source:
- Inspect the exact source repository's LICENSE and file-level headers.
- Record upstream project, URL, commit/revision, SPDX license, and modified files.
- Preserve copyright and attribution notices.
- Ensure compatibility with ADA's repository license and redistribution plan.
- Prefer clean-room interface/behavior reimplementation when licensing is unclear.
- Do not copy model weights or tokenizer assets into this repository as part of runtime code.

## Proposed implementation sequence

1. **Runtime contracts:** task envelope, task identifiers, lifecycle enum, state transition validator, runtime exceptions.
2. **Events:** immutable event schema, event sink protocol, in-memory implementation, ordering and isolation tests.
3. **Scheduler:** FIFO in-memory queue, bounded capacity, deterministic dequeue, backpressure behavior.
4. **Execution:** handler registry, safe dispatch, cancellation and timeout semantics, exception propagation.
5. **Fault recovery:** bounded retries with exponential backoff policy abstraction; no automatic retry for non-retryable errors.
6. **Monitoring:** structured runtime metrics hooks and health snapshot.
7. **Integration:** end-to-end tests proving accepted task → queued → running → terminal state, with event trace.
8. **Optional model adapter:** only after runtime contracts stabilize; Mamba/ADA-TLA execution is owned by a separate model-kernel layer.

## Acceptance criteria

- Unit tests cover valid and invalid lifecycle transitions.
- Queue capacity/backpressure behavior is deterministic.
- Each accepted task reaches exactly one terminal state.
- A failed task does not corrupt unrelated tasks.
- Cancellation and timeout are distinguishable outcomes.
- Event consumers cannot mutate published event records.
- No hidden global mutable state.
- Tests run on Python 3.11 and 3.12.
- All upstream-derived code, if any, has documented provenance and compatible licensing.

## Explicit non-goals for this phase

- No claim of ADA Mamba/Mamba3 implementation.
- No claim of ADA-TLA kernel implementation.
- No GPU/CUDA/Triton performance claims.
- No distributed scheduler or multi-node execution.
- No autonomous retry loops without policy and observability.
- No model-weight loading or generation API until the model interface is specified.
