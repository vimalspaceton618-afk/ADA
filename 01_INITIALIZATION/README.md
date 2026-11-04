# 01 — Initialization

**Status:** Engineering specification and bootstrap contracts  
**Maturity:** Design baseline; implementation pending unless a specific file is marked executable  
**Scope:** Establish a reproducible, resource-aware, diagnosable process before any ADA workload starts.

Initialization is the boundary between an arbitrary host environment and a controlled ADA process. It MUST discover capabilities, load and validate configuration, verify dependencies, estimate resource requirements, prepare state, and either hand off a valid runtime context or stop with an actionable error.

This directory does **not** implement the neural architecture, training loop, inference engine, or WSPPI. It prepares the environment in which those systems may later run.

## Directory map

| Area | Responsibility | Primary document |
|---|---|---|
| bootstrap | Entry-point order, process context, reproducibility | [BOOTSTRAP_AND_ENTRYPOINT.md](bootstrap/BOOTSTRAP_AND_ENTRYPOINT.md) |
| configuration | Typed configuration, profiles, validation | [CONFIGURATION_SPECIFICATION.md](configuration/CONFIGURATION_SPECIFICATION.md) |
| dependency_manager | Runtime dependency checks and environment locking | [DEPENDENCY_AND_BOOTSTRAP_SPEC.md](dependency_manager/DEPENDENCY_AND_BOOTSTRAP_SPEC.md) |
| environment_detection | OS, Python, framework, accelerator capability detection | [ENVIRONMENT_DETECTION_SPEC.md](environment_detection/ENVIRONMENT_DETECTION_SPEC.md) |
| hardware_discovery | CPU, memory, GPU/TPU, interconnect and capability inventory | [HARDWARE_DISCOVERY_SPEC.md](hardware_discovery/HARDWARE_DISCOVERY_SPEC.md) |
| resource_allocator | Conservative memory budgets and workload admission | [RESOURCE_ALLOCATION_POLICY.md](resource_allocator/RESOURCE_ALLOCATION_POLICY.md) |
| health_monitor | Startup and runtime readiness probes | [INITIALIZATION_HEALTH_CHECKS.md](health_monitor/INITIALIZATION_HEALTH_CHECKS.md) |
| startup_sequence | Ordered initialization state machine | [STARTUP_SEQUENCE.md](startup_sequence/STARTUP_SEQUENCE.md) |
| shutdown_sequence | Cancellation, checkpoint flush, release and exit | [SHUTDOWN_SEQUENCE.md](shutdown_sequence/SHUTDOWN_SEQUENCE.md) |
| Root | Integrated acceptance criteria and research profile | [INITIALIZATION_VALIDATION_PLAN.md](INITIALIZATION_VALIDATION_PLAN.md), [ADA_V45_RESEARCH.yaml](profiles/ADA_V45_RESEARCH.yaml) |

## Initialization contract

A successful initialization returns an immutable RuntimeContext containing validated configuration and digest; environment and hardware inventory; dependency/runtime report; resource admission decision; run ID and seed policy; and health status.

Initialization MUST fail closed for invalid configuration, incompatible checkpoint metadata, missing required packages, insufficient resources, failed integrity checks, and unsupported device/backend combinations. It MUST NOT silently change model dimensions, precision, tokenizer, sequence length, or architecture to make a workload fit.

## Evidence discipline

The July 2026 ADA v4.5 research paper specifies a proposed production configuration. The uploaded Colab notebook is a separate history of experiments: it contains simplified early FNY-ADA blocks and later TensorNet linear-attention, scan, memory and MoE prototypes. Those are not one interchangeable implementation.

The profile records paper settings as a **research target**, not a ready-to-run production configuration. Initialization MUST reject it when the selected implementation does not declare support for requested components.

## Non-goals

- No automatic model download or remote code execution.
- No implicit package installation during production startup.
- No automatic GPU/TPU selection based only on device availability.
- No claim that a detected accelerator can fit a particular training job.
- No silent fallback from BF16 to FP16/FP32 or CUDA to CPU.
- No training, online learning, weight mutation, or artifact promotion during initialization.

## Startup result

- READY: all required gates passed and a runtime context was created.
- DEGRADED: only explicitly optional capabilities are unavailable and the selected profile permits them.
- REJECTED: workload cannot safely start.
- FAILED: initialization itself failed unexpectedly.

See [00_FOUNDATION](../00_FOUNDATION/README.md) for decision hierarchy and evidence requirements.
