# Startup Sequence

**Status:** Proposed process state machine

## States

`CREATED → CONFIGURING → VALIDATING → DISCOVERING → ADMITTING → PREPARING → HEALTH_CHECKING → READY`

Any required-stage failure transitions to REJECTED or FAILED, then cleanup. Cancellation at any point transitions to STOPPING.

## Stage responsibilities

| Stage | Work | Must not do |
|---|---|---|
| Created | create run identity and trace | allocate model or dataset |
| Configuring | resolve profile and overrides | fetch arbitrary remote config |
| Validating | schema and architecture contract | mutate config silently |
| Discovering | host, framework and accelerator probes | infer unsupported capability |
| Admitting | memory/cost/resource decision | assume all VRAM is free |
| Preparing | create directories, load declared metadata | load unverified checkpoint |
| Health checking | smoke probes and integrity checks | launch expensive training |
| Ready | publish immutable context | mutate run identity |

## Resource acquisition

Register every acquired resource with a cleanup handler. Acquisitions should be idempotent where possible. Directory creation uses explicit configured paths and safe permissions. Distributed process groups, device contexts and temporary files are released on failed startup.

## Readiness

READY means the process is safe to start the selected command; it does not mean the model is trained, capable, benchmarked or deployment-ready. Training has its own data, model, optimizer and evaluation checks.
