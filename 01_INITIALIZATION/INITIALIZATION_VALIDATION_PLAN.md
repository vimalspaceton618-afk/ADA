# Initialization Validation Plan

**Status:** Acceptance plan; tests not yet implemented by this document

## Gate I0 — Configuration correctness
- Valid config parses and canonicalizes.
- Invalid types, unknown required keys and out-of-range values are rejected.
- GQA divisibility and MoE top-k constraints are tested.
- Tokenizer/model vocabulary mismatch is rejected.
- Unsupported research features are rejected for executable profiles.

## Gate I1 — Environment discovery
- CPU-only host is identified accurately.
- CUDA, ROCm and XLA backends are not conflated.
- Unavailable capability returns structured unknown/unavailable state.
- Detection does not leak secrets or require network access.

## Gate I2 — Resource admission
- Weight/gradient/optimizer/activation/workspace estimates are reported separately.
- Low-memory device is rejected for incompatible workload.
- Safety reserve is honored.
- No silent precision or dimension changes occur.

## Gate I3 — Bootstrap lifecycle
- Failure at every stage invokes cleanup.
- Startup is repeatable and idempotent where promised.
- Run ID and config digest are stable within one run.
- Cancellation before READY reaches a terminal state.

## Gate I4 — Health and artifact integrity
- Corrupt checkpoint and mismatched config are rejected.
- Missing optional dependency produces WARN only when optional.
- Required kernel probe failure blocks the selected profile.
- Causal, scan, routing and gradient probes link to architecture-level tests.

## Gate I5 — Shutdown and recovery
- Interrupt triggers bounded shutdown.
- Checkpoint is atomic and verified.
- Resume restores optimizer, scheduler, RNG and data position.
- Failed checkpoint writes preserve previous valid checkpoint.

## Notebook reconciliation

The notebook demonstrated experimental code and small synthetic correctness checks, particularly for TensorNet linear-attention scan equivalence and gradient flow. It also contains multiple historical model definitions, notebook dependencies, inconsistent checkpoint/model variables, and simplified blocks that are not the paper's production architecture.

Initialization tests validate selected implementation by identity and contract, not by class name alone.

## Release criterion

This directory is ready for implementation review when configuration schemas, machine-readable profiles, health probes and lifecycle transitions have executable tests. It is not implemented merely because specifications exist.
