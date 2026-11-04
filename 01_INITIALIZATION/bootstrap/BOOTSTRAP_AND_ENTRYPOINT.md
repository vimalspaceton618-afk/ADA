# Bootstrap and Entrypoint Contract

**Status:** Proposed implementation contract

## Purpose

Create one controlled entrypoint for training, evaluation, inference and maintenance. Notebook cells are exploratory; they MUST NOT be treated as a production bootstrap sequence.

## Required order

1. Parse command and profile path.
2. Resolve configuration sources and record provenance.
3. Validate schema, ranges, cross-field constraints and implementation compatibility.
4. Establish logging and a unique run ID.
5. Seed Python, NumPy and the selected framework where available.
6. Detect operating system, Python and framework runtime.
7. Verify declared dependencies without installing packages.
8. Discover devices and collect capability metadata.
9. Estimate workload memory and request resource admission.
10. Verify tokenizer, dataset manifests and checkpoint metadata when required.
11. Run startup health probes.
12. Construct immutable RuntimeContext and transition to READY.

Any failed required step stops startup. Cleanup must run for partially acquired resources.

## Proposed command surface

```text
ada doctor --profile profiles/ADA_V45_RESEARCH.yaml
ada config validate --profile <path>
ada train --config <path>
ada evaluate --config <path>
ada infer --config <path>
ada shutdown --run-id <id>
```

This command surface is a design proposal, not an implemented CLI.

## RuntimeContext contract

The context MUST carry run ID, UTC start time, command/profile; validated config and digest; software manifest; hardware inventory and selected backend/device; resource admission and memory headroom; seed/determinism policy; tokenizer/model/data/checkpoint identities when applicable; health state, log path and cleanup registry.

Modules consume the same recorded inventory rather than independently re-detecting hardware.

## Reproducibility

- Require explicit seeds for experiment profiles; generate and record one only when policy permits.
- Seed Python, NumPy and PyTorch independently.
- Record deterministic-algorithm settings and known nondeterministic operations.
- For JAX, record PRNG key lineage.
- Record distributed rank, world size, sampler state and data cursor for resumable training.
- Never claim strict determinism merely because random seeds were set.

## Failure behavior

Errors use stable codes, stage name, retryability and remediation. Do not swallow exceptions and continue with partially initialized globals. Broad except-and-print blocks MUST NOT be primary control flow.

## Notebook migration

Notebook experiments may be imported into package modules only after removing Colab magics and Drive paths; removing repeated class redefinitions and interactive loops; fixing dependency/device assumptions; adding type/shape checks and automated tests; and recording source notebook digest.
