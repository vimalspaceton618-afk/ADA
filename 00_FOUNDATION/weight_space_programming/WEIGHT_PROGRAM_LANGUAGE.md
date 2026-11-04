# Weight Program Language (WPL)

**Status:** Proposed declarative interface | **Version:** Draft 0.1

## 1. Purpose

WPL describes what parameter transformation is requested without allowing arbitrary executable code. A trusted compiler maps a small, reviewed operation vocabulary to implementations. WPL is not a general-purpose programming language.

## 2. Program schema

A program includes identity/version, base artifact, objective, task distribution, ordered operations, tensor selectors, preconditions, constraints, resource budget, required evaluations, proposer, and approval requirements.

Illustrative YAML (schema draft; not executable configuration):

~~~yaml
api_version: wspi.ada/v0
program:
  id: adapter-composition-study-001
  version: 1
  base_artifact: ada-base@sha256:<digest>
  objective:
    description: "Evaluate bounded coding adapter composition"
    target_tasks: ["heldout-coding-evaluation"]
    success_metrics:
      - metric: task_success
        direction: maximize
      - metric: protected_regression_rate
        direction: no_worse_than
  operations:
    - op: apply_adapter
      source: coding-adapter@sha256:<digest>
      selector:
        include: ["blocks.*.adapter_*"]
      parameters:
        scale: 0.35
  constraints:
    max_relative_delta_norm: 0.02
    preserve_base_artifact: true
    forbid_unregistered_operations: true
  budget:
    max_runtime_minutes: 30
    max_memory_gb: 16
  evaluations:
    required:
      - coding-heldout-v1
      - general-regression-v1
      - safety-regression-v1
  approval:
    promotion_requires_human: true
~~~

Values are illustrative. Real schemas must validate ranges, compatibility, and artifact identities.

## 3. Tensor selector rules

Selectors resolve against the model manifest, never arbitrary runtime object traversal. The compiler must report all matched tensors and shapes, reject empty or ambiguous matches, enforce protected regions, and record the resolved selector set.

## 4. Operation families

### Tier 0 — inspection and integrity
- inspect_tensor_stats
- compute_delta_stats
- verify_artifact

These operations do not modify weights and should be implemented first.

### Tier 1 — bounded transformations
- apply_adapter: apply a compatible adapter or low-rank update.
- scale_delta: multiply an approved parameter delta by a bounded scalar.
- interpolate: interpolate compatible tensors under explicit alignment and validation.
- compose_adapters: compose adapters only when targets and semantics are compatible.
- mask_parameters: apply a declared mask with sparsity and protected-region constraints.

### Tier 2 — advanced research operations
- merge_checkpoints: only with architecture and parameter correspondence checks; alignment may be required.
- reparameterize: change representation while preserving declared function or behavior within measured tolerance.
- distill_to_artifact: train a student against a declared teacher and data manifest; this is a training job, not simple tensor arithmetic.
- generate_delta: create a delta from an approved training or optimization run.
- quantize_artifact: change numeric representation with accuracy and runtime evaluation.

Tier 2 requires separate implementation review.

## 5. Prohibited by default

- Arbitrary Python, shell, or notebook execution from a weight program.
- Unbounded tensor writes or wildcard writes to all parameters.
- Direct production artifact overwrite.
- Loading remote or unverified custom operations.
- Deleting source checkpoints or audit records.
- Hiding evaluation failures.
- Editing protected policy components without dedicated review.

## 6. Numerical constraints

For a parameter delta, report relative update magnitude:

[
r = \frac{\|\Delta\theta\|_2}{\|\theta\|_2 + \epsilon}
]

This is only a coarse diagnostic, not a safety or quality guarantee. Per-layer and per-module changes must also be inspected because a small global norm can hide a large local change.

Operations must define dtype behavior, accumulation precision, finite-value checks, clipping where relevant, and numerical tolerance.

## 7. Compilation pipeline

1. Parse syntax and schema.
2. Resolve base artifact and verify digest.
3. Resolve selectors against manifest.
4. Type-check operations and dimensions.
5. Validate preconditions and constraints.
6. Estimate resources and risk.
7. Produce immutable compiled WIR.
8. Integrity-protect the WIR.
9. Dispatch to an isolated worker.

The compiler never executes model-generated code.

## 8. Reproducibility and evolution

Record operation implementation versions, parameters, selector expansion, source digests, seeds, framework versions, precision, and execution environment. Unknown operations fail closed. Breaking language changes require a new API version.

A model may propose WPL; the compiler and policy controller decide whether it is valid and executable.
