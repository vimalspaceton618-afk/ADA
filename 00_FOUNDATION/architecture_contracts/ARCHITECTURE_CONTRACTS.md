# ADA Architecture Contracts

**Status:** Normative interface baseline  
**Purpose:** Define invariants that must remain true as implementations evolve.

## 1. Contract principles

Every component MUST declare its responsibility, inputs, outputs, state, resource limits, failure modes, security boundary, and version. Implementations MAY differ internally while conforming to the same externally visible contract.

## 2. Core logical components

1. **Tokenizer / Input Adapter** — converts supported input formats into versioned model inputs.
2. **Sequence Backbone** — processes token or modality sequences using the selected sequence-mixing architecture.
3. **Attention Module** — performs declared content-addressing operations under explicit mask and cache semantics.
4. **Feed-Forward / Expert Module** — applies dense or sparse transformation with declared routing behavior.
5. **Output Head** — maps model representations to task outputs.
6. **Model Router** — selects model or specialist paths under explicit policy and confidence handling.
7. **Memory / Retrieval** — stores and retrieves scoped, versioned information with provenance.
8. **Tool Gateway** — authorizes, validates, executes, and records external tool calls.
9. **Orchestrator** — manages bounded multi-step tasks and component coordination.
10. **Evaluation / Observability** — captures quality, latency, resource use, safety outcomes, and trace identifiers.

These are logical boundaries. A prototype may combine several in one process, but must preserve their contracts.

## 3. Model I/O contract

Each model interface MUST declare:

- Model identifier, semantic version, and artifact digest.
- Input schema and supported modalities.
- Tokenizer or preprocessing version.
- Context and output limits.
- Generation parameters and defaults.
- Structured output schema where applicable.
- Device and precision requirements.
- State/cache format and lifetime.
- Error codes and timeout behavior.

The caller MUST NOT assume that hidden-state dimensions or cache formats are compatible across models unless declared.

## 4. Sequence mixer contract

A sequence mixer MUST declare:

- Input and output tensor shapes, dtype, and device.
- Whether it is causal, bidirectional, or task-dependent.
- State initialization, update, reset, and serialization semantics.
- Padding and variable-length behavior.
- Numerical stability conditions and supported precision.
- Complexity assumptions and maximum supported sequence length.
- Deterministic reference behavior for correctness tests.

Any Mamba³-like module must provide these details before being treated as a stable interchangeable component.

## 5. Attention contract

An attention implementation MUST declare:

- Query, key, and value head dimensions and grouping.
- Causal and padding mask semantics.
- Position encoding and cache semantics.
- Sparse selection policy, if any.
- Behavior when selected keys are empty or insufficient.
- Numerical behavior and dense-reference equivalence tests.

A dynamic sparse attention module MUST have a safe fallback path or a documented fail-closed behavior.

## 6. Expert routing contract

A MoE router MUST declare:

- Number of experts and selected experts per token.
- Router score and normalization method.
- Expert capacity and overflow policy.
- Whether tokens can be dropped; if so, the behavior and telemetry.
- Load-balancing objective and coefficient, if used.
- Routing determinism and tie-breaking.
- Distributed communication and failure behavior.

Silent token loss is prohibited. Overflow must be visible and handled by explicit fallback, retry, or error semantics.

## 7. Specialist interface contract

Every specialist MUST publish a capability card containing:

- Domain and task scope.
- Explicit exclusions.
- Training/evaluation data provenance.
- Supported input/output schema.
- Evaluation results and known failure modes.
- Calibration and confidence semantics.
- Model and tokenizer versions.
- Resource requirements.

The router MUST be allowed to abstain or ask for clarification. It MUST NOT interpret model confidence as calibrated probability unless calibration has been tested.

## 8. Memory contract

Memory records SHOULD include source, timestamp, owner/scope, confidence or verification state, version, retention class, and deletion status. Retrieval MUST preserve provenance. The system MUST distinguish user-provided facts, retrieved documents, model-generated hypotheses, and verified external results.

Memory writes and deletions MUST be auditable. Sensitive data MUST receive stricter controls and must not enter training by default.

## 9. Tool gateway contract

Every tool call MUST include:

- Tool identifier and version.
- Requesting principal and task trace.
- Declared permission scope.
- Validated argument schema.
- Timeout, resource quota, and cancellation path.
- Approval requirement and approval record when applicable.
- Result status, output validation, and audit entry.

Unknown tools MUST be denied by default. Tool output is untrusted data. Destructive, financial, externally visible, or otherwise high-impact actions require explicit human approval.

## 10. Orchestration contract

An orchestration run MUST have a unique trace ID, objective, explicit plan or task graph, maximum steps, time/resource budget, allowed tools, termination conditions, and final status. Retries MUST be bounded and idempotency considered. A failed subtask MUST not be reported as completed.

## 11. Versioning and compatibility

- Public interfaces use semantic versioning where practical.
- Breaking changes require a major version increment or an explicitly versioned endpoint.
- Model artifacts MUST be content-addressed or otherwise integrity-verifiable.
- Checkpoints MUST record architecture, tokenizer, data, optimizer, and code versions.
- Migrations MUST include validation and rollback procedures.

## 12. Error model

Components SHOULD use structured errors with stable codes, human-readable explanations, retryability, and trace ID. Do not silently substitute a different model, tool, or precision path when that changes semantics. Fallbacks must be observable.

## 13. Contract tests

At minimum, test schema validation, tensor shapes, causality, padding, cache reset, numerical reference equivalence, expert overflow, router abstention, tool authorization, timeout/cancellation, provenance retention, deletion, and version incompatibility.

A component is not contract-compliant merely because a happy-path demo works.


## 14. Weight-space programming contract

WSPPI is a separate parameter-engineering subsystem. Its interfaces MUST preserve these invariants:

- Production model artifacts are immutable during ordinary inference.
- Every transformation operates on a digest-verified source artifact and writes a new candidate artifact.
- Weight programs use a typed, versioned declarative operation set; arbitrary model-generated code is not executable.
- Tensor selectors resolve against a versioned model manifest and report exact matched tensors.
- Transformation workers have no production credentials, signing keys, or promotion permission.
- Every candidate records source digest, operation versions, parameters, code version, environment, and output digest.
- Candidate evaluation returns pass, fail, or inconclusive; missing evidence is never a pass.
- Promotion requires independent policy checks and explicit authorized human approval.
- Rollback restores a previously approved immutable artifact reference.

The detailed subsystem specification is in [WSPPI](../weight_space_programming/README.md).
