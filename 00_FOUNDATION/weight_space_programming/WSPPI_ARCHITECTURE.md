# WSPPI Architecture

**Status:** Proposed subsystem architecture | **Version:** 0.1 | **Implementation:** Not implemented

## 1. Purpose and design goals

WSPPI is ADA's controlled parameter-space engineering environment. It treats model weights as versioned, typed artifacts rather than mutable inference-time state.

It MUST:
1. Make transformations explicit, inspectable, reproducible, and reversible.
2. Support parameter deltas and adapters before unrestricted full-weight editing.
3. Enable experimental composition of compatible specialist capabilities.
4. Detect regression, forgetting, unsafe behavioral shifts, and artifact corruption.
5. Separate proposal generation from execution, evaluation, and release authorization.
6. Keep inference independent of WSPPI availability.

It is not a mechanism for live weight mutation, arbitrary model-generated code execution, guaranteed capability improvement, or automatic production promotion.

## 2. Logical architecture

WSPPI contains six planes.

### A. Artifact and identity plane

**Weight Registry**
- Stores immutable base checkpoints, adapters, deltas, candidates, and releases.
- Assigns artifact IDs and cryptographic content digests.
- Records architecture, tokenizer, tensor schema, framework, precision, provenance, parent artifact, and creation run.
- Enforces access control and retention.

**Model Manifest**
- Describes architecture version, parameter names and shapes, dtypes, tied parameters, tokenizer, normalization conventions, quantization format, and compatibility constraints.
- Has its own digest and model-family identifier.

**Artifact Inspector**
- Computes parameter counts, tensor statistics, norms, sparsity, distributions, outliers, and delta magnitude.
- Does not claim semantic meaning for individual weights without validated methods.

### B. Programming and planning plane

**Weight Program Compiler**
- Parses declarative programs into a typed Weight Intermediate Representation (WIR).
- Validates tensor selectors, dimensions, operation compatibility, preconditions, budgets, and constraints.
- Rejects unknown operations and undeclared code execution.

**Parameter Planner**
- Converts a stated objective into candidate operations.
- Produces expected benefits, costs, risks, constraints, and required evaluations.
- Accepts researcher-authored or model-proposed plans.

**Operation Library**
- Contains reviewed, versioned transformations.
- Starts with inspection, low-rank deltas, adapter application, bounded interpolation, and parameter masking.
- Enables advanced operations only after separate research and review.

### C. Execution plane

**Isolated Transformation Worker**
- Reads source artifacts and writes only to a candidate workspace.
- Has no production credentials or release permissions.
- Applies compiled operations and records exact inputs, code version, parameters, seeds, and outputs.
- Enforces memory, time, disk, and compute quotas.

**Compatibility and Alignment Engine**
- Checks architecture and tensor compatibility.
- Verifies that parameter correspondence is meaningful before merging.
- Requires explicit alignment for different permutations, architectures, or training lineages.
- Rejects unsafe shape-only assumptions.

### D. Evaluation plane

**Evaluation Harness**
- Runs pre-registered capability, regression, robustness, calibration, safety, and efficiency suites.
- Uses held-out datasets and contamination controls.
- Compares candidate and parent under the same protocol and resource budget.

**Behavioral Differential Analyzer**
- Identifies task families that improved, degraded, or changed unexpectedly.
- Tracks confidence, refusal behavior, tool use, instruction following, and output formats where relevant.
- Treats unexplained behavioral shifts as release blockers.

**Risk and Regression Gate**
- Returns pass, fail, or inconclusive.
- Missing evidence cannot be silently treated as a pass.

### E. Governance plane

**Policy and Permission Controller**
- Restricts operations by artifact, role, and risk.
- Separates research permissions from release permissions.
- Enforces least privilege and provenance requirements.

**Promotion Controller**
- Promotes immutable candidates only after all gates and human approvals.
- Never edits production weights in place.
- Supports staged rollout, canary evaluation, rollback, and revocation.

**Audit Ledger**
- Records proposals, approvals, operations, digests, evaluations, failures, promotion, and rollback events.

### F. ADA integration plane

WSPPI integrates with the Model Registry, Research Orchestrator, Specialist Portfolio, Evaluation System, Memory/Knowledge layer, and Inference Runtime. The runtime reads only approved immutable versions. Memory stores experiment records and verified findings; it is not a hidden weight-write channel.

## 3. End-to-end workflow

1. Select a base artifact and verify manifest and digest.
2. Define objective, scope, constraints, and acceptance tests.
3. Create a declarative weight program.
4. Compile to typed WIR and validate preconditions.
5. Estimate compute, memory, storage, and risk.
6. Execute in an isolated worker.
7. Verify integrity and structural compatibility.
8. Run capability, regression, robustness, safety, and efficiency evaluations.
9. Produce a reproducibility report.
10. Obtain authorized human review for release-bound changes.
11. Promote the immutable artifact through staged rollout.
12. Monitor and roll back if triggers are met.

Any failure before promotion leaves the production reference unchanged.

## 4. Controlled self-evolution loop

**Observe → Diagnose → Propose → Compile → Transform → Evaluate → Review → Promote or Reject**

- Observe measured weaknesses from approved telemetry or evaluation results.
- Diagnose the cause; do not assume every failure is caused by weights.
- Propose candidate weight programs with falsifiable expected effects.
- Compile proposals to typed WIR and reject unsupported operations.
- Transform in isolation.
- Evaluate against fixed target tasks and protected regression suites.
- Present evidence, trade-offs, and uncertainty to an authorized reviewer.
- Promote only after explicit approval, otherwise reject and preserve the report.

The correct intervention may instead be better data, retrieval, tools, architecture, or no change.

## 5. Program and artifact lifecycle

A program moves through: DRAFT, VALIDATED, QUEUED, EXECUTING, CANDIDATE, EVALUATED, APPROVED, PROMOTED. Alternative terminal states are REJECTED, FAILED, CANCELLED, and EXPIRED.

A candidate is not a release. An evaluation pass is not an approval.

## 6. Core interfaces

**Artifact reference:** artifact ID, content digest, model family, architecture version, manifest digest, tokenizer digest, framework/format, parent artifact, provenance, creator, timestamp.

**Weight program:** program ID/version, base artifact, objective, operations, preconditions, constraints, resource budget, required evaluations, proposer, approval policy.

**Candidate report:** source and candidate digests, exact operations, reproducibility metadata, structural checks, capability and regression deltas, safety and efficiency results, limitations, recommendation, reviewer decision.

## 7. Failure handling

- Invalid manifest or digest mismatch: reject before transformation.
- Incompatible schema: reject or require explicit conversion.
- Worker crash: preserve source and mark candidate failed.
- Evaluation timeout: mark inconclusive, never pass.
- Regression: reject and preserve report.
- Registry or ledger outage: fail closed for promotion.
- Promotion failure: retain previous approved reference.
- Rollback trigger: restore previous immutable reference and record incident.

## 8. Minimum viable implementation order

1. Artifact manifest and immutable registry.
2. Tensor inspector and integrity verification.
3. Typed declarative parser/compiler.
4. Isolated worker and quotas.
5. Low-rank delta and adapter operations.
6. Evaluation harness and candidate report.
7. Human-reviewed promotion and rollback.
8. Self-evolution proposal generator.
9. Advanced composition and weight-space operations.

Do not begin with unrestricted model-generated tensor editing.
