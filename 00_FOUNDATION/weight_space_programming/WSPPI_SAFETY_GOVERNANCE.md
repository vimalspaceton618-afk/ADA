# WSPPI Safety and Governance

**Status:** Proposed normative controls  
**Scope:** Weight-space programming, candidate generation, self-evolution, and artifact promotion.

## 1. Governing invariant

**No model-generated proposal may directly modify an approved production artifact or authorize its own promotion.** Production inference reads immutable, integrity-verified artifacts. Transformations occur in isolated candidate workspaces. Release changes are reference swaps between immutable artifacts, never in-place edits.

## 2. Trust boundaries

1. **Proposal domain:** ADA may describe a proposed change; its output is untrusted.
2. **Compiler domain:** validates syntax, operation allowlists, selectors, preconditions, and budgets.
3. **Worker domain:** executes reviewed operations without production credentials.
4. **Evaluation domain:** evaluates candidates using controlled datasets and fixed protocols.
5. **Approval domain:** authorized human reviews evidence and decides promotion.
6. **Production domain:** serves only approved, integrity-verified artifacts.

No domain inherits permissions because another domain produced its input.

## 3. Threat model

| Threat | Example | Required control |
|---|---|---|
| Malicious proposal | Attempt to overwrite all tensors | Typed allowlist, selector validation, sandbox |
| Backdoor insertion | Hidden trigger behavior | Trigger-focused red-team and behavioral evaluation |
| Catastrophic forgetting | Target skill improves, general capability collapses | Protected regression suites and retention thresholds |
| Evaluator overfitting | Candidate tuned to known examples | Held-out tests, private evaluations, contamination checks |
| Data poisoning | Malicious samples drive an update | Provenance, validation, anomaly detection, review |
| Artifact substitution | Candidate or base replaced | Content digests, immutable registry |
| Privilege escalation | Model requests release credentials | No credentials in model context; separate approval |
| Resource exhaustion | Huge operation consumes compute | Static estimates, quotas, timeout, cancellation |
| Supply-chain compromise | Unreviewed plugin executes | Signed/versioned operation registry and code review |
| Audit tampering | Failed evaluations hidden | Append-only or tamper-evident audit ledger |

## 4. Permission model

- **Reader:** inspect metadata and reports.
- **Researcher:** create programs and run approved operations in isolated workspaces.
- **Evaluator:** manage evaluation suites and sign reports.
- **Release approver:** authorize promotion after reviewing evidence.
- **Administrator:** manage registry, identities, and policy.

A single automated model identity must not hold all roles. Production promotion requires an authorized human approver and separate policy check.

## 5. Protected resources

Protect approved production artifacts and manifests; safety-critical policy and authorization components; hidden evaluation labels; signing keys and credentials; audit and provenance records; user and confidential data. Workers never receive release signing keys.

## 6. Candidate validation gates

Promotion requires all applicable checks to pass:

1. Integrity: source and output digests match manifests.
2. Structural: tensor names, shapes, dtypes, and architecture compatibility are valid.
3. Numerical: no NaN/Inf or unexplained instability.
4. Target capability: predefined improvement is demonstrated.
5. Protected regression: retention thresholds are met.
6. Robustness: distribution-shift and adversarial probes are acceptable.
7. Safety: refusal, privacy, and tool-authorization behavior remain acceptable.
8. Efficiency: deployment resource limits are met.
9. Provenance: data, operations, code, and approvals are traceable.
10. Review: authorized human approves the complete report.

Missing, unavailable, or inconclusive evidence is not a pass.

## 7. Release and rollback

Candidate artifacts are immutable after evaluation. Promotion records old/new digests, reviewer, report, and timestamp. Use staged rollout or canary where appropriate. Rollback restores the previous approved reference. Triggers include integrity failure, severe regression, safety incident, abnormal operations, or unexplained behavior changes. Every rollback is recorded and investigated.

## 8. Self-evolution limits

Self-evolution may analyze approved evaluations, suggest hypotheses and WPL programs, run approved transformations in sandboxes, and generate reports.

It may not grant itself permissions; disable gates; modify protected policy or approval logic; access production signing credentials; promote its own candidate; erase audit trails; or change acceptance thresholds after seeing held-out test outcomes.

## 9. Data and privacy

Every update dataset requires provenance, lawful usage basis, and appropriate privacy review. Personal, confidential, or sensitive data must not enter parameter updates by default. Data deletion and retention obligations must be considered in the data pipeline and model governance.

## 10. Incident response

If an artifact may contain a backdoor, harmful behavior, unauthorized data, or unexplained regression: freeze promotion; quarantine artifact; preserve digests, logs, manifests, and reports; revert serving to last known-good artifact if deployed; investigate source data, operation, code, and proposer path; run independent evaluation after remediation; and record root cause before resuming.

WSPPI is a parameter engineering and research capability, not an independent authority. Its power comes from precise transformations and evidence; its limits come from isolation, authorization, independent evaluation, and human accountability.
