# ADA Design Principles

**Status:** Normative baseline  
The terms **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are used as engineering requirements.

## 1. Evidence before claims

- Capability claims MUST be tied to reproducible evaluations.
- Comparisons MUST disclose model versions, prompts, tools, datasets, compute, decoding settings, and scoring methods where relevant.
- A benchmark score MUST NOT be treated as proof of broad intelligence.
- Negative and inconclusive results MUST be retained in research records.

## 2. Modularity with explicit boundaries

- Model components, data systems, memory, retrieval, tools, and orchestration MUST have defined interfaces.
- Components SHOULD be independently testable and replaceable.
- Cross-component communication MUST use versioned schemas and explicit error states.
- Modularity MUST not become an excuse for hidden state or untraceable behavior.

## 3. Adaptive computation, not complexity for its own sake

- Conditional computation SHOULD be used when it improves measured quality, latency, memory, or energy under a declared workload.
- Routing overhead, expert imbalance, synchronization, and communication cost MUST be measured.
- A sparse design MUST be compared against a dense baseline at equal or clearly reported compute budgets.
- Architectural novelty is not an objective function.

## 4. Efficient by measurement

Measure at least the applicable dimensions: parameter count, active parameters per token, FLOPs, accelerator utilization, peak memory, tokens per second, time to first token, end-to-end latency, energy, throughput under concurrency, and cost per useful task.

A design that reduces theoretical FLOPs but increases wall-clock latency or total cost is not automatically an efficiency improvement.

## 5. Data responsibility

- Every training and evaluation dataset MUST have a source record, license or usage basis, collection method, processing history, and known limitations.
- Private, personal, confidential, and sensitive information MUST receive appropriate protection and access controls.
- Train/evaluation/test splits MUST be constructed to minimize leakage.
- Data deduplication, contamination checks, and quality filtering SHOULD be documented.
- Synthetic data MUST be labeled as synthetic and evaluated for artifacts and feedback loops.

## 6. Learning must be controlled

- Production model weights MUST be immutable during ordinary inference.
- Updates MUST pass through versioned training or adaptation pipelines, evaluations, approval, and release gates.
- Online learning, if explored, MUST be isolated, bounded, monitored, and reversible.
- Memory writes MUST be scoped, attributable, inspectable, and subject to retention and deletion policies.
- Feedback MUST NOT be treated as ground truth without validation.

## 7. Safety and security by construction

- Default-deny authorization MUST apply to tools and plugins.
- Least privilege, explicit allowlists, scoped credentials, timeouts, quotas, and audit logs SHOULD be standard.
- Untrusted input MUST be treated as data, not policy or executable instruction.
- Tool outputs MUST be validated before being used in consequential actions.
- High-impact or irreversible external actions MUST require explicit human approval.
- Secrets MUST never be embedded in source code, prompts, datasets, logs, or model artifacts.

## 8. Human control and accountability

- The system MUST expose why an action was proposed, what permissions it needs, and what it will change.
- Users MUST be able to stop execution and revoke access.
- Autonomous plans MUST have bounded steps, budgets, and termination conditions.
- The system MUST not represent a simulated action as a completed external action.

## 9. Reproducibility and observability

- Experiments SHOULD pin code, data manifests, configurations, seeds, dependencies, and hardware details.
- Training and evaluation runs MUST have unique identifiers and machine-readable results.
- Important routing, retrieval, tool, and safety decisions SHOULD be traceable.
- Reproducibility limits caused by nondeterministic kernels or distributed execution MUST be stated.

## 10. Portability and practical constraints

- Architecture decisions SHOULD consider available hardware, interconnect, memory capacity, software maturity, and cost.
- Research code SHOULD separate algorithmic definitions from backend-specific kernels.
- CPU fallback and low-resource development paths SHOULD exist where feasible.
- Accelerator-specific optimizations MUST retain a correctness reference implementation.

## 11. Responsible cybersecurity research

Cybersecurity work MUST be authorized, isolated, and defensive by default. Evaluations SHOULD use controlled labs, synthetic targets, or explicitly authorized environments. The project MUST NOT facilitate unauthorized access, persistence, exfiltration, or destructive activity.

## 12. Simplicity and maintainability

- Prefer the smallest design that meets measured requirements.
- Every abstraction SHOULD have a clear responsibility.
- Experimental features MUST be marked and separated from stable interfaces.
- Deprecated behavior MUST include a migration note or removal rationale.

## 13. Decision review

A decision with significant impact on data, architecture, safety, cost, or compatibility SHOULD be recorded as an ADR with alternatives and a rollback plan. Revisit decisions when new evidence invalidates their assumptions.
