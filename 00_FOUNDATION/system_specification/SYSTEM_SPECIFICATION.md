# ADA System Specification — Baseline 0.1

**Status:** Proposed logical architecture  
**Implementation status:** Not implemented by this document  
**Primary concern:** A modular, efficient, controllable system for model research and composition.

## 1. System overview

ADA is conceived as a layered system:

1. **Data and research layer** — dataset ingestion, provenance, curation, deduplication, licensing, and evaluation manifests.
2. **Model training layer** — tokenizer, model definitions, optimization, distributed training, checkpointing, and experiment tracking.
3. **Inference layer** — optimized model serving, batching, cache management, quantization, and telemetry.
4. **Intelligence layer** — sequence backbone, attention, dense or sparse feed-forward modules, and task heads.
5. **Specialist layer** — optional domain models with capability cards and independent evaluation.
6. **Cognition and orchestration layer** — routing, planning, bounded task execution, and result synthesis.
7. **Memory and knowledge layer** — retrieval, scoped memory, source tracking, and retention controls.
8. **Tool and action layer** — permissioned tool gateway, sandboxing, approval gates, and audit logs.
9. **Evaluation and safety layer** — continuous tests, policy enforcement, monitoring, incident response, and release gates.

## 2. Candidate model architecture

The current research direction explores a hybrid of:

- **Mamba³-like selective state-space blocks** for efficient sequence processing.
- **Dense or dynamic sparse Grouped-Query Attention (GQA)** at selected layers for content-based retrieval.
- **RoPE or another explicitly specified positional mechanism** where compatible with the selected attention design.
- **SwiGLU or another measured feed-forward design**.
- **RMSNorm or another measured normalization design**.
- **Mixture-of-Experts (MoE)** for conditional feed-forward computation, with top-k routing as a candidate.
- **Residual and gating patterns** chosen through controlled ablations.

The phrase **Mamba³ + MoE + Dense + Pangu-style scaling** is shorthand for a research direction, not a complete architecture specification and not a reproduction claim. It does not imply that a particular external implementation, proprietary design, or scaling method has been copied.

### Candidate layer schedule

A prior design discussion proposed a 24-layer allocation containing 21 state-space layers and 3 attention layers (7:1 ratio). Treat this as one experimental configuration. It MUST be compared against alternative ratios, layer placements, and baselines. It is not a fixed law or a validated design.

The architecture specification for an actual implementation must define dimensions, head counts, state size, gating, precision, initialization, normalization, residual paths, cache semantics, and exact routing behavior.

## 3. Specialist model strategy

A portfolio of narrow models is a candidate way to develop concentrated competence in selected areas. Initial domains discussed include:

- Defensive cybersecurity and security analysis in authorized environments.
- Complex software engineering and coding.
- Mathematics and formal problem solving.
- Physics and scientific computation.
- Biology and biological knowledge tasks.

The portfolio may later be composed into a unified ADA experience. That composition could mean routing, ensembling, distillation, shared representation training, or another method. These are not interchangeable. The strategy must be selected through controlled evaluation.

A proposed 20M-parameter specialist and 120M-parameter Ada preview concept appeared in earlier design discussions. These are historical exploration points, not committed production sizes. Likewise, the 120M concept’s proposed 24-layer 21:3 composition remains an experiment.

## 4. Model scale and data

Earlier ambitions included models above 15B, 50B, and 100B parameters, very large token budgets, and high-quality domain data. Such figures are planning scenarios, not feasibility guarantees. The project must first establish:

- Data rights, provenance, quality, and contamination controls.
- Compute and memory estimates.
- Training stability at small scale.
- Evaluation sets and target metrics.
- Cost, energy, and infrastructure constraints.
- Inference and deployment requirements.

A token count or parameter target must never substitute for a data-quality or capability argument.

## 5. Training lifecycle

1. Specify the task distribution and success criteria.
2. Build and audit dataset manifests.
3. Implement a correctness-first reference model.
4. Run unit tests and small-scale smoke tests.
5. Conduct controlled ablations.
6. Train with versioned configurations and checkpoints.
7. Evaluate capability, robustness, calibration, safety, and efficiency.
8. Review results and record an ADR.
9. Promote only validated artifacts into release candidates.
10. Preserve rollback and reproducibility metadata.

## 6. Inference lifecycle

Input validation → policy and task classification → optional specialist routing → model inference → output validation → optional retrieval/tool execution under authorization → result synthesis → response and trace record.

Not every request needs a tool, specialist, memory lookup, or multi-step plan. The simplest valid execution path should be preferred.

## 7. Memory and adaptation

The default architecture separates model weights from external memory. Memory can support continuity and retrieval, but is not equivalent to learning or a change in model parameters.

Any weight adaptation must occur in a controlled pipeline with data review, evaluation, approval, versioning, and rollback. No unrestricted self-modification is part of this baseline.

## 8. Security and control plane

The control plane owns identity, authorization, quotas, tool registry, approval gates, traceability, and emergency stop. It must be logically separate from model-generated plans. Model text cannot create permissions or alter policy.

Tool execution should use isolated processes or sandboxes, scoped credentials, egress restrictions where appropriate, and strict input/output validation.

## 9. Evaluation gates

| Gate | Required evidence |
|---|---|
| G0 — Definition | Task scope, baselines, metrics, data rights, risk assessment |
| G1 — Correctness | Unit tests, reference equivalence, shape and mask tests |
| G2 — Small-scale viability | Stable training, informative ablations, reproducible run |
| G3 — Capability | Held-out task performance, robustness, calibration |
| G4 — Efficiency | Measured latency, throughput, memory, energy/cost |
| G5 — Safety | Tool authorization, prompt-injection resilience, privacy and misuse tests |
| G6 — Release | Versioned artifact, documentation, monitoring, rollback, owner approval |

A failed gate blocks progression until the issue is resolved or the scope is explicitly changed.

## 10. Hardware and framework strategy

Architecture research should be portable across PyTorch and JAX where practical, while acknowledging that backend-specific kernels and distributed execution differ. A correctness-first implementation should precede custom kernels. Hardware selection must follow workload profiling and realistic budget analysis; no single accelerator type is assumed.

## 11. Release definition

An ADA model release must identify the artifact, intended use, excluded use, architecture version, tokenizer, data documentation, evaluations, limitations, license/terms, hardware requirements, and known risks. A demo is not a release. A checkpoint is not a validated product.

## 12. Open specifications

Before implementation freeze, define exact model dimensions; state-space recurrence; attention positions and sparsity; MoE routing and capacity; normalization and residual topology; tokenizer; training objective; optimizer and schedule; data composition; inference cache; specialist protocol; and safety policy. These must be version-controlled and tied to testable configurations.
