# Remaining subsystem implementation baseline (15-27)

This document records the current executable reference scope. It does not
promote a research proposal into a production capability.

| Domain | Implemented baseline | External or production adapter still required |
|---|---|---|
| 15 Safety and governance | Default deny, role permissions, HMAC-bound one-use approvals, hash-chain audit ledger, governed tool dispatch, ten-gate artifact promotion | Identity provider, durable WORM audit storage, KMS/HSM, OS/container sandbox and independent human workflow |
| 16 Data infrastructure | Canonical record preparation, deduplication, immutable dataset manifests, license/source metadata, quality gate | Object storage, corpus ingestion, legal review, PII detection/redaction and high-scale streaming |
| 17 Training | Validated config, explicit run lifecycle, metrics, checkpoint manifest registry | PyTorch/JAX trainer, optimizer execution, distributed backend, durable checkpoint storage and GPU/TPU jobs |
| 18 Inference | Model manifest/registry, limits, explicit injected runner, latency result | Weight loader, tokenizer, decoding engine, KV cache, batching, quantization and serving fleet |
| 19 Interaction | Conversation append/idempotency and feedback contracts | Authenticated user store, encryption, retention/deletion service and multimodal front ends |
| 20 Distributed infrastructure | Heartbeat registry, capability scheduling and TTL leases | Network transport, consensus, fencing tokens, durable lease store, accelerator discovery and failover |
| 21 Observability | Structured event/metric/span capture and bounded in-memory store | OpenTelemetry/exporters, durable metrics backend, alert routing and production SLOs |
| 22 Interoperability | Plugin allowlist, API versioning and strict JSON codec | Plugin process isolation, signed packages, protocol clients and remote transport |
| 23 Deployment | Release evidence, approval and staged/canary/rollback state model | Container/cloud provider adapters, health probes, traffic shifting, secrets and rollback orchestration |
| 24 Testing | Test runner registry and result aggregation | Hardware stress, red-team, scale, fault-injection and external benchmark corpora |
| 25 Research | Hypothesis ledger, fixed seed/config/data digest protocols and bound results | Experiment execution, statistical analysis, peer review and publication workflow |
| 26 Documentation | Versioned doc catalog and evidence-backed ADR model | Generated API docs, hosted search/indexing and release publishing |
| 27 Operations | Incident and maintenance state machines, verified backup metadata | On-call integration, real backup/restore execution, immutable storage and incident paging |

All APIs in this baseline are standard-library reference implementations.
No weights, secrets, external systems, containers, cloud infrastructure or
hardware were operated by these modules. Every production adapter must retain
the same fail-closed, provenance, compatibility and audit invariants.

## Nested feature package inventory

All 88 feature subdirectories under domains 15-27 now contain importable Python
packages. The supported import surface is ada.systems, for example
ada.systems.safety.action_approval, ada.systems.data.processed_datasets,
ada.systems.training.post_training, and ada.systems.inference.dynamic_routing.

| Domain | Nested packages implemented |
|---|---|
| 15 | action approval; audit journal; data classification/redaction; artifact integrity; scoped permissions; policy bundles; fail-closed sandbox adapter; security lockout monitor |
| 16 | JSONL pipeline; schema quality; dataset version graph; evaluator-only corpus access; token packing; raw provenance; synthetic review; training corpus; validation corpus and split leakage |
| 17 | atomic checkpoints; continual schedule; distributed training adapter; PyTorch causal step; ADA AdamW and warmup-cosine; four-head MTP loss; causal batch/token budget; optional representation contrastive objective; training health monitor |
| 18 | position-sorted MoD router; token-budget batching; bias-corrected top-k MoE routing and sparse dispatch; incremental generation adapter; inference telemetry; external-memory read cache; digest-verified model loader |
| 19 | conversation turn contract; reviewed feedback queue; context-window builder; multimodal payload metadata; consent/expiry user context |
| 20 | accelerator discovery; cluster snapshot; retry/circuit-breaker; transport envelope/loopback; managed node operations; content-addressed local storage; fencing/version barrier; capability workload distribution |
| 21 | diagnostics; JSON logging; counters/gauges/histograms; latency percentiles; telemetry exporter; context-local spans |
| 22 | module compatibility contracts; allowlisted plugin host; protocol envelope; policy-gated public inference API; strict JSON codec |
| 23 | cloud provider protocol/dry-run; cluster capacity plan; deployment config; hardened container command spec; local runtime adapter; release channel registry |
| 24 | integration graph runner; Wilson reliability bounds; reproducibility capture; safety probes; concurrent stress helper; domain import smoke checks; unit assertion helpers |
| 25 | factorial ablation plans; architecture search candidates; immutable experiment result store; falsifiable hypotheses; literature claim catalogue; scaling calculations; publication readiness record |
| 26 | API index; architecture catalogue; ADR lifecycle; developer guide index; research notebook; subsystem specifications |
| 27 | atomic local backup/restore; incident workflow; service lifecycle manager; maintenance run registry; guarded system update state machine |

These packages are executable reference implementations and adapter contracts.
The paper's core architecture remains the 32-layer
[Mamba2(MoD), Mamba2(MoD), ContinuousMemory, AgentAttn] × 8 design, with
position-sorted 50% MoD, 16-query/4-KV GQA and 75% head gating, top-6/64
bias-corrected experts, read-only differentiable memory, and four MTP heads
using L_main + 0.3 × L_mtp. The four MTP heads and routing components have
CPU PyTorch tests. This does not claim hardware-scale training, measured
performance, production GPU kernels, production container isolation, remote
cluster consensus, or hosted deployment.
