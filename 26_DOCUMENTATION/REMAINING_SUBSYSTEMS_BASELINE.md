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
