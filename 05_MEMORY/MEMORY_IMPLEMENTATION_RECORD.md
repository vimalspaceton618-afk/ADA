# ADA Memory implementation record

## Implemented

- Immutable memory record, query and search-result contracts.
- Source kind, source identifier and SHA-256 provenance fields.
- Working, episodic, semantic and procedural memory kinds.
- Namespace-scoped bounded in-memory storage.
- Optimistic version checks and immutable creation timestamps.
- Active, archived and terminal deleted lifecycle states.
- Expiry-aware deterministic lexical retrieval.
- Representation-to-memory ingestion using the representation's source content SHA-256.
- Stable `ada.memory` facade.
- PDF-aligned neural Energy Memory:
  - 512 learnable slots by default.
  - `X @ M.T` associative scores.
  - temperature-scaled softmax over slots.
  - retrieval with `attention @ M`.
  - feature-wise sigmoid gate via `W_mem_gate`.
  - optional attention/raw/gate diagnostics.
  - gradient-based bank learning; no unsupported online write/erase policy.
- Dedicated unit tests for neural output shapes, probability normalization, gating, gradients, validation and state-dict round-trip.

## Research considered

- ADA's own `Ada-src.pdf` is the implementation authority for the Energy Memory read path.
- Tencent Hy-Memory informs the separation of learned memory writing/state revision from neural associative retrieval. Its large writer model is not a runtime dependency.
- R^3Mem, S^3-Attention and SYNAPSE are documented as future research directions, not copied or claimed as implemented.

## Deliberately deferred

- Durable database backends, encryption at rest, multi-process consistency and distributed replication.
- Vector embeddings, ANN indexes, learned relevance ranking and semantic search.
- Learned memory writer, fact verification, conflict resolution and cross-session consolidation.
- Knowledge graph construction, spreading activation, episodic reconstruction and procedural execution.
- Integration/fusion into a specific model FFN/MoE block; this requires inspecting and changing the upstream model source.

## Verification

CI is authoritative. Do not mark this milestone verified until a GitHub Actions run against the final commit SHA completes successfully for every configured Python version. Previous runs failed in the test-suite step; see the current Actions page before relying on status.
