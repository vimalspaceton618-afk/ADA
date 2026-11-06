# ADA Memory implementation record

## Implemented

- Immutable memory record, query and search-result contracts.
- Source kind, source identifier and SHA-256 provenance fields.
- Working, episodic, semantic and procedural memory kinds.
- Namespace-scoped bounded in-memory storage.
- Optimistic version checks and immutable creation timestamps.
- Active, archived and terminal deleted lifecycle states.
- Expiry-aware deterministic lexical retrieval.
- Representation-to-memory ingestion preserving the upstream source content SHA-256 and storing the independent representation digest in metadata.
- Stable `ada.memory` facade.
- PDF-aligned neural Energy Memory:
  - 512 learnable slots by default.
  - `X @ M.T` associative scores.
  - temperature-scaled softmax over slots.
  - retrieval with `attention @ M`.
  - feature-wise sigmoid gate via `W_mem_gate`.
  - optional attention/raw/gate diagnostics.
  - gradient-based bank learning; no unsupported online write/erase policy.
- Dedicated unit tests for neural output shapes, probability normalization, gating, gradients, exact Hopfield energy derivative, validation and state-dict round-trip.
- PDF-aligned noisy top-1 MoE FFN with sparse expert dispatch, straight-through router gradients, load-balancing loss, learned MoE/memory blending and residual contract.

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

Verified on commit `12c2723e7f4bf0593df0417b78ab576abbfea1b0`: [GitHub Actions run #230](https://github.com/vimalspaceton618-afk/ADA/actions/runs/42172158875) passed on Python 3.11 and 3.12 (110 tests passed in each job). This verification applies to the code and tests at that commit; later code changes require a new green run.
