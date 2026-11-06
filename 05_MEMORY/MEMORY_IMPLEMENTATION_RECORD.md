# Memory v1 implementation record

## Implemented

- Immutable memory record, query and search-result contracts.
- Source kind, source identifier and SHA-256 provenance fields.
- Working, episodic, semantic and procedural memory kinds.
- Namespace-scoped bounded in-memory storage.
- Optimistic version checks and immutable creation timestamps.
- Active, archived and terminal deleted lifecycle states.
- Expiry-aware lexical retrieval with deterministic scoring and stable tie-breaking.
- Representation-to-memory ingestion preserving representation digest and source metadata.
- Stable `ada.memory` facade and unit tests.

## Deliberately deferred

- Durable database backends, encryption at rest, multi-process consistency and distributed replication.
- Vector embeddings, ANN indexes, learned relevance ranking and semantic search.
- Automatic memory extraction, summarization, fact verification and conflict resolution.
- Knowledge graph construction, associative traversal, episodic reconstruction and procedural execution.
- Automated consolidation/decay policies and user-facing retention administration.

## Verification

The implementation is covered by `tests/test_memory.py`. CI is the authority for pass/fail status; this record must only be updated with a run URL and commit SHA after the corresponding workflow succeeds.
