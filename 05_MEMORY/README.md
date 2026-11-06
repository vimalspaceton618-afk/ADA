# ADA Memory

05_MEMORY/ is the canonical, provenance-aware memory boundary between representation and downstream cognition. The stable application import is `ada.memory`.

## v1 architecture

- `contracts.py`: immutable memory records, namespaces, kinds, lifecycle states, provenance and query contracts.
- `store.py`: bounded, thread-safe in-memory store with optimistic version checks and terminal deletion.
- `retrieval.py`: deterministic lexical retrieval with namespace, kind, tag, source and lifecycle filters.
- `engine.py`: record creation, representation ingestion, retrieval, archive and deletion.

## Quick start

```python
from ada.memory import MemoryEngine, MemoryKind, MemorySourceKind

memory = MemoryEngine()
record = memory.remember(
    {"text": "ADA received a verified source observation."},
    kind=MemoryKind.EPISODIC,
    source_kind=MemorySourceKind.USER,
    source_id="session-001",
    source_digest="a" * 64,  # Replace with the actual upstream SHA-256.
    namespace="research",
    tags=("ada", "observation"),
)
matches = memory.retrieve("verified source observation")
```

Representation handoff:

```python
from ada.perception import Modality, PerceptionEngine, PerceptionInput
from ada.representation import RepresentationEngine
from ada.memory import MemoryEngine

observed = PerceptionEngine().perceive(
    PerceptionInput(modality=Modality.TEXT, content="A source-grounded statement.")
)
projected = RepresentationEngine().transform(observed)
memory = MemoryEngine()
for item in projected.representations:
    memory.remember_representation(item, namespace="default")
```

## Invariants and safety

- Every record carries a source kind, source ID and lowercase SHA-256 source digest.
- Content and metadata are finite JSON-compatible values, deeply immutable after construction.
- Namespace isolation is enforced by every list and retrieval operation.
- Retrieval excludes expired and deleted records; archived records require explicit opt-in.
- Updates support optimistic version checks to prevent lost updates.
- Deletion is a terminal tombstone; v1 does not silently restore deleted records.
- Expiry is enforced during retrieval and can be physically marked deleted with `purge_expired()`.
- No automatic extraction, fact inference, summarization, embedding, or semantic similarity is claimed.

## Explicit v1 limits

The store is process-local and volatile. It is not a database, vector index, distributed cache, durable event log, or encrypted-at-rest persistence layer. Restarting the process loses records. Lexical retrieval is a baseline, not semantic retrieval. Knowledge-graph edges, associative traversal, episodic reconstruction, procedural execution, consolidation policies and learned memory ranking are deferred. The corresponding scaffold directories are extension boundaries, not implemented features.

No mandatory third-party dependency is introduced.
