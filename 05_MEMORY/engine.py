"""Public memory orchestration API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .contracts import (
    MemoryKind, MemoryQuery, MemoryRecord, MemorySearchResult, MemorySourceKind,
    MemoryStatus, utc_now,
)
from .retrieval import LexicalMemoryRetriever
from .store import InMemoryMemoryStore


class MemoryEngine:
    """Create, retrieve, archive and delete provenance-linked memory records.

    The v1 engine is in-memory and deterministic. It does not infer facts,
    summarize experiences, embed content, or persist data across processes.
    """

    def __init__(self, *, store: InMemoryMemoryStore | None = None, max_records: int = 100_000) -> None:
        self.store = store or InMemoryMemoryStore(max_records=max_records)
        self.retriever = LexicalMemoryRetriever(self.store)

    def remember(self, content: Mapping[str, Any], *, kind: MemoryKind,
                 source_kind: MemorySourceKind, source_id: str, source_digest: str,
                 namespace: str = "default", tags: tuple[str, ...] = (),
                 metadata: Mapping[str, Any] | None = None,
                 expires_at: datetime | None = None) -> MemoryRecord:
        record = MemoryRecord(
            kind=kind, content=content, source_kind=source_kind, source_id=source_id,
            source_digest=source_digest, namespace=namespace, tags=tags,
            metadata=metadata or {}, expires_at=expires_at,
        )
        self.store.put(record)
        return record

    def remember_representation(self, representation, *, namespace: str = "default",
                                 tags: tuple[str, ...] = ()) -> MemoryRecord:
        """Store an ADA Representation contract without altering its provenance."""
        from importlib import import_module
        module = import_module("04_REPRESENTATION.contracts")
        if not isinstance(representation, module.Representation):
            raise TypeError("representation must be a 04_REPRESENTATION.Representation")
        return self.remember(
            representation.to_dict(), kind=MemoryKind.SEMANTIC,
            source_kind=MemorySourceKind.REPRESENTATION,
            source_id=representation.representation_id,
            source_digest=representation.representation_digest,
            namespace=namespace, tags=tags,
            metadata={"source_input_id": representation.source_input_id,
                      "modality": representation.modality,
                      "representation_kind": representation.kind.value},
        )

    def retrieve(self, query: str, filters: MemoryQuery | None = None) -> tuple[MemorySearchResult, ...]:
        return self.retriever.search(query, filters)

    def get(self, memory_id: str) -> MemoryRecord:
        return self.store.get(memory_id)

    def archive(self, memory_id: str, *, expected_version: int | None = None) -> int:
        return self.store.replace_status(memory_id, MemoryStatus.ARCHIVED, expected_version=expected_version)

    def delete(self, memory_id: str, *, expected_version: int | None = None) -> int:
        return self.store.delete(memory_id, expected_version=expected_version)

    def purge_expired(self, *, at: datetime | None = None, namespace: str | None = None) -> int:
        return self.store.purge_expired(at=at, namespace=namespace)
