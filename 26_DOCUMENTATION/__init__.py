"""Versioned documentation catalog and architecture decision records."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("content_digest must be lowercase SHA-256")


class DecisionStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class DocumentationEntry:
    path: str
    title: str
    version: str
    content_digest: str
    owner: str
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("path", "title", "version", "owner"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        if self.path.startswith("/") or ".." in self.path.split("/"):
            raise ValueError("documentation path must be repository-relative")
        _sha(self.content_digest)


@dataclass(frozen=True, slots=True)
class ArchitectureDecisionRecord:
    adr_id: str
    title: str
    context: str
    decision: str
    consequences: str
    alternatives: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    status: DecisionStatus = DecisionStatus.PROPOSED

    def __post_init__(self):
        for name in ("adr_id", "title", "context", "decision", "consequences"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        if not isinstance(self.status, DecisionStatus):
            raise TypeError("status must be DecisionStatus")
        if self.status is DecisionStatus.ACCEPTED and not self.evidence_refs:
            raise ValueError("accepted ADRs must cite evidence")


class DocumentationCatalog:
    def __init__(self):
        self._entries = {}
        self._adrs = {}
        self._lock = threading.RLock()

    def add_entry(self, entry: DocumentationEntry):
        if not isinstance(entry, DocumentationEntry):
            raise TypeError("entry must be DocumentationEntry")
        key = (entry.path, entry.version)
        with self._lock:
            old = self._entries.get(key)
            if old is not None and old != entry:
                raise ValueError("documentation version is immutable")
            self._entries[key] = entry
        return entry

    def add_adr(self, adr: ArchitectureDecisionRecord):
        if not isinstance(adr, ArchitectureDecisionRecord):
            raise TypeError("adr must be ArchitectureDecisionRecord")
        with self._lock:
            if adr.adr_id in self._adrs:
                raise ValueError("ADR ID already exists")
            self._adrs[adr.adr_id] = adr
        return adr

    def search(self, query: str):
        terms = {x for x in query.casefold().split() if x}
        with self._lock:
            values = tuple(self._entries.values())
        return tuple(sorted(
            (x for x in values if terms.issubset(
                set((x.path + " " + x.title + " " + " ".join(x.tags)).casefold().split())
            )),
            key=lambda x: (x.path, x.version),
        ))

    @property
    def decisions(self):
        with self._lock:
            return tuple(self._adrs[k] for k in sorted(self._adrs))


__all__ = [
    "DecisionStatus", "DocumentationEntry", "ArchitectureDecisionRecord",
    "DocumentationCatalog",
]
