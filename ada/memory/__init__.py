"""Stable public import facade for ADA Memory."""

from importlib import import_module

_contracts = import_module("05_MEMORY.contracts")
_engine = import_module("05_MEMORY.engine")
_retrieval = import_module("05_MEMORY.retrieval")
_store = import_module("05_MEMORY.store")

SCHEMA_VERSION = _contracts.SCHEMA_VERSION
MemoryKind = _contracts.MemoryKind
MemoryQuery = _contracts.MemoryQuery
MemoryRecord = _contracts.MemoryRecord
MemorySearchResult = _contracts.MemorySearchResult
MemorySourceKind = _contracts.MemorySourceKind
MemoryStatus = _contracts.MemoryStatus
MemoryEngine = _engine.MemoryEngine
LexicalMemoryRetriever = _retrieval.LexicalMemoryRetriever
InMemoryMemoryStore = _store.InMemoryMemoryStore
MemoryConflictError = _store.MemoryConflictError
MemoryNotFoundError = _store.MemoryNotFoundError

__all__ = [
    "SCHEMA_VERSION", "MemoryKind", "MemoryQuery", "MemoryRecord",
    "MemorySearchResult", "MemorySourceKind", "MemoryStatus", "MemoryEngine",
    "LexicalMemoryRetriever", "InMemoryMemoryStore", "MemoryConflictError",
    "MemoryNotFoundError",
]
