"""Canonical implementation package for ADA Memory v1."""

from .contracts import (
    SCHEMA_VERSION, MemoryKind, MemoryQuery, MemoryRecord, MemorySearchResult,
    MemorySourceKind, MemoryStatus,
)
from .engine import MemoryEngine
from .retrieval import LexicalMemoryRetriever
from .store import InMemoryMemoryStore, MemoryConflictError, MemoryNotFoundError

__all__ = [
    "SCHEMA_VERSION", "MemoryKind", "MemoryQuery", "MemoryRecord",
    "MemorySearchResult", "MemorySourceKind", "MemoryStatus", "MemoryEngine",
    "LexicalMemoryRetriever", "InMemoryMemoryStore", "MemoryConflictError",
    "MemoryNotFoundError",
]
