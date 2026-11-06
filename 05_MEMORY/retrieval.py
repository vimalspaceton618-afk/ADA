"""Deterministic lexical retrieval baseline; no embeddings or semantic claims."""

from __future__ import annotations

import re
from collections import Counter

from .contracts import MemoryQuery, MemorySearchResult, MemoryStatus, thaw_json, utc_now
from .store import InMemoryMemoryStore

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _terms(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _WORD.findall(text))


def _flatten(value) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_flatten(item) for item in value)
    if value is None:
        return ""
    return str(value)


class LexicalMemoryRetriever:
    def __init__(self, store: InMemoryMemoryStore) -> None:
        self.store = store

    def search(self, query: str, filters: MemoryQuery | None = None) -> tuple[MemorySearchResult, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty")
        spec = filters or MemoryQuery()
        query_terms = tuple(dict.fromkeys(_terms(query)))
        if not query_terms:
            return ()
        now = utc_now()
        results = []
        for record in self.store.list_records(namespace=spec.namespace, include_archived=spec.include_archived):
            if record.expires_at is not None and record.expires_at <= now:
                continue
            if spec.kinds and record.kind not in spec.kinds:
                continue
            if spec.tags and not set(spec.tags).issubset(record.tags):
                continue
            if spec.source_id is not None and record.source_id != spec.source_id:
                continue
            text = _flatten(thaw_json(record.content))
            haystack = set(_terms(text))
            matched = tuple(term for term in query_terms if term in haystack)
            if not matched:
                continue
            score = len(matched) / len(query_terms)
            results.append(MemorySearchResult(record, score, matched))
        results.sort(key=lambda item: (-item.score, -item.record.updated_at.timestamp(), item.record.memory_id))
        return tuple(results[:spec.limit])
