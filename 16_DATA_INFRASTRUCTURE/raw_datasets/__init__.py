"""Provenance-bearing raw dataset intake; no network fetch is performed."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping, Any

@dataclass(frozen=True, slots=True)
class RawDatasetSource:
    source_id: str
    uri: str
    license_id: str
    retrieved_at: str
    content_digest: str
    def __post_init__(self):
        if any(not x.strip() for x in (self.source_id, self.uri, self.license_id, self.retrieved_at)):
            raise ValueError("source provenance fields are required")
        if len(self.content_digest) != 64: raise ValueError("content_digest must be SHA-256")

def canonical_records_digest(records: Iterable[Mapping[str, Any]]) -> str:
    import json
    raw = json.dumps(list(records), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

__all__ = ["RawDatasetSource", "canonical_records_digest"]
