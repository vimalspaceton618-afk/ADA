"""Structured literature references with claim/evidence provenance."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class LiteratureReference:
    reference_id: str
    title: str
    authors: tuple[str, ...]
    year: int
    identifier: str
    url: str | None = None
    def __post_init__(self):
        if not self.reference_id.strip() or not self.title.strip() or not self.authors: raise ValueError("reference metadata required")
        if type(self.year) is not int or self.year < 1900: raise ValueError("invalid publication year")

@dataclass(frozen=True, slots=True)
class LiteratureClaim:
    claim_id: str
    statement: str
    reference_ids: tuple[str, ...]
    confidence: str = "unassessed"
    def __post_init__(self):
        if not self.claim_id.strip() or not self.statement.strip() or not self.reference_ids: raise ValueError("claim requires references")
        if self.confidence not in {"unassessed", "supported", "mixed", "contradicted"}: raise ValueError("invalid confidence label")

class LiteratureCatalog:
    def __init__(self): self.references = {}; self.claims = {}
    def add_reference(self, item):
        if not isinstance(item, LiteratureReference): raise TypeError("LiteratureReference required")
        self.references[item.reference_id] = item
    def add_claim(self, item):
        if not isinstance(item, LiteratureClaim): raise TypeError("LiteratureClaim required")
        if any(ref not in self.references for ref in item.reference_ids): raise KeyError("unknown reference ID")
        self.claims[item.claim_id] = item

__all__ = ["LiteratureReference", "LiteratureClaim", "LiteratureCatalog"]
