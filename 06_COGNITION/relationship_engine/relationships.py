"""Typed, provenance-linked relationship records; no automatic inference."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Relation:
    subject_id: str
    predicate: str
    object_id: str
    evidence_id: str
    def __post_init__(self):
        for name in ("subject_id", "predicate", "object_id", "evidence_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.subject_id == self.object_id and self.predicate.casefold() == "causes":
            raise ValueError("self-causation is not accepted")

def build_relation(subject_id, predicate, object_id, evidence_id) -> Relation:
    return Relation(subject_id, predicate, object_id, evidence_id)
