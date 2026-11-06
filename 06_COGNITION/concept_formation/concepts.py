"""Create explicit concepts from caller-labelled observations."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Concept:
    label: str
    evidence_ids: tuple[str, ...]
    attributes: tuple[str, ...] = ()
    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be non-empty")
        if not self.evidence_ids or any(not x for x in self.evidence_ids):
            raise ValueError("concept requires evidence IDs")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence IDs must be unique")

def form_concept(label: str, evidence_ids, attributes=()) -> Concept:
    return Concept(label, tuple(evidence_ids), tuple(attributes))
