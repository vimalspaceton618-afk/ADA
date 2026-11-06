"""Explicit relation records; edges are never inferred from co-occurrence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RelationEdge:
    source_representation_id: str
    target_representation_id: str
    relation_type: str
    evidence_input_ids: tuple[str, ...]
    direction: str = "directed"

    def __post_init__(self) -> None:
        if not self.source_representation_id or not self.target_representation_id:
            raise ValueError("relation endpoints must be non-empty representation IDs")
        if self.source_representation_id == self.target_representation_id:
            raise ValueError("self-relations are not supported by this contract")
        if not isinstance(self.relation_type, str) or not self.relation_type.strip():
            raise ValueError("relation_type must be non-empty")
        if self.direction not in {"directed", "undirected"}:
            raise ValueError("direction must be 'directed' or 'undirected'")
        evidence = tuple(self.evidence_input_ids)
        if not evidence or any(
            not isinstance(value, str) or not value for value in evidence
        ):
            raise ValueError("relation edges require source input provenance")
        object.__setattr__(self, "evidence_input_ids", evidence)
