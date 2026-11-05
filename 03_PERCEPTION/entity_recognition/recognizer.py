"""Source-grounded entity recognition contracts; no heuristic NER is enabled."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..contracts import Observation


@dataclass(frozen=True, slots=True)
class EntityMention:
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float | None = None
    canonical_id: str | None = None

    def __post_init__(self) -> None:
        if not self.text or not self.entity_type:
            raise ValueError("entity text and entity_type must be non-empty")
        if type(self.start) is not int or type(self.end) is not int or self.start < 0 or self.end < self.start:
            raise ValueError("entity offsets must be a valid half-open range")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


class EntityRecognizer(Protocol):
    def recognize(self, observation: Observation) -> tuple[EntityMention, ...]:
        """Return entity mentions grounded in the observation's source content."""
        ...
