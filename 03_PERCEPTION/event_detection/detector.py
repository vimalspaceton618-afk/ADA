"""Event detection contracts with temporal and source provenance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..contracts import Observation


@dataclass(frozen=True, slots=True)
class DetectedEvent:
    event_type: str
    description: str
    source_input_id: str
    start_time_seconds: float | None = None
    end_time_seconds: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.event_type or not self.description or not self.source_input_id:
            raise ValueError("event type, description and source input id are required")
        if self.start_time_seconds is not None and self.start_time_seconds < 0:
            raise ValueError("event start time must be non-negative")
        if self.end_time_seconds is not None and self.end_time_seconds < 0:
            raise ValueError("event end time must be non-negative")
        if (
            self.start_time_seconds is not None and self.end_time_seconds is not None
            and self.end_time_seconds < self.start_time_seconds
        ):
            raise ValueError("event end time cannot precede start time")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


class EventDetector(Protocol):
    def detect(self, observation: Observation) -> tuple[DetectedEvent, ...]:
        """Detect source-grounded events; temporal units are seconds."""
        ...
