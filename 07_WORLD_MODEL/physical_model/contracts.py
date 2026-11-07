"""Typed physical state observations in SI units."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _triple(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    result = tuple(float(x) for x in value)
    if any(not math.isfinite(x) for x in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True, slots=True)
class PhysicalState:
    entity_id: str
    coordinate_frame: str
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    acceleration_m_s2: tuple[float, float, float]
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_id: str = ""
    source_digest: str = ""

    def __post_init__(self):
        for name in ("entity_id", "coordinate_frame", "source_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("position_m", "velocity_m_s", "acceleration_m_s2"):
            object.__setattr__(self, name, _triple(getattr(self, name), name))
        if not isinstance(self.observed_at, datetime) or self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        if not isinstance(self.source_digest, str) or len(self.source_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_digest
        ):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")
