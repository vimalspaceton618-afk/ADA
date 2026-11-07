"""Privacy-conscious observed behavior records, not personality inference."""
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ObservedBehaviorEvent:
    """An explicitly observed action label with pseudonymous actor reference."""
    actor_ref: str
    action_label: str
    context_label: str
    observed_at: datetime
    source_id: str
    source_digest: str

    def __post_init__(self):
        for name in ("actor_ref", "action_label", "context_label", "source_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.observed_at, datetime) or self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        if not isinstance(self.source_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_digest):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")
