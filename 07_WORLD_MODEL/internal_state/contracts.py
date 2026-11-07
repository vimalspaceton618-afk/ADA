"""Versioned runtime-state references separate from observed world facts."""
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ModelRuntimeState:
    episode_id: str
    step_index: int
    snapshot_id: str
    model_revision: str
    uncertainty: float
    updated_at: datetime
    latent_state_digest: str

    def __post_init__(self):
        for name in ("episode_id", "snapshot_id", "model_revision"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        if isinstance(self.uncertainty, bool) or not isinstance(self.uncertainty, (int, float)) or not 0 <= self.uncertainty <= 1:
            raise ValueError("uncertainty must be in [0,1]")
        if not isinstance(self.updated_at, datetime) or self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        if not isinstance(self.latent_state_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.latent_state_digest):
            raise ValueError("latent_state_digest must be a lowercase SHA-256 digest")
