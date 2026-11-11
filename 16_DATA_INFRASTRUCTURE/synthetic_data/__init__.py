"""Synthetic sample provenance and review gates."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class SyntheticStatus(str, Enum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"

@dataclass(frozen=True, slots=True)
class SyntheticSample:
    sample_id: str
    content: str
    generator_id: str
    prompt_digest: str
    status: SyntheticStatus = SyntheticStatus.CANDIDATE
    def __post_init__(self):
        if any(not x.strip() for x in (self.sample_id, self.content, self.generator_id)): raise ValueError("synthetic sample fields required")
        if len(self.prompt_digest) != 64: raise ValueError("prompt_digest must be SHA-256")

@dataclass(frozen=True, slots=True)
class SyntheticReview:
    sample_id: str
    reviewer_id: str
    status: SyntheticStatus
    reason: str
    def __post_init__(self):
        if self.status is SyntheticStatus.CANDIDATE: raise ValueError("review must be terminal")
        if not self.reviewer_id.strip() or not self.reason.strip(): raise ValueError("reviewer and reason required")

def eligible_for_training(sample: SyntheticSample, review: SyntheticReview | None) -> bool:
    return isinstance(sample, SyntheticSample) and review is not None and review.sample_id == sample.sample_id and review.status is SyntheticStatus.APPROVED

__all__ = ["SyntheticStatus", "SyntheticSample", "SyntheticReview", "eligible_for_training"]
