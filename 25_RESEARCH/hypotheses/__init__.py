"""Falsifiable hypothesis records and explicit review status."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class ReviewStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"

@dataclass(frozen=True, slots=True)
class Hypothesis:
    hypothesis_id: str
    claim: str
    null_claim: str
    metric: str
    threshold: float
    status: ReviewStatus = ReviewStatus.PROPOSED
    evidence_refs: tuple[str, ...] = ()
    def __post_init__(self):
        if any(not x.strip() for x in (self.hypothesis_id, self.claim, self.null_claim, self.metric)):
            raise ValueError("hypothesis requires claim, null, metric and ID")
        if self.status is ReviewStatus.ACCEPTED and not self.evidence_refs:
            raise ValueError("accepted hypothesis requires evidence refs")

__all__ = ["ReviewStatus", "Hypothesis"]
