"""Feedback review gate; user labels never become training targets automatically."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from .. import FeedbackRecord, FeedbackLabel

class FeedbackDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"

@dataclass(frozen=True, slots=True)
class FeedbackReview:
    feedback_id: str
    reviewer_id: str
    disposition: FeedbackDisposition
    reason: str
    def __post_init__(self):
        if not self.feedback_id.strip() or not self.reviewer_id.strip() or not self.reason.strip():
            raise ValueError("feedback review provenance is required")

class FeedbackReviewQueue:
    def __init__(self): self._pending = {}; self._reviews = {}
    def submit(self, record: FeedbackRecord):
        if not isinstance(record, FeedbackRecord): raise TypeError("record must be FeedbackRecord")
        self._pending.setdefault(record.feedback_id, record)
        return self._pending[record.feedback_id]
    def review(self, review: FeedbackReview):
        if not isinstance(review, FeedbackReview): raise TypeError("review must be FeedbackReview")
        if review.feedback_id not in self._pending: raise KeyError(review.feedback_id)
        if review.feedback_id in self._reviews: raise ValueError("feedback already reviewed")
        self._reviews[review.feedback_id] = review
        return review
    def training_label(self, feedback_id: str):
        review = self._reviews.get(feedback_id)
        if review is None or review.disposition is not FeedbackDisposition.ACCEPTED: return None
        return self._pending[feedback_id].label

__all__ = ["FeedbackDisposition", "FeedbackReview", "FeedbackReviewQueue"]
