"""Data classification, deterministic redaction and retention decisions."""
from __future__ import annotations
import re, time
from dataclasses import dataclass
from enum import IntEnum

class DataClass(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    SENSITIVE = 3

@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    max_age_seconds: float
    allow_training: bool = False
    def __post_init__(self):
        if self.max_age_seconds <= 0: raise ValueError("max_age_seconds must be positive")

@dataclass(frozen=True, slots=True)
class ProtectedRecord:
    record_id: str
    classification: DataClass
    created_at: float
    content_digest: str
    training_eligible: bool = False
    def __post_init__(self):
        if not self.record_id.strip() or not isinstance(self.classification, DataClass):
            raise ValueError("record identity and classification are required")
        if len(self.content_digest) != 64: raise ValueError("content_digest must be SHA-256")

_PATTERNS = (
    (re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"), "[REDACTED_PHONE]"),
    (re.compile(r"(?i)\b(?:sk|ghp|github_pat|AKIA)[-_A-Za-z0-9]{12,}\b"), "[REDACTED_SECRET]"),
)

def redact_text(text: str) -> str:
    if not isinstance(text, str): raise TypeError("text must be str")
    output = text
    for pattern, replacement in _PATTERNS:
        output = pattern.sub(replacement, output)
    return output

def may_train(record: ProtectedRecord, policy: RetentionPolicy, *, now: float | None = None) -> bool:
    if not isinstance(record, ProtectedRecord) or not isinstance(policy, RetentionPolicy):
        raise TypeError("record and policy have invalid types")
    current = time.time() if now is None else float(now)
    age_ok = 0 <= current - record.created_at <= policy.max_age_seconds
    return age_ok and policy.allow_training and record.training_eligible and record.classification <= DataClass.INTERNAL

__all__ = ["DataClass", "RetentionPolicy", "ProtectedRecord", "redact_text", "may_train"]
