"""Typed, provenance-preserving contracts for ADA perception.

Perception outputs describe observations and extracted evidence. They do not
assert that an unimplemented neural model has understood an input.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def freeze_metadata(value: Any, *, path: str = "metadata") -> Any:
    """Deep-freeze JSON-compatible metadata and reject ambiguous values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = freeze_metadata(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_metadata(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{path} must contain JSON-compatible values, got {type(value).__name__}")


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"


class PerceptionStatus(str, Enum):
    OBSERVED = "observed"
    PARTIAL = "partial"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PerceptionInput:
    modality: Modality
    content: str | bytes | Mapping[str, Any]
    source: str = "user"
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str | None = None
    content_type: str | None = None
    received_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.modality, Modality):
            raise TypeError("modality must be a Modality")
        if not isinstance(self.input_id, str) or not self.input_id.strip():
            raise ValueError("input_id must be non-empty")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be non-empty")
        if self.correlation_id is not None and (
            not isinstance(self.correlation_id, str) or not self.correlation_id.strip()
        ):
            raise ValueError("correlation_id must be non-empty or None")
        if not isinstance(self.received_at, datetime):
            raise TypeError("received_at must be a datetime")
        if not isinstance(self.content, (str, bytes, Mapping)):
            raise TypeError("content must be text, bytes, or a mapping")
        if isinstance(self.content, Mapping):
            object.__setattr__(self, "content", freeze_metadata(self.content, path="content"))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """A source-grounded character or byte range; offsets are half-open."""

    start: int
    end: int
    text: str
    label: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int:
            raise TypeError("evidence offsets must be integers")
        if self.start < 0 or self.end < self.start:
            raise ValueError("evidence offsets must be a valid half-open range")
        if not isinstance(self.text, str) or not isinstance(self.label, str) or not self.label:
            raise ValueError("evidence text and label must be strings; label cannot be empty")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise TypeError("confidence must be a number or None")
            if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class Observation:
    modality: Modality
    status: PerceptionStatus
    summary: str
    source_input_id: str
    content_sha256: str
    observed_at: datetime = field(default_factory=utc_now)
    extracted_text: str | None = None
    evidence: tuple[EvidenceSpan, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.modality, Modality) or not isinstance(self.status, PerceptionStatus):
            raise TypeError("modality and status must use their enum types")
        if not self.source_input_id or not self.summary:
            raise ValueError("source_input_id and summary must be non-empty")
        if len(self.content_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "attributes", freeze_metadata(self.attributes, path="observation.attributes"))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    input_id: str
    correlation_id: str | None
    modality: Modality
    status: PerceptionStatus
    observation: Observation | None
    content_sha256: str
    processed_at: datetime = field(default_factory=utc_now)
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.input_id:
            raise ValueError("input_id must be non-empty")
        if self.status is PerceptionStatus.OBSERVED and self.observation is None:
            raise ValueError("OBSERVED result requires an observation")
        if self.observation is not None and self.observation.source_input_id != self.input_id:
            raise ValueError("observation source_input_id must match result input_id")


def content_digest(content: str | bytes | Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest without serializing arbitrary objects."""
    if isinstance(content, str):
        raw = content.encode("utf-8")
    elif isinstance(content, bytes):
        raw = content
    else:
        # Stable representation for the JSON-compatible, frozen mapping tree.
        import json
        raw = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
