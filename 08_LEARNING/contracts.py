"""Immutable contracts for ADA's provenance-aware learning system.

The core learning plane is framework-agnostic. It records experience and
candidate learning artifacts; it does not mutate model weights or execute
environment actions.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = "1.0.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _digest(value: Any, name: str = "source_digest") -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def freeze_value(value: Any, path: str = "value") -> Any:
    """Deeply freeze finite JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            frozen[key] = freeze_value(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{path} must be JSON-compatible, got {type(value).__name__}")


def to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    return value


def digest_json(value: Any) -> str:
    raw = json.dumps(
        to_primitive(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ExperienceKind(str, Enum):
    SUPERVISED = "supervised"
    SELF_SUPERVISED = "self_supervised"
    REINFORCEMENT = "reinforcement"
    IMITATION = "imitation"
    OBSERVATION = "observation"
    FEEDBACK = "feedback"
    WORLD_MODEL = "world_model"


class ExperienceStatus(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class OutcomeKind(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ArtifactStatus(str, Enum):
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class ExperienceRecord:
    """One immutable learning event with source and task provenance."""
    kind: ExperienceKind
    domain: str
    task_id: str
    source_kind: str
    source_id: str
    source_digest: str
    payload: Mapping[str, Any]
    outcome: OutcomeKind = OutcomeKind.UNKNOWN
    quality_score: float | None = None
    importance: float = 1.0
    status: ExperienceStatus = ExperienceStatus.CANDIDATE
    tags: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("domain", "task_id", "source_kind", "source_id", "experience_id"):
            _text(getattr(self, name), name)
        if not isinstance(self.kind, ExperienceKind):
            raise TypeError("kind must be ExperienceKind")
        if not isinstance(self.outcome, OutcomeKind):
            raise TypeError("outcome must be OutcomeKind")
        if not isinstance(self.status, ExperienceStatus):
            raise TypeError("status must be ExperienceStatus")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported learning schema: {self.schema_version}")
        _digest(self.source_digest)
        if not isinstance(self.payload, Mapping) or not self.payload:
            raise ValueError("payload must be a non-empty mapping")
        object.__setattr__(self, "payload", freeze_value(self.payload, "payload"))
        if isinstance(self.importance, bool) or not isinstance(self.importance, (int, float)):
            raise TypeError("importance must be numeric")
        if not math.isfinite(float(self.importance)) or not 0.0 <= float(self.importance) <= 1.0:
            raise ValueError("importance must be finite and in [0, 1]")
        if self.quality_score is not None:
            if isinstance(self.quality_score, bool) or not isinstance(self.quality_score, (int, float)):
                raise TypeError("quality_score must be numeric or None")
            if not math.isfinite(float(self.quality_score)) or not 0.0 <= float(self.quality_score) <= 1.0:
                raise ValueError("quality_score must be finite and in [0, 1]")
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        tags = tuple(self.tags)
        if any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError("tags must contain non-empty strings")
        object.__setattr__(self, "tags", tags)

    @property
    def digest(self) -> str:
        return digest_json({
            "schema_version": self.schema_version,
            "experience_id": self.experience_id,
            "kind": self.kind,
            "domain": self.domain,
            "task_id": self.task_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_digest": self.source_digest,
            "payload": self.payload,
            "outcome": self.outcome,
            "quality_score": self.quality_score,
            "importance": self.importance,
            "status": self.status,
            "tags": self.tags,
            "created_at": self.created_at,
        })


@dataclass(frozen=True, slots=True)
class LearningProposal:
    """A proposed change to a model, adapter, skill or strategy artifact."""
    target: str
    parent_digest: str
    proposal: Mapping[str, Any]
    rationale: str
    proposer_id: str
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("target", "rationale", "proposer_id", "proposal_id"):
            _text(getattr(self, name), name)
        _digest(self.parent_digest, "parent_digest")
        if not isinstance(self.proposal, Mapping) or not self.proposal:
            raise ValueError("proposal must be a non-empty mapping")
        object.__setattr__(self, "proposal", freeze_value(self.proposal, "proposal"))
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")

    @property
    def digest(self) -> str:
        return digest_json({
            "target": self.target, "parent_digest": self.parent_digest,
            "proposal": self.proposal, "rationale": self.rationale,
            "proposer_id": self.proposer_id, "proposal_id": self.proposal_id,
            "created_at": self.created_at,
        })


@dataclass(frozen=True, slots=True)
class MetricSet:
    """Named, finite evaluation metrics; larger values are not assumed better."""
    evaluator_id: str
    dataset_digest: str
    metrics: Mapping[str, float]
    sample_count: int
    evaluated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _text(self.evaluator_id, "evaluator_id")
        _digest(self.dataset_digest, "dataset_digest")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise ValueError("metrics must be a non-empty mapping")
        frozen = {}
        for name, value in self.metrics.items():
            _text(name, "metric name")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"metric {name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {name} must be finite")
            frozen[name] = float(value)
        object.__setattr__(self, "metrics", MappingProxyType(frozen))
        if not isinstance(self.evaluated_at, datetime) or self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")

    @property
    def digest(self) -> str:
        return digest_json({
            "evaluator_id": self.evaluator_id, "dataset_digest": self.dataset_digest,
            "metrics": self.metrics, "sample_count": self.sample_count,
            "evaluated_at": self.evaluated_at,
        })


@dataclass(frozen=True, slots=True)
class LearningArtifact:
    """Immutable version descriptor; the underlying weights live elsewhere."""
    target: str
    version: int
    artifact_digest: str
    parent_digest: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: ArtifactStatus = ArtifactStatus.PROPOSED
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _text(self.target, "target")
        _digest(self.artifact_digest, "artifact_digest")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("version must be a positive integer")
        if self.parent_digest is not None:
            _digest(self.parent_digest, "parent_digest")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", freeze_value(self.metadata, "metadata"))
        if not isinstance(self.status, ArtifactStatus):
            raise TypeError("status must be ArtifactStatus")
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class LearningDecision:
    accepted: bool
    reason: str
    parent_digest: str
    candidate_digest: str
    baseline_metrics_digest: str
    candidate_metrics_digest: str
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decided_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be bool")
        _text(self.reason, "reason")
        for value, name in (
            (self.parent_digest, "parent_digest"),
            (self.candidate_digest, "candidate_digest"),
            (self.baseline_metrics_digest, "baseline_metrics_digest"),
            (self.candidate_metrics_digest, "candidate_metrics_digest"),
        ):
            _digest(value, name)
        if not isinstance(self.decided_at, datetime) or self.decided_at.tzinfo is None or self.decided_at.utcoffset() is None:
            raise ValueError("decided_at must be timezone-aware")
