"""Versioned, immutable intermediate-representation contracts for ADA."""

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


def freeze_value(value: Any, *, path: str = "value") -> Any:
    """Freeze finite JSON-compatible data without changing its meaning."""
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
            frozen[key] = freeze_value(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain JSON-compatible values, got {type(value).__name__}"
    )


def to_primitive(value: Any) -> Any:
    """Convert immutable contracts to plain JSON-compatible Python values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("serialized datetimes must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {key: to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    return value


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


class RepresentationKind(str, Enum):
    TOKEN = "token"
    VECTOR = "vector"
    SYMBOLIC = "symbolic"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    RELATIONAL = "relational"
    MULTIMODAL = "multimodal"


class RepresentationStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class RepresentationResultStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Representation:
    """One typed projection of a perception observation."""

    kind: RepresentationKind
    modality: str
    source_input_id: str
    source_content_sha256: str
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: RepresentationStatus = RepresentationStatus.COMPLETE
    schema_version: str = SCHEMA_VERSION
    representation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RepresentationKind):
            raise TypeError("kind must be a RepresentationKind")
        if not isinstance(self.status, RepresentationStatus):
            raise TypeError("status must be a RepresentationStatus")
        if not isinstance(self.modality, str) or not self.modality.strip():
            raise ValueError("modality must be a non-empty string")
        if not isinstance(self.source_input_id, str) or not self.source_input_id.strip():
            raise ValueError("source_input_id must be non-empty")
        _validate_sha256(self.source_content_sha256, "source_content_sha256")
        if not isinstance(self.representation_id, str) or not self.representation_id.strip():
            raise ValueError("representation_id must be non-empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported representation schema_version: {self.schema_version}"
            )
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if not isinstance(self.payload, Mapping) or not self.payload:
            raise ValueError("payload must be a non-empty mapping")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "payload", freeze_value(self.payload, path="payload"))
        object.__setattr__(self, "metadata", freeze_value(self.metadata, path="metadata"))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible envelope."""
        return to_primitive({
            "schema_version": self.schema_version,
            "representation_id": self.representation_id,
            "kind": self.kind,
            "modality": self.modality,
            "source_input_id": self.source_input_id,
            "source_content_sha256": self.source_content_sha256,
            "status": self.status,
            "created_at": self.created_at,
            "payload": self.payload,
            "metadata": self.metadata,
        })

    @property
    def representation_digest(self) -> str:
        """Digest semantic representation content, excluding generated ID/time."""
        stable = self.to_dict()
        stable.pop("representation_id")
        stable.pop("created_at")
        raw = json.dumps(
            stable,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Representation":
        if not isinstance(value, Mapping):
            raise TypeError("representation document must be a mapping")
        try:
            return cls(
                schema_version=value["schema_version"],
                representation_id=value["representation_id"],
                kind=RepresentationKind(value["kind"]),
                modality=value["modality"],
                source_input_id=value["source_input_id"],
                source_content_sha256=value["source_content_sha256"],
                status=RepresentationStatus(value["status"]),
                created_at=datetime.fromisoformat(value["created_at"]),
                payload=value["payload"],
                metadata=value.get("metadata", {}),
            )
        except KeyError as exc:
            raise ValueError(f"representation document is missing {exc.args[0]}") from exc


@dataclass(frozen=True, slots=True)
class RepresentationResult:
    """All canonical projections created for one perception input."""

    input_id: str
    correlation_id: str | None
    modality: str
    source_content_sha256: str
    status: RepresentationResultStatus
    representations: tuple[Representation, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.input_id, str) or not self.input_id.strip():
            raise ValueError("input_id must be non-empty")
        if not isinstance(self.modality, str) or not self.modality.strip():
            raise ValueError("modality must be non-empty")
        _validate_sha256(self.source_content_sha256, "source_content_sha256")
        if not isinstance(self.status, RepresentationResultStatus):
            raise TypeError("status must be a RepresentationResultStatus")
        items = tuple(self.representations)
        if any(not isinstance(item, Representation) for item in items):
            raise TypeError("representations must contain Representation values")
        if self.status in {
            RepresentationResultStatus.COMPLETE,
            RepresentationResultStatus.PARTIAL,
        } and not items:
            raise ValueError("successful representation results require representations")
        if self.status in {
            RepresentationResultStatus.UNSUPPORTED,
            RepresentationResultStatus.REJECTED,
            RepresentationResultStatus.FAILED,
        } and items:
            raise ValueError("unsuccessful representation results cannot contain outputs")
        for item in items:
            if item.source_input_id != self.input_id:
                raise ValueError("representation source_input_id does not match result")
            if item.source_content_sha256 != self.source_content_sha256:
                raise ValueError("representation digest does not match result")
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        object.__setattr__(self, "representations", items)
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "warnings", tuple(self.warnings))
