"""Immutable, provenance-aware memory contracts for ADA."""

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


def freeze_json(value: Any, *, path: str = "value") -> Any:
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
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, path=f"{path}[{i}]") for i, item in enumerate(value))
    raise TypeError(f"{path} must contain JSON-compatible values, got {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


class MemoryKind(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemorySourceKind(str, Enum):
    PERCEPTION = "perception"
    REPRESENTATION = "representation"
    USER = "user"
    SYSTEM = "system"
    IMPORTED = "imported"


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    kind: MemoryKind
    content: Mapping[str, Any]
    source_kind: MemorySourceKind
    source_id: str
    source_digest: str
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    namespace: str = "default"
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    expires_at: datetime | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, enum_type in (("kind", MemoryKind), ("source_kind", MemorySourceKind), ("status", MemoryStatus)):
            if not isinstance(getattr(self, name), enum_type):
                raise TypeError(f"{name} must be a {enum_type.__name__}")
        for name in ("memory_id", "namespace", "source_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        _digest(self.source_digest, "source_digest")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported memory schema_version: {self.schema_version}")
        if not isinstance(self.content, Mapping) or not self.content:
            raise ValueError("content must be a non-empty mapping")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        for name in ("created_at", "updated_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.expires_at is not None and (self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None):
            raise ValueError("expires_at must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must follow created_at")
        clean_tags = tuple(dict.fromkeys(self.tags))
        if any(not isinstance(tag, str) or not tag.strip() for tag in clean_tags):
            raise ValueError("tags must be non-empty strings")
        object.__setattr__(self, "tags", clean_tags)
        object.__setattr__(self, "content", freeze_json(self.content, path="content"))
        object.__setattr__(self, "metadata", freeze_json(self.metadata, path="metadata"))

    @property
    def content_digest(self) -> str:
        raw = json.dumps(thaw_json(self.content), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        def stamp(value):
            return value.astimezone(timezone.utc).isoformat() if value else None
        return {
            "schema_version": self.schema_version, "memory_id": self.memory_id,
            "kind": self.kind.value, "content": thaw_json(self.content),
            "source_kind": self.source_kind.value, "source_id": self.source_id,
            "source_digest": self.source_digest, "namespace": self.namespace,
            "tags": list(self.tags), "metadata": thaw_json(self.metadata),
            "status": self.status.value, "created_at": stamp(self.created_at),
            "updated_at": stamp(self.updated_at), "expires_at": stamp(self.expires_at),
        }


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    namespace: str = "default"
    kinds: tuple[MemoryKind, ...] = ()
    tags: tuple[str, ...] = ()
    source_id: str | None = None
    include_archived: bool = False
    limit: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("namespace must be non-empty")
        if type(self.limit) is not int or not 1 <= self.limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        if any(not isinstance(kind, MemoryKind) for kind in self.kinds):
            raise TypeError("kinds must contain MemoryKind values")
        if any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise ValueError("tags must be non-empty strings")


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    record: MemoryRecord
    score: float
    matched_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and between 0 and 1")
