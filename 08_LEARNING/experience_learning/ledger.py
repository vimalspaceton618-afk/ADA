"""Validated experience intake and provenance-aware ledger."""
from __future__ import annotations

import threading

from ..contracts import ExperienceRecord, ExperienceStatus


class ExperienceLedger:
    """Bounded process-local ledger; only explicitly validated records are replayable."""

    def __init__(self, capacity: int = 50000) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._records: dict[str, ExperienceRecord] = {}
        self._lock = threading.RLock()

    def ingest(self, record: ExperienceRecord, *, validated: bool = False) -> ExperienceRecord:
        if not isinstance(record, ExperienceRecord):
            raise TypeError("record must be ExperienceRecord")
        if type(validated) is not bool:
            raise TypeError("validated must be bool")
        if validated and record.status is not ExperienceStatus.VALIDATED:
            raise ValueError("validated ingestion requires a VALIDATED record")
        if not validated and record.status is ExperienceStatus.VALIDATED:
            raise ValueError("candidate ingestion cannot silently mark data as validated")
        with self._lock:
            existing = self._records.get(record.experience_id)
            if existing is not None:
                if existing.digest != record.digest:
                    raise ValueError("experience ID collision with different content")
                return existing
            if len(self._records) >= self.capacity:
                raise OverflowError("experience ledger capacity reached; export or prune explicitly")
            self._records[record.experience_id] = record
            return record

    def validate(self, experience_id: str) -> ExperienceRecord:
        with self._lock:
            current = self._records.get(experience_id)
            if current is None:
                raise KeyError(experience_id)
            if current.status is ExperienceStatus.REJECTED:
                raise ValueError("rejected experience cannot be validated")
            validated = ExperienceRecord(
                kind=current.kind, domain=current.domain, task_id=current.task_id,
                source_kind=current.source_kind, source_id=current.source_id,
                source_digest=current.source_digest, payload=current.payload,
                outcome=current.outcome, quality_score=current.quality_score,
                importance=current.importance, status=ExperienceStatus.VALIDATED,
                tags=current.tags, schema_version=current.schema_version,
                experience_id=current.experience_id, created_at=current.created_at,
            )
            self._records[experience_id] = validated
            return validated

    def reject(self, experience_id: str) -> ExperienceRecord:
        with self._lock:
            current = self._records.get(experience_id)
            if current is None:
                raise KeyError(experience_id)
            rejected = ExperienceRecord(
                kind=current.kind, domain=current.domain, task_id=current.task_id,
                source_kind=current.source_kind, source_id=current.source_id,
                source_digest=current.source_digest, payload=current.payload,
                outcome=current.outcome, quality_score=current.quality_score,
                importance=current.importance, status=ExperienceStatus.REJECTED,
                tags=current.tags, schema_version=current.schema_version,
                experience_id=current.experience_id, created_at=current.created_at,
            )
            self._records[experience_id] = rejected
            return rejected

    def get(self, experience_id: str) -> ExperienceRecord | None:
        with self._lock:
            return self._records.get(experience_id)

    def list(self, *, domain: str | None = None) -> tuple[ExperienceRecord, ...]:
        with self._lock:
            items = self._records.values()
            if domain is not None:
                items = (item for item in items if item.domain == domain)
            return tuple(sorted(items, key=lambda item: (item.created_at, item.experience_id)))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
