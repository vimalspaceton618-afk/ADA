"""Thread-safe in-memory memory store. Persistence is intentionally an adapter boundary."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime
from typing import Iterable

from .contracts import MemoryRecord, MemoryStatus, utc_now


class MemoryConflictError(RuntimeError):
    """Raised when an optimistic version check fails."""


class MemoryNotFoundError(KeyError):
    """Raised when a memory identifier is unknown."""


class InMemoryMemoryStore:
    def __init__(self, *, max_records: int = 100_000) -> None:
        if type(max_records) is not int or max_records < 1:
            raise ValueError("max_records must be a positive integer")
        self._max_records = max_records
        self._lock = threading.RLock()
        self._records: dict[str, MemoryRecord] = {}
        self._versions: dict[str, int] = {}

    def put(self, record: MemoryRecord, *, expected_version: int | None = None) -> int:
        if not isinstance(record, MemoryRecord):
            raise TypeError("record must be a MemoryRecord")
        with self._lock:
            exists = record.memory_id in self._records
            current = self._versions.get(record.memory_id, 0)
            if expected_version is not None and expected_version != current:
                raise MemoryConflictError(f"expected version {expected_version}, found {current}")
            if not exists and len(self._records) >= self._max_records:
                raise OverflowError("memory store capacity reached")
            if exists and record.created_at != self._records[record.memory_id].created_at:
                raise ValueError("updates must preserve created_at")
            version = current + 1
            self._records[record.memory_id] = record
            self._versions[record.memory_id] = version
            return version

    def get(self, memory_id: str, *, include_deleted: bool = False) -> MemoryRecord:
        with self._lock:
            record = self._records.get(memory_id)
            if record is None or (record.status is MemoryStatus.DELETED and not include_deleted):
                raise MemoryNotFoundError(memory_id)
            return record

    def version(self, memory_id: str) -> int:
        with self._lock:
            if memory_id not in self._records:
                raise MemoryNotFoundError(memory_id)
            return self._versions[memory_id]

    def list_records(self, *, namespace: str = "default", include_archived: bool = False,
                     include_deleted: bool = False) -> tuple[MemoryRecord, ...]:
        with self._lock:
            items = []
            for record in self._records.values():
                if record.namespace != namespace:
                    continue
                if record.status is MemoryStatus.DELETED and not include_deleted:
                    continue
                if record.status is MemoryStatus.ARCHIVED and not include_archived:
                    continue
                items.append(record)
            return tuple(sorted(items, key=lambda item: (item.created_at, item.memory_id)))

    def replace_status(self, memory_id: str, status: MemoryStatus, *,
                       expected_version: int | None = None, at: datetime | None = None) -> int:
        if not isinstance(status, MemoryStatus):
            raise TypeError("status must be MemoryStatus")
        current = self.get(memory_id, include_deleted=True)
        if current.status is MemoryStatus.DELETED and status is not MemoryStatus.DELETED:
            raise ValueError("deleted memory is terminal and cannot be restored")
        updated = replace(current, status=status, updated_at=at or utc_now())
        return self.put(updated, expected_version=expected_version)

    def delete(self, memory_id: str, *, expected_version: int | None = None) -> int:
        return self.replace_status(memory_id, MemoryStatus.DELETED, expected_version=expected_version)

    def purge_expired(self, *, at: datetime | None = None, namespace: str | None = None) -> int:
        now = at or utc_now()
        with self._lock:
            targets = [
                item for item in self._records.values()
                if item.status is MemoryStatus.ACTIVE and item.expires_at is not None
                and item.expires_at <= now and (namespace is None or item.namespace == namespace)
            ]
        for item in targets:
            self.replace_status(item.memory_id, MemoryStatus.DELETED)
        return len(targets)

    def count(self, *, namespace: str = "default") -> int:
        return len(self.list_records(namespace=namespace, include_archived=True))
