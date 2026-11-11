"""Incident lifecycle, backup manifests and maintenance tracking."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA-256 digest")


class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class Incident:
    incident_id: str
    title: str
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.OPEN
    owner: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class IncidentManager:
    _allowed = {
        IncidentStatus.OPEN: {IncidentStatus.ACKNOWLEDGED},
        IncidentStatus.ACKNOWLEDGED: {IncidentStatus.MITIGATED},
        IncidentStatus.MITIGATED: {IncidentStatus.RESOLVED},
        IncidentStatus.RESOLVED: set(),
    }

    def __init__(self):
        self._items = {}
        self._lock = threading.RLock()

    def create(self, title: str, severity: IncidentSeverity):
        if not title.strip() or not isinstance(severity, IncidentSeverity):
            raise ValueError("title and valid severity are required")
        item = Incident(str(uuid.uuid4()), title, severity)
        with self._lock:
            self._items[item.incident_id] = item
        return item

    def transition(self, incident_id: str, status: IncidentStatus, *, owner: str | None = None):
        from dataclasses import replace
        with self._lock:
            current = self._items[incident_id]
            if status not in self._allowed[current.status]:
                raise ValueError("invalid incident transition")
            updated = replace(current, status=status, owner=owner or current.owner,
                              updated_at=time.time())
            self._items[incident_id] = updated
            return updated

    def get(self, incident_id: str):
        with self._lock:
            return self._items[incident_id]


@dataclass(frozen=True, slots=True)
class BackupManifest:
    backup_id: str
    artifact_digest: str
    data_digest: str
    created_at: float
    storage_ref: str
    verified: bool

    def __post_init__(self):
        _sha(self.artifact_digest)
        _sha(self.data_digest)
        if not self.backup_id.strip() or not self.storage_ref.strip():
            raise ValueError("backup_id and storage_ref are required")


class BackupRegistry:
    def __init__(self):
        self._items = {}
        self._lock = threading.RLock()

    def register(self, manifest: BackupManifest):
        if not isinstance(manifest, BackupManifest):
            raise TypeError("manifest must be BackupManifest")
        with self._lock:
            previous = self._items.get(manifest.backup_id)
            if previous is not None and previous != manifest:
                raise ValueError("backup ID is immutable")
            self._items[manifest.backup_id] = manifest
        return manifest

    def latest_verified(self):
        with self._lock:
            values = [x for x in self._items.values() if x.verified]
        return max(values, key=lambda x: (x.created_at, x.backup_id), default=None)


class MaintenanceStatus(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MaintenanceTask:
    task_id: str
    description: str
    status: MaintenanceStatus = MaintenanceStatus.SCHEDULED


class MaintenanceManager:
    _allowed = {
        MaintenanceStatus.SCHEDULED: {MaintenanceStatus.RUNNING},
        MaintenanceStatus.RUNNING: {MaintenanceStatus.COMPLETED, MaintenanceStatus.FAILED},
        MaintenanceStatus.COMPLETED: set(),
        MaintenanceStatus.FAILED: set(),
    }

    def __init__(self):
        self._tasks = {}
        self._lock = threading.RLock()

    def create(self, description: str):
        if not description.strip():
            raise ValueError("description is required")
        task = MaintenanceTask(str(uuid.uuid4()), description)
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def transition(self, task_id: str, status: MaintenanceStatus):
        from dataclasses import replace
        with self._lock:
            current = self._tasks[task_id]
            if status not in self._allowed[current.status]:
                raise ValueError("invalid maintenance transition")
            updated = replace(current, status=status)
            self._tasks[task_id] = updated
            return updated


__all__ = [
    "IncidentSeverity", "IncidentStatus", "Incident", "IncidentManager",
    "BackupManifest", "BackupRegistry", "MaintenanceStatus",
    "MaintenanceTask", "MaintenanceManager",
]
