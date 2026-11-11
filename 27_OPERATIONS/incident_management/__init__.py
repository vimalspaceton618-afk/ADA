"""Incident response workflow, severity escalation and postmortem records."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import time, threading

class IncidentPhase(str, Enum):
    DETECTED = "detected"
    TRIAGED = "triaged"
    CONTAINED = "contained"
    REMEDIATED = "remediated"
    CLOSED = "closed"

@dataclass(frozen=True, slots=True)
class IncidentRecord:
    incident_id: str
    title: str
    severity: str
    phase: IncidentPhase
    owner: str | None
    created_at: float
    updated_at: float
    def __post_init__(self):
        if not self.incident_id.strip() or not self.title.strip(): raise ValueError("incident identity required")
        if self.severity not in {"low", "medium", "high", "critical"}: raise ValueError("invalid severity")

@dataclass(frozen=True, slots=True)
class Postmortem:
    incident_id: str
    root_cause: str
    impact: str
    corrective_actions: tuple[str, ...]
    published_at: float
    def __post_init__(self):
        if not self.root_cause.strip() or not self.impact.strip() or not self.corrective_actions:
            raise ValueError("postmortem must include cause, impact and actions")

class IncidentWorkflow:
    _next = {
        IncidentPhase.DETECTED: {IncidentPhase.TRIAGED},
        IncidentPhase.TRIAGED: {IncidentPhase.CONTAINED},
        IncidentPhase.CONTAINED: {IncidentPhase.REMEDIATED},
        IncidentPhase.REMEDIATED: {IncidentPhase.CLOSED},
        IncidentPhase.CLOSED: set(),
    }
    def __init__(self): self._items = {}; self._lock = threading.RLock()
    def create(self, incident_id: str, title: str, severity: str, *, now: float | None = None):
        stamp = time.time() if now is None else float(now)
        record = IncidentRecord(incident_id, title, severity, IncidentPhase.DETECTED, None, stamp, stamp)
        with self._lock:
            if incident_id in self._items: raise ValueError("incident already exists")
            self._items[incident_id] = record
        return record
    def transition(self, incident_id: str, phase: IncidentPhase, *, owner: str | None = None, now: float | None = None):
        from dataclasses import replace
        stamp = time.time() if now is None else float(now)
        with self._lock:
            current = self._items[incident_id]
            if phase not in self._next[current.phase]: raise ValueError("invalid incident transition")
            updated = replace(current, phase=phase, owner=owner or current.owner, updated_at=stamp)
            self._items[incident_id] = updated
            return updated

__all__ = ["IncidentPhase", "IncidentRecord", "Postmortem", "IncidentWorkflow"]
