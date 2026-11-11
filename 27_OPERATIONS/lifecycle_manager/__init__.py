"""Managed service lifecycle with guarded state transitions."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import threading, time

class ServiceState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"

@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    service_id: str
    state: ServiceState
    generation: int
    changed_at: float
    detail: str = ""

class LifecycleManager:
    _allowed = {
        ServiceState.CREATED: {ServiceState.STARTING, ServiceState.FAILED},
        ServiceState.STARTING: {ServiceState.RUNNING, ServiceState.FAILED, ServiceState.STOPPING},
        ServiceState.RUNNING: {ServiceState.STOPPING, ServiceState.FAILED},
        ServiceState.STOPPING: {ServiceState.STOPPED, ServiceState.FAILED},
        ServiceState.STOPPED: {ServiceState.STARTING},
        ServiceState.FAILED: {ServiceState.STARTING, ServiceState.STOPPED},
    }
    def __init__(self): self._items = {}; self._lock = threading.RLock()
    def create(self, service_id: str, *, now: float | None = None):
        if not service_id.strip(): raise ValueError("service_id required")
        record = LifecycleRecord(service_id, ServiceState.CREATED, 0, time.time() if now is None else float(now))
        with self._lock:
            if service_id in self._items: raise ValueError("service already exists")
            self._items[service_id] = record
        return record
    def transition(self, service_id: str, state: ServiceState, *, detail: str = "", now: float | None = None):
        from dataclasses import replace
        stamp = time.time() if now is None else float(now)
        with self._lock:
            current = self._items[service_id]
            if state not in self._allowed[current.state]: raise ValueError("invalid service lifecycle transition")
            generation = current.generation + (1 if current.state in {ServiceState.STOPPED, ServiceState.FAILED} and state is ServiceState.STARTING else 0)
            updated = replace(current, state=state, generation=generation, changed_at=stamp, detail=detail)
            self._items[service_id] = updated
            return updated
    def get(self, service_id: str):
        with self._lock: return self._items[service_id]

__all__ = ["ServiceState", "LifecycleRecord", "LifecycleManager"]
