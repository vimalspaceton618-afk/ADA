"""Scheduled maintenance run registry with explicit outcome recording."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import threading, time

class MaintenanceState(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass(frozen=True, slots=True)
class MaintenanceRun:
    run_id: str
    task: str
    scheduled_at: float
    state: MaintenanceState = MaintenanceState.SCHEDULED
    started_at: float | None = None
    finished_at: float | None = None
    detail: str = ""

class MaintenanceRegistry:
    _allowed = {
        MaintenanceState.SCHEDULED: {MaintenanceState.RUNNING, MaintenanceState.CANCELLED},
        MaintenanceState.RUNNING: {MaintenanceState.SUCCEEDED, MaintenanceState.FAILED},
        MaintenanceState.SUCCEEDED: set(), MaintenanceState.FAILED: set(), MaintenanceState.CANCELLED: set(),
    }
    def __init__(self): self._items = {}; self._lock = threading.RLock()
    def schedule(self, run_id: str, task: str, *, scheduled_at: float | None = None):
        if not run_id.strip() or not task.strip(): raise ValueError("run ID and task required")
        item = MaintenanceRun(run_id, task, time.time() if scheduled_at is None else float(scheduled_at))
        with self._lock:
            if run_id in self._items: raise ValueError("maintenance run already exists")
            self._items[run_id] = item
        return item
    def transition(self, run_id: str, state: MaintenanceState, *, detail: str = "", now: float | None = None):
        from dataclasses import replace
        stamp = time.time() if now is None else float(now)
        with self._lock:
            current = self._items[run_id]
            if state not in self._allowed[current.state]: raise ValueError("invalid maintenance transition")
            updated = replace(current, state=state,
                              started_at=stamp if state is MaintenanceState.RUNNING else current.started_at,
                              finished_at=stamp if state in {MaintenanceState.SUCCEEDED, MaintenanceState.FAILED, MaintenanceState.CANCELLED} else current.finished_at,
                              detail=detail)
            self._items[run_id] = updated
            return updated

__all__ = ["MaintenanceState", "MaintenanceRun", "MaintenanceRegistry"]
