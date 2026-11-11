"""In-process security signal aggregation and temporary principal lockout."""
from __future__ import annotations
import threading, time
from collections import defaultdict, deque
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SecurityEvent:
    principal_id: str
    event_type: str
    timestamp: float
    severity: str = "warning"
    def __post_init__(self):
        if not self.principal_id.strip() or not self.event_type.strip(): raise ValueError("principal and event type required")
        if self.severity not in {"info", "warning", "high", "critical"}: raise ValueError("invalid severity")

class SecurityMonitor:
    def __init__(self, *, window_seconds: float = 60.0, threshold: int = 5, lockout_seconds: float = 120.0):
        if window_seconds <= 0 or threshold < 1 or lockout_seconds <= 0: raise ValueError("invalid monitor limits")
        self.window_seconds, self.threshold, self.lockout_seconds = window_seconds, threshold, lockout_seconds
        self._events = defaultdict(deque); self._locked = {}; self._lock = threading.RLock()
    def record(self, event: SecurityEvent, *, now: float | None = None) -> bool:
        if not isinstance(event, SecurityEvent): raise TypeError("event must be SecurityEvent")
        current = time.time() if now is None else float(now)
        with self._lock:
            bucket = self._events[event.principal_id]
            bucket.append(current)
            while bucket and current - bucket[0] > self.window_seconds: bucket.popleft()
            if event.severity in {"high", "critical"} or len(bucket) >= self.threshold:
                self._locked[event.principal_id] = current + self.lockout_seconds
            return self.is_locked(event.principal_id, now=current)
    def is_locked(self, principal_id: str, *, now: float | None = None) -> bool:
        current = time.time() if now is None else float(now)
        with self._lock:
            expiry = self._locked.get(principal_id, 0.0)
            if current >= expiry:
                self._locked.pop(principal_id, None)
                return False
            return True
    def unlock(self, principal_id: str):
        with self._lock: return self._locked.pop(principal_id, None) is not None

__all__ = ["SecurityEvent", "SecurityMonitor"]
