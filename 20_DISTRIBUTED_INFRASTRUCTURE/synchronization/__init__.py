"""In-process version barrier and fencing tokens for adapter-backed workers."""
from __future__ import annotations
import threading

class FencingTokenRegistry:
    def __init__(self): self._tokens = {}; self._lock = threading.Lock()
    def issue(self, resource: str):
        if not resource.strip(): raise ValueError("resource required")
        with self._lock:
            next_value = self._tokens.get(resource, 0) + 1
            self._tokens[resource] = next_value
            return next_value
    def valid(self, resource: str, token: int):
        with self._lock: return self._tokens.get(resource) == token

class VersionBarrier:
    def __init__(self, workers: int):
        if type(workers) is not int or workers < 1: raise ValueError("workers must be positive")
        self.workers = workers; self._arrived = set(); self._condition = threading.Condition(); self.version = 0
    def arrive(self, worker_id: str, *, version: int, timeout: float | None = None):
        with self._condition:
            if version != self.version: raise ValueError("worker version does not match barrier version")
            self._arrived.add(worker_id)
            if len(self._arrived) == self.workers:
                self._arrived.clear(); self.version += 1; self._condition.notify_all(); return self.version
            ok = self._condition.wait_for(lambda: self.version > version, timeout=timeout)
            if not ok: raise TimeoutError("barrier timed out")
            return self.version

__all__ = ["FencingTokenRegistry", "VersionBarrier"]
