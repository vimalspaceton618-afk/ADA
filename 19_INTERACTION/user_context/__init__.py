"""Consent-scoped, expiring user context; no default training eligibility."""
from __future__ import annotations
import time, threading
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class ContextValue:
    key: str
    value: Any
    expires_at: float
    consented: bool
    classification: str = "internal"
    def __post_init__(self):
        if not self.key.strip(): raise ValueError("context key required")
        if self.classification not in {"public", "internal", "confidential", "sensitive"}:
            raise ValueError("invalid classification")

class UserContextStore:
    def __init__(self): self._values = {}; self._lock = threading.RLock()
    def set(self, user_id: str, item: ContextValue):
        if not user_id.strip() or not isinstance(item, ContextValue): raise ValueError("user and ContextValue required")
        with self._lock: self._values[(user_id, item.key)] = item
    def get(self, user_id: str, key: str, *, now: float | None = None):
        current = time.time() if now is None else float(now)
        with self._lock:
            item = self._values.get((user_id, key))
            if item is None: return None
            if current >= item.expires_at:
                self._values.pop((user_id, key), None); return None
            return item
    def delete_user(self, user_id: str):
        with self._lock:
            keys = [key for key in self._values if key[0] == user_id]
            for key in keys: self._values.pop(key, None)
            return len(keys)

__all__ = ["ContextValue", "UserContextStore"]
