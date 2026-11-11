"""Integrity-checked system update manifest and two-phase apply protocol."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import hashlib, threading

class UpdateState(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"

@dataclass(frozen=True, slots=True)
class UpdateManifest:
    update_id: str
    component: str
    from_version: str
    to_version: str
    payload_digest: str
    def __post_init__(self):
        if any(not x.strip() for x in (self.update_id, self.component, self.from_version, self.to_version)):
            raise ValueError("update identity fields required")
        if len(self.payload_digest) != 64 or any(c not in "0123456789abcdef" for c in self.payload_digest):
            raise ValueError("payload_digest must be lowercase SHA-256")

@dataclass(frozen=True, slots=True)
class UpdateRecord:
    manifest: UpdateManifest
    state: UpdateState
    detail: str = ""

class UpdateManager:
    _allowed = {
        UpdateState.PROPOSED: {UpdateState.VERIFIED, UpdateState.REJECTED},
        UpdateState.VERIFIED: {UpdateState.APPLIED, UpdateState.REJECTED},
        UpdateState.APPLIED: {UpdateState.ROLLED_BACK},
        UpdateState.REJECTED: set(), UpdateState.ROLLED_BACK: set(),
    }
    def __init__(self): self._items = {}; self._lock = threading.RLock()
    def propose(self, manifest: UpdateManifest):
        if not isinstance(manifest, UpdateManifest): raise TypeError("manifest must be UpdateManifest")
        record = UpdateRecord(manifest, UpdateState.PROPOSED)
        with self._lock:
            if manifest.update_id in self._items: raise ValueError("update ID already exists")
            self._items[manifest.update_id] = record
        return record
    def transition(self, update_id: str, state: UpdateState, *, detail: str = ""):
        from dataclasses import replace
        with self._lock:
            current = self._items[update_id]
            if state not in self._allowed[current.state]: raise ValueError("invalid update transition")
            updated = replace(current, state=state, detail=detail)
            self._items[update_id] = updated
            return updated

__all__ = ["UpdateState", "UpdateManifest", "UpdateRecord", "UpdateManager"]
