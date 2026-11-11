"""Release channel metadata and immutable promotion records."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import threading

class ReleaseChannel(str, Enum):
    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    STABLE = "stable"
    DEPRECATED = "deprecated"

@dataclass(frozen=True, slots=True)
class Release:
    release_id: str
    version: str
    artifact_digest: str
    channel: ReleaseChannel
    evaluation_digest: str
    safety_digest: str
    def __post_init__(self):
        if any(not x.strip() for x in (self.release_id, self.version)): raise ValueError("release ID/version required")
        for value in (self.artifact_digest, self.evaluation_digest, self.safety_digest):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value): raise ValueError("release references must be SHA-256")

class ReleaseRegistry:
    def __init__(self): self._items = {}; self._lock = threading.RLock()
    def register(self, release: Release):
        if not isinstance(release, Release): raise TypeError("release must be Release")
        with self._lock:
            if release.release_id in self._items and self._items[release.release_id] != release:
                raise ValueError("release IDs are immutable")
            self._items[release.release_id] = release
        return release
    def promote(self, release_id: str, channel: ReleaseChannel, *, gates_passed: bool):
        if gates_passed is not True: raise PermissionError("release gates must pass")
        if channel is ReleaseChannel.EXPERIMENTAL: raise ValueError("promotion target must not be experimental")
        with self._lock:
            current = self._items[release_id]
            updated = Release(current.release_id, current.version, current.artifact_digest,
                              channel, current.evaluation_digest, current.safety_digest)
            self._items[release_id] = updated
            return updated
    def get(self, release_id: str):
        with self._lock: return self._items[release_id]

__all__ = ["ReleaseChannel", "Release", "ReleaseRegistry"]
