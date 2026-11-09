"""Bounded, deterministic stratified replay buffer for validated experience."""
from __future__ import annotations

import random
import threading
from collections import defaultdict

from ..contracts import ExperienceRecord, ExperienceStatus


class ReplayBuffer:
    """In-memory replay buffer with domain/kind stratification.

    This component is deliberately process-local. Persistence and distributed
    replay are separate adapters and are not implied by this implementation.
    """

    def __init__(self, capacity: int = 10000, *, seed: int = 0) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        self.capacity = capacity
        self._rng = random.Random(seed)
        self._items: dict[str, ExperienceRecord] = {}
        self._lock = threading.RLock()

    def add(self, record: ExperienceRecord) -> bool:
        if not isinstance(record, ExperienceRecord):
            raise TypeError("record must be ExperienceRecord")
        if record.status is not ExperienceStatus.VALIDATED:
            raise ValueError("only validated experience can enter replay")
        with self._lock:
            if record.experience_id in self._items:
                existing = self._items[record.experience_id]
                if existing.digest != record.digest:
                    raise ValueError("experience ID collision with different content")
                return False
            if len(self._items) >= self.capacity:
                self._evict_one()
            self._items[record.experience_id] = record
            return True

    def _evict_one(self) -> None:
        # Evict lowest-importance oldest item; deterministic tie breaking.
        victim = min(
            self._items.values(),
            key=lambda item: (item.importance, item.created_at, item.experience_id),
        )
        del self._items[victim.experience_id]

    def sample(self, count: int, *, domain: str | None = None) -> tuple[ExperienceRecord, ...]:
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        if domain is not None and (not isinstance(domain, str) or not domain.strip()):
            raise ValueError("domain must be a non-empty string or None")
        with self._lock:
            groups: dict[tuple[str, str], list[ExperienceRecord]] = defaultdict(list)
            for item in self._items.values():
                if domain is not None and item.domain != domain:
                    continue
                groups[(item.domain, item.kind.value)].append(item)
            if not groups:
                return ()
            keys = sorted(groups)
            pools = [sorted(groups[key], key=lambda item: (item.created_at, item.experience_id)) for key in keys]
            # Round-robin strata prevent dominant domains/kinds from monopolizing a batch.
            candidates = []
            while any(pools):
                for pool in pools:
                    if pool:
                        candidates.append(pool.pop())
            self._rng.shuffle(candidates)
            return tuple(candidates[:count])

    def get(self, experience_id: str) -> ExperienceRecord | None:
        with self._lock:
            return self._items.get(experience_id)

    def remove(self, experience_id: str) -> bool:
        with self._lock:
            return self._items.pop(experience_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def snapshot(self) -> tuple[ExperienceRecord, ...]:
        with self._lock:
            return tuple(sorted(self._items.values(), key=lambda item: (item.created_at, item.experience_id)))
