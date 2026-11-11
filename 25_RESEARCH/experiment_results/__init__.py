"""Immutable, protocol-bound experiment result storage."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json

@dataclass(frozen=True, slots=True)
class ExperimentMetric:
    name: str
    value: float
    unit: str = "1"
    def __post_init__(self):
        if not self.name.strip() or not self.unit.strip(): raise ValueError("metric name/unit required")

@dataclass(frozen=True, slots=True)
class ExperimentResultRecord:
    experiment_id: str
    protocol_digest: str
    code_digest: str
    dataset_digests: tuple[str, ...]
    metrics: tuple[ExperimentMetric, ...]
    limitations: tuple[str, ...] = ()
    def __post_init__(self):
        for value in (self.protocol_digest, self.code_digest, *self.dataset_digests):
            if len(value) != 64: raise ValueError("experiment references must be SHA-256")
        if not self.experiment_id.strip() or not self.metrics: raise ValueError("experiment ID and metrics required")
    @property
    def digest(self):
        body = (self.experiment_id, self.protocol_digest, self.code_digest, self.dataset_digests,
                tuple((m.name, m.value, m.unit) for m in self.metrics), self.limitations)
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

class ExperimentResultStore:
    def __init__(self): self._items = {}
    def add(self, item: ExperimentResultRecord):
        if not isinstance(item, ExperimentResultRecord): raise TypeError("item must be ExperimentResultRecord")
        old = self._items.get(item.experiment_id)
        if old is not None and old != item: raise ValueError("experiment result is immutable")
        self._items[item.experiment_id] = item
        return item
    def get(self, experiment_id): return self._items[experiment_id]

__all__ = ["ExperimentMetric", "ExperimentResultRecord", "ExperimentResultStore"]
