"""Per-request inference latency, token throughput and failure counters."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True, slots=True)
class InferenceObservation:
    request_id: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    succeeded: bool
    @property
    def tokens_per_second(self):
        return self.output_tokens / self.latency_seconds if self.latency_seconds > 0 else 0.0

class InferenceMonitor:
    def __init__(self): self._items = []
    def record(self, observation: InferenceObservation):
        if not isinstance(observation, InferenceObservation): raise TypeError("observation must be InferenceObservation")
        if not math.isfinite(observation.latency_seconds) or observation.latency_seconds < 0:
            raise ValueError("latency must be finite and non-negative")
        self._items.append(observation); return observation
    @property
    def summary(self):
        if not self._items: return {"requests": 0, "failures": 0, "tokens_per_second": 0.0}
        total = sum(x.latency_seconds for x in self._items)
        tokens = sum(x.output_tokens for x in self._items)
        return {"requests": len(self._items), "failures": sum(not x.succeeded for x in self._items),
                "tokens_per_second": tokens / total if total else 0.0}

__all__ = ["InferenceObservation", "InferenceMonitor"]
