"""Thread-safe counters, gauges and bounded histogram summaries."""
from __future__ import annotations
import math, threading
from collections import defaultdict, deque

class MetricRegistry:
    def __init__(self, *, sample_capacity: int = 10_000):
        if sample_capacity < 1: raise ValueError("sample_capacity must be positive")
        self._capacity = sample_capacity
        self._counters = defaultdict(float)
        self._gauges = {}
        self._histograms = defaultdict(lambda: deque(maxlen=sample_capacity))
        self._lock = threading.RLock()
    def inc(self, name: str, value: float = 1.0):
        if not name.strip() or not math.isfinite(float(value)): raise ValueError("invalid counter")
        with self._lock: self._counters[name] += float(value)
        return self._counters[name]
    def gauge(self, name: str, value: float):
        if not name.strip() or not math.isfinite(float(value)): raise ValueError("invalid gauge")
        with self._lock: self._gauges[name] = float(value)
    def observe(self, name: str, value: float):
        if not name.strip() or not math.isfinite(float(value)): raise ValueError("invalid observation")
        with self._lock: self._histograms[name].append(float(value))
    def snapshot(self):
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: tuple(v) for k, v in self._histograms.items()},
            }

__all__ = ["MetricRegistry"]
