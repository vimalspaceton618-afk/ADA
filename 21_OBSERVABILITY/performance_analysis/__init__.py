"""Latency and throughput summaries from observed requests."""
from __future__ import annotations
import math

def percentile(values, p: float):
    items = sorted(float(x) for x in values)
    if not items: return None
    if not 0 <= p <= 100: raise ValueError("percentile must be [0,100]")
    pos = (len(items) - 1) * p / 100
    low, high = math.floor(pos), math.ceil(pos)
    if low == high: return items[low]
    return items[low] * (high - pos) + items[high] * (pos - low)

def latency_report(samples):
    values = tuple(float(x) for x in samples)
    if any(not math.isfinite(x) or x < 0 for x in values): raise ValueError("latencies must be finite and non-negative")
    if not values: return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {"count": len(values), "mean": sum(values) / len(values),
            "p50": percentile(values, 50), "p95": percentile(values, 95), "p99": percentile(values, 99)}

__all__ = ["percentile", "latency_report"]
