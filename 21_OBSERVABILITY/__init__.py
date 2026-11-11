"""Bounded structured telemetry, metrics and trace primitives."""
from __future__ import annotations

import math
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EventLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    name: str
    level: EventLevel
    timestamp: float
    trace_id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if not self.name.strip() or not self.trace_id.strip():
            raise ValueError("event name and trace_id are required")
        if not isinstance(self.level, EventLevel):
            raise TypeError("level must be EventLevel")
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        if any(not isinstance(k, str) for k in self.attributes):
            raise TypeError("attribute keys must be strings")


@dataclass(frozen=True, slots=True)
class MetricSample:
    name: str
    value: float
    timestamp: float
    unit: str = "1"
    labels: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if not self.name.strip() or not self.unit.strip():
            raise ValueError("metric name and unit are required")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(float(self.value)):
            raise ValueError("metric value must be finite numeric")
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")


@dataclass(frozen=True, slots=True)
class Span:
    trace_id: str
    span_id: str
    name: str
    started_at: float
    ended_at: float | None = None
    parent_span_id: str | None = None
    status: str = "unset"

    def __post_init__(self):
        if not self.trace_id.strip() or not self.span_id.strip() or not self.name.strip():
            raise ValueError("trace, span and name are required")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("span end cannot precede start")


class ObservabilityStore:
    """Process-local ring buffer; export and durable retention are adapters."""

    def __init__(self, capacity: int = 50_000):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._events = []
        self._metrics = []
        self._spans = {}
        self._lock = threading.RLock()

    def record_event(self, event: TelemetryEvent):
        if not isinstance(event, TelemetryEvent):
            raise TypeError("event must be TelemetryEvent")
        with self._lock:
            self._events.append(event)
            self._events = self._events[-self.capacity:]
        return event

    def record_metric(self, sample: MetricSample):
        if not isinstance(sample, MetricSample):
            raise TypeError("sample must be MetricSample")
        with self._lock:
            self._metrics.append(sample)
            self._metrics = self._metrics[-self.capacity:]
        return sample

    def start_span(self, name: str, *, trace_id: str | None = None,
                   parent_span_id: str | None = None, now: float | None = None):
        stamp = time.time() if now is None else float(now)
        span = Span(trace_id or str(uuid.uuid4()), str(uuid.uuid4()), name,
                    stamp, parent_span_id=parent_span_id)
        with self._lock:
            self._spans[span.span_id] = span
        return span

    def end_span(self, span_id: str, *, status: str = "ok", now: float | None = None):
        from dataclasses import replace
        stamp = time.time() if now is None else float(now)
        with self._lock:
            current = self._spans[span_id]
            if current.ended_at is not None:
                raise ValueError("span is already closed")
            updated = replace(current, ended_at=stamp, status=status)
            self._spans[span_id] = updated
            return updated

    def events(self, *, trace_id: str | None = None):
        with self._lock:
            values = tuple(self._events)
        return tuple(x for x in values if trace_id is None or x.trace_id == trace_id)

    def metric_summary(self, name: str):
        with self._lock:
            values = [float(x.value) for x in self._metrics if x.name == name]
        if not values:
            return None
        return {"count": len(values), "min": min(values),
                "max": max(values), "mean": sum(values) / len(values)}

    @property
    def spans(self):
        with self._lock:
            return tuple(self._spans.values())


__all__ = ["EventLevel", "TelemetryEvent", "MetricSample", "Span", "ObservabilityStore"]
