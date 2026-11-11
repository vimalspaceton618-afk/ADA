"""Adapter contract to export process-local observability batches."""
from __future__ import annotations
from typing import Protocol
from .. import ObservabilityStore

class TelemetryExporter(Protocol):
    def export(self, events, metrics, spans) -> None: ...

class TelemetryFlush:
    def __init__(self, store: ObservabilityStore, exporter: TelemetryExporter):
        if not isinstance(store, ObservabilityStore) or not callable(getattr(exporter, "export", None)):
            raise TypeError("store and exporter are required")
        self.store, self.exporter = store, exporter
    def flush(self):
        events = self.store.events()
        metrics = self.store.metric_samples
        spans = self.store.spans
        self.exporter.export(events, metrics, spans)
        return {"events": len(events), "metrics": len(metrics), "spans": len(spans)}

__all__ = ["TelemetryExporter", "TelemetryFlush"]
