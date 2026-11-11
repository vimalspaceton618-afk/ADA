"""Context-local trace/span lifecycle helpers."""
from __future__ import annotations
import contextvars, time, uuid
from contextlib import contextmanager
from .. import ObservabilityStore

_current_trace = contextvars.ContextVar("ada_trace_id", default=None)
_current_span = contextvars.ContextVar("ada_span_id", default=None)

def current_trace():
    return _current_trace.get(), _current_span.get()

@contextmanager
def span(store: ObservabilityStore, name: str):
    if not isinstance(store, ObservabilityStore): raise TypeError("store must be ObservabilityStore")
    trace_id = _current_trace.get() or str(uuid.uuid4())
    parent = _current_span.get()
    item = store.start_span(name, trace_id=trace_id, parent_span_id=parent)
    trace_token = _current_trace.set(trace_id)
    span_token = _current_span.set(item.span_id)
    status = "ok"
    try:
        yield item
    except Exception:
        status = "error"
        raise
    finally:
        store.end_span(item.span_id, status=status)
        _current_span.reset(span_token)
        _current_trace.reset(trace_token)

__all__ = ["current_trace", "span"]
