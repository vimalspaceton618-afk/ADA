"""Descriptive aggregate action frequencies; never infer hidden human traits."""
from __future__ import annotations
from collections import Counter
from collections.abc import Iterable
from .contracts import ObservedBehaviorEvent


class EmpiricalActionPrior:
    """Count observed action labels by context; no individual prediction API."""
    def __init__(self, events: Iterable[ObservedBehaviorEvent] = (), *, smoothing: float = 0.5):
        if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)) or smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        self.smoothing = float(smoothing)
        self._counts: dict[str, Counter[str]] = {}
        self._observed_events: set[tuple[str, str, str, str, str]] = set()
        self.update(events)

    def update(self, events: Iterable[ObservedBehaviorEvent]) -> None:
        for event in events:
            if not isinstance(event, ObservedBehaviorEvent):
                raise TypeError("events must contain ObservedBehaviorEvent values")
            event_key = (event.actor_ref, event.action_label, event.context_label,
                         event.observed_at.isoformat(), event.source_id + ":" + event.source_digest)
            if event_key in self._observed_events:
                continue
            self._counts.setdefault(event.context_label, Counter())[event.action_label] += 1
            self._observed_events.add(event_key)

    def distribution(self, context_label: str) -> dict[str, float]:
        if not isinstance(context_label, str) or not context_label.strip():
            raise ValueError("context_label must be non-empty")
        counts = self._counts.get(context_label, Counter())
        if not counts:
            return {}
        labels = sorted(counts)
        denominator = sum(counts.values()) + self.smoothing * len(labels)
        return {
            label: (counts[label] + self.smoothing) / denominator
            for label in labels
        }

    def counts(self, context_label: str) -> dict[str, int]:
        return dict(self._counts.get(context_label, Counter()))
