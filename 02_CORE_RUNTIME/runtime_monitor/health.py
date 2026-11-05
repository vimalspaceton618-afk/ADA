"""Dependency-free health snapshots derived from runtime state."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ..contracts import TaskSnapshot, TaskStatus


@dataclass(frozen=True, slots=True)
class RuntimeHealthSnapshot:
    queue_depth: int
    queue_capacity: int
    tasks_total: int
    status_counts: Mapping[str, int]
    terminal_tasks: int
    average_execution_seconds: float | None
    retained_events: int
    event_delivery_failures: int
    active_processes: int = 0

    @property
    def queue_at_capacity(self) -> bool:
        return self.queue_depth >= self.queue_capacity


class RuntimeMonitor:
    @staticmethod
    def snapshot(
        tasks: Sequence[TaskSnapshot], *, queue_depth: int, queue_capacity: int,
        retained_events: int, event_delivery_failures: int, active_processes: int = 0,
    ) -> RuntimeHealthSnapshot:
        if queue_capacity < 1 or not 0 <= queue_depth <= queue_capacity:
            raise ValueError("queue depth/capacity are inconsistent")
        counts = Counter(item.status.value for item in tasks)
        for status in TaskStatus:
            counts.setdefault(status.value, 0)
        durations = [
            (item.finished_at - item.started_at).total_seconds()
            for item in tasks
            if item.started_at is not None and item.finished_at is not None
            and item.finished_at >= item.started_at
        ]
        average = sum(durations) / len(durations) if durations else None
        terminal = sum(counts[item.value] for item in TaskStatus if item.is_terminal)
        return RuntimeHealthSnapshot(
            queue_depth, queue_capacity, len(tasks), MappingProxyType(dict(counts)),
            terminal, average, retained_events, event_delivery_failures, active_processes,
        )
