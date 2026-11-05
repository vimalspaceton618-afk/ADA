"""Dependency-free runtime health and lifecycle metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .contracts import TaskSnapshot, TaskStatus


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

    @property
    def queue_at_capacity(self) -> bool:
        return self.queue_depth >= self.queue_capacity


class RuntimeMonitor:
    """Build a point-in-time health summary from runtime-owned snapshots."""

    @staticmethod
    def snapshot(
        tasks: Sequence[TaskSnapshot],
        *,
        queue_depth: int,
        queue_capacity: int,
        retained_events: int,
        event_delivery_failures: int,
    ) -> RuntimeHealthSnapshot:
        if queue_capacity < 1 or not 0 <= queue_depth <= queue_capacity:
            raise ValueError("queue depth/capacity values are inconsistent")
        counts = Counter(task.status.value for task in tasks)
        for status in TaskStatus:
            counts.setdefault(status.value, 0)
        durations: list[float] = []
        for task in tasks:
            if task.started_at is not None and task.finished_at is not None:
                duration = (task.finished_at - task.started_at).total_seconds()
                if duration >= 0:
                    durations.append(duration)
        average = sum(durations) / len(durations) if durations else None
        immutable_counts = MappingProxyType(dict(counts))
        terminal = sum(
            counts[status.value]
            for status in TaskStatus
            if status.is_terminal
        )
        return RuntimeHealthSnapshot(
            queue_depth=queue_depth,
            queue_capacity=queue_capacity,
            tasks_total=len(tasks),
            status_counts=immutable_counts,
            terminal_tasks=terminal,
            average_execution_seconds=average,
            retained_events=retained_events,
            event_delivery_failures=event_delivery_failures,
        )
