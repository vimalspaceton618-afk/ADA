"""Capability and priority-aware workload assignment over node leases."""
from __future__ import annotations
from dataclasses import dataclass
from .. import WorkloadScheduler, WorkLease

@dataclass(frozen=True, slots=True)
class WorkRequest:
    task_id: str
    capability: str
    priority: int = 0
    estimated_slots: int = 1
    def __post_init__(self):
        if not self.task_id.strip() or not self.capability.strip(): raise ValueError("task and capability required")
        if type(self.priority) is not int or type(self.estimated_slots) is not int or self.estimated_slots < 1:
            raise ValueError("invalid priority or slot estimate")

class WorkloadDistributor:
    def __init__(self, scheduler: WorkloadScheduler):
        if not isinstance(scheduler, WorkloadScheduler): raise TypeError("scheduler must be WorkloadScheduler")
        self.scheduler = scheduler
    def assign(self, requests, *, now: float, ttl_seconds: float = 60.0):
        ordered = sorted(tuple(requests), key=lambda x: (-x.priority, x.task_id))
        return tuple(self.scheduler.lease(x.task_id, x.capability, ttl_seconds=ttl_seconds, now=now)
                     for x in ordered)

__all__ = ["WorkRequest", "WorkloadDistributor"]
