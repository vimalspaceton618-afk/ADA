"""Resource-aware node registry and deterministic lease scheduler.

This is a control-plane reference. It does not open network connections or
claim distributed consensus or fault tolerance.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class NodeDescriptor:
    node_id: str
    capabilities: frozenset[str]
    total_slots: int
    last_heartbeat: float = field(default_factory=time.time)
    enabled: bool = True

    def __post_init__(self):
        if not self.node_id.strip() or not self.capabilities:
            raise ValueError("node_id and capabilities are required")
        if type(self.total_slots) is not int or self.total_slots < 1:
            raise ValueError("total_slots must be positive")


@dataclass(frozen=True, slots=True)
class WorkLease:
    lease_id: str
    task_id: str
    node_id: str
    capability: str
    expires_at: float


class NodeRegistry:
    def __init__(self, heartbeat_timeout: float = 30.0):
        if heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be positive")
        self.heartbeat_timeout = heartbeat_timeout
        self._nodes = {}
        self._lock = threading.RLock()

    def register(self, node: NodeDescriptor):
        if not isinstance(node, NodeDescriptor):
            raise TypeError("node must be NodeDescriptor")
        with self._lock:
            if node.node_id in self._nodes:
                raise ValueError("node already registered")
            self._nodes[node.node_id] = node

    def heartbeat(self, node_id: str, *, now: float | None = None):
        import dataclasses
        with self._lock:
            current = self._nodes[node_id]
            updated = dataclasses.replace(current, last_heartbeat=time.time() if now is None else float(now))
            self._nodes[node_id] = updated
            return updated

    def available(self, *, capability: str, now: float | None = None):
        current = time.time() if now is None else float(now)
        with self._lock:
            return tuple(sorted(
                (n for n in self._nodes.values()
                 if n.enabled and capability in n.capabilities
                 and 0 <= current - n.last_heartbeat <= self.heartbeat_timeout),
                key=lambda n: n.node_id,
            ))


class WorkloadScheduler:
    def __init__(self, registry: NodeRegistry):
        if not isinstance(registry, NodeRegistry):
            raise TypeError("registry must be NodeRegistry")
        self.registry = registry
        self._leases = {}
        self._task_index = {}
        self._lock = threading.RLock()

    def lease(self, task_id: str, capability: str, *, ttl_seconds: float = 60.0,
              now: float | None = None):
        if not task_id.strip() or not capability.strip() or ttl_seconds <= 0:
            raise ValueError("task, capability and positive TTL are required")
        current = time.time() if now is None else float(now)
        with self._lock:
            prior_id = self._task_index.get(task_id)
            if prior_id is not None:
                prior = self._leases.get(prior_id)
                if prior is not None and prior.expires_at > current:
                    if prior.capability != capability:
                        raise ValueError("task already leased for a different capability")
                    return prior
            candidates = self.registry.available(capability=capability, now=current)
            usage = {node.node_id: sum(1 for lease in self._leases.values()
                      if lease.node_id == node.node_id and lease.expires_at > current)
                     for node in candidates}
            candidates = [n for n in candidates if usage[n.node_id] < n.total_slots]
            if not candidates:
                raise RuntimeError("no live node has capacity for requested capability")
            node = min(candidates, key=lambda n: (usage[n.node_id] / n.total_slots, n.node_id))
            lease = WorkLease(str(uuid.uuid4()), task_id, node.node_id, capability, current + ttl_seconds)
            self._leases[lease.lease_id] = lease
            self._task_index[task_id] = lease.lease_id
            return lease

    def release(self, lease_id: str):
        with self._lock:
            lease = self._leases.pop(lease_id)
            self._task_index.pop(lease.task_id, None)
            return lease


__all__ = ["NodeDescriptor", "WorkLease", "NodeRegistry", "WorkloadScheduler"]
