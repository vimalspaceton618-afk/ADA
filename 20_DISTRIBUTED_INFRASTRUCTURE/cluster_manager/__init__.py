"""Read-only cluster capacity snapshots derived from live node heartbeats."""
from __future__ import annotations
from dataclasses import dataclass
from .. import NodeRegistry

@dataclass(frozen=True, slots=True)
class ClusterSnapshot:
    timestamp: float
    live_nodes: int
    total_slots: int
    capabilities: tuple[str, ...]

def snapshot(registry: NodeRegistry, *, now: float):
    if not isinstance(registry, NodeRegistry): raise TypeError("registry must be NodeRegistry")
    live = tuple(n for n in registry.nodes()
                 if n.enabled and 0 <= now - n.last_heartbeat <= registry.heartbeat_timeout)
    return ClusterSnapshot(now, len(live), sum(n.total_slots for n in live),
                           tuple(sorted({c for n in live for c in n.capabilities})))

__all__ = ["ClusterSnapshot", "snapshot"]
