"""Managed node registry with explicit enable/disable and heartbeat operations."""
from __future__ import annotations
from .. import NodeDescriptor, NodeRegistry

class ManagedNodeRegistry:
    def __init__(self, registry: NodeRegistry | None = None):
        self.registry = registry or NodeRegistry()
    def register(self, node: NodeDescriptor): self.registry.register(node); return node
    def heartbeat(self, node_id: str, *, now: float):
        return self.registry.heartbeat(node_id, now=now)
    def set_enabled(self, node_id: str, enabled: bool):
        return self.registry.set_enabled(node_id, enabled)
    def available(self, capability: str, *, now: float):
        return self.registry.available(capability=capability, now=now)

__all__ = ["ManagedNodeRegistry"]
