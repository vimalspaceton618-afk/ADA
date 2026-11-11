"""Cluster deployment capacity validation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True, slots=True)
class ClusterDeploymentPlan:
    deployment_id: str
    artifact_digest: str
    replicas: int
    capability: str
    resource_slots_per_replica: int = 1
    def __post_init__(self):
        if not self.deployment_id.strip() or not self.capability.strip(): raise ValueError("deployment ID and capability required")
        if len(self.artifact_digest) != 64: raise ValueError("artifact digest must be SHA-256")
        if type(self.replicas) is not int or self.replicas < 1: raise ValueError("replicas must be positive")
        if type(self.resource_slots_per_replica) is not int or self.resource_slots_per_replica < 1:
            raise ValueError("slots per replica must be positive")

def validate_cluster_plan(plan: ClusterDeploymentPlan, available_slots: Mapping[str, int]):
    if not isinstance(plan, ClusterDeploymentPlan): raise TypeError("plan must be ClusterDeploymentPlan")
    slots = sum(max(0, int(value)) for value in available_slots.values())
    required = plan.replicas * plan.resource_slots_per_replica
    return {"accepted": slots >= required, "required_slots": required, "available_slots": slots}

__all__ = ["ClusterDeploymentPlan", "validate_cluster_plan"]
