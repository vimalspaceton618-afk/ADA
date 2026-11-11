"""Cloud provider protocol and dry-run adapter; no cloud API is called."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Mapping, Any

@dataclass(frozen=True, slots=True)
class CloudTarget:
    provider: str
    region: str
    resource_id: str
    def __post_init__(self):
        if any(not x.strip() for x in (self.provider, self.region, self.resource_id)):
            raise ValueError("cloud target fields required")

class CloudProvider(Protocol):
    def deploy(self, artifact_digest: str, target: CloudTarget, config: Mapping[str, Any]): ...
    def rollback(self, target: CloudTarget, artifact_digest: str): ...

class DryRunCloudProvider:
    def deploy(self, artifact_digest: str, target: CloudTarget, config: Mapping[str, Any]):
        if len(artifact_digest) != 64: raise ValueError("artifact digest must be SHA-256")
        if not isinstance(target, CloudTarget) or not isinstance(config, Mapping): raise TypeError("invalid deploy inputs")
        return {"status": "dry_run", "provider": target.provider, "resource_id": target.resource_id, "artifact_digest": artifact_digest}
    def rollback(self, target: CloudTarget, artifact_digest: str):
        if len(artifact_digest) != 64: raise ValueError("artifact digest must be SHA-256")
        return {"status": "dry_run_rollback", "resource_id": target.resource_id, "artifact_digest": artifact_digest}

__all__ = ["CloudTarget", "CloudProvider", "DryRunCloudProvider"]
