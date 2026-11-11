"""Release manifest validation and guarded deployment state transitions.

Provider-specific operations are injected adapters; this module never deploys
containers, uploads credentials or mutates infrastructure by itself.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("digest must be lowercase SHA-256")


class DeploymentStatus(str, Enum):
    STAGED = "staged"
    CANARY = "canary"
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ReleaseArtifact:
    artifact_id: str
    model_id: str
    version: str
    artifact_digest: str
    evaluation_report_digest: str
    safety_report_digest: str

    def __post_init__(self):
        for name in ("artifact_id", "model_id", "version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        for digest in (self.artifact_digest, self.evaluation_report_digest, self.safety_report_digest):
            _sha(digest)


@dataclass(frozen=True, slots=True)
class DeploymentRecord:
    deployment_id: str
    artifact: ReleaseArtifact
    environment: str
    status: DeploymentStatus
    approver_id: str | None = None
    prior_artifact_digest: str | None = None
    limitations: tuple[str, ...] = ()


class DeploymentManager:
    """Requires explicit evidence and an approver before staging a release."""

    _allowed = {
        DeploymentStatus.STAGED: {DeploymentStatus.CANARY, DeploymentStatus.REJECTED},
        DeploymentStatus.CANARY: {DeploymentStatus.ACTIVE, DeploymentStatus.ROLLED_BACK},
        DeploymentStatus.ACTIVE: {DeploymentStatus.ROLLED_BACK},
        DeploymentStatus.ROLLED_BACK: set(),
        DeploymentStatus.REJECTED: set(),
    }

    def __init__(self):
        self._records = {}
        self._active = {}
        self._lock = threading.RLock()

    def stage(self, artifact: ReleaseArtifact, *, environment: str,
              gate_passed: bool, approver_id: str):
        if not isinstance(artifact, ReleaseArtifact):
            raise TypeError("artifact must be ReleaseArtifact")
        if not environment.strip() or not approver_id.strip():
            raise ValueError("environment and approver are required")
        if gate_passed is not True:
            raise PermissionError("release gate must explicitly pass")
        with self._lock:
            prior = self._active.get((artifact.model_id, environment))
            record = DeploymentRecord(
                str(uuid.uuid4()), artifact, environment, DeploymentStatus.STAGED,
                approver_id, prior,
            )
            self._records[record.deployment_id] = record
            return record

    def transition(self, deployment_id: str, status: DeploymentStatus):
        with self._lock:
            current = self._records[deployment_id]
            if status not in self._allowed[current.status]:
                raise ValueError("invalid deployment state transition")
            updated = DeploymentRecord(
                current.deployment_id, current.artifact, current.environment, status,
                current.approver_id, current.prior_artifact_digest, current.limitations,
            )
            self._records[deployment_id] = updated
            key = (current.artifact.model_id, current.environment)
            if status is DeploymentStatus.ACTIVE:
                self._active[key] = current.artifact.artifact_digest
            elif status is DeploymentStatus.ROLLED_BACK:
                if current.prior_artifact_digest is None:
                    self._active.pop(key, None)
                else:
                    self._active[key] = current.prior_artifact_digest
            return updated

    def active_digest(self, model_id: str, environment: str):
        with self._lock:
            return self._active.get((model_id, environment))

    def get(self, deployment_id: str):
        with self._lock:
            return self._records[deployment_id]


__all__ = ["DeploymentStatus", "ReleaseArtifact", "DeploymentRecord", "DeploymentManager"]
