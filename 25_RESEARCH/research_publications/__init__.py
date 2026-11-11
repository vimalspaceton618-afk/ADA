"""Research artifact release checklist and reproducibility metadata."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PublicationArtifact:
    artifact_id: str
    title: str
    code_commit: str
    license_id: str
    data_statement: str
    limitations: tuple[str, ...]
    evaluation_report_ids: tuple[str, ...] = ()
    def __post_init__(self):
        if any(not x.strip() for x in (self.artifact_id, self.title, self.code_commit, self.license_id, self.data_statement)):
            raise ValueError("publication metadata is incomplete")
        if not self.limitations: raise ValueError("limitations must be disclosed")

def release_ready(item: PublicationArtifact):
    if not isinstance(item, PublicationArtifact): raise TypeError("PublicationArtifact required")
    return bool(item.evaluation_report_ids)

__all__ = ["PublicationArtifact", "release_ready"]
