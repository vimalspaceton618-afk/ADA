"""Training-only dataset views with license and provenance requirements."""
from __future__ import annotations
from dataclasses import dataclass
from .. import DatasetManifest, DatasetSplit

@dataclass(frozen=True, slots=True)
class TrainingCorpus:
    manifest: DatasetManifest
    records: tuple
    def __post_init__(self):
        if not isinstance(self.manifest, DatasetManifest) or self.manifest.split is not DatasetSplit.TRAIN:
            raise ValueError("training corpus requires TRAIN split")
        if not self.manifest.license_id or self.manifest.license_id.lower() in {"unknown", "none", "unlicensed"}:
            raise PermissionError("training corpus requires declared usage license")
        if len(self.records) != self.manifest.sample_count: raise ValueError("sample_count mismatch")

__all__ = ["TrainingCorpus"]
