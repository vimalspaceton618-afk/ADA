"""Validation corpus views and split leakage checks."""
from __future__ import annotations
from dataclasses import dataclass
from .. import DatasetManifest, DatasetSplit

@dataclass(frozen=True, slots=True)
class ValidationCorpus:
    manifest: DatasetManifest
    records: tuple
    sample_digests: frozenset[str]
    def __post_init__(self):
        if not isinstance(self.manifest, DatasetManifest) or self.manifest.split is not DatasetSplit.VALIDATION:
            raise ValueError("validation corpus requires VALIDATION split")
        if len(self.records) != self.manifest.sample_count: raise ValueError("sample_count mismatch")
        if any(len(x) != 64 for x in self.sample_digests): raise ValueError("sample digests must be SHA-256")
def check_split_leakage(train_digests, validation_digests):
    return frozenset(train_digests) & frozenset(validation_digests)

__all__ = ["ValidationCorpus", "check_split_leakage"]
