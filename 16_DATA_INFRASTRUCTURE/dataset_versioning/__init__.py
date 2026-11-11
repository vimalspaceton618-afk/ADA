"""Immutable dataset version graph with content-addressed parent links."""
from __future__ import annotations
from dataclasses import dataclass
from .. import DatasetManifest, DatasetRegistry

@dataclass(frozen=True, slots=True)
class DatasetVersion:
    manifest: DatasetManifest
    parent_version: str | None = None
    change_summary: str = ""
    def __post_init__(self):
        if not isinstance(self.manifest, DatasetManifest): raise TypeError("manifest must be DatasetManifest")
        if self.parent_version == self.manifest.version: raise ValueError("dataset version cannot parent itself")

class DatasetVersionGraph:
    def __init__(self):
        self.registry = DatasetRegistry()
        self._versions = {}
    def add(self, item: DatasetVersion):
        if not isinstance(item, DatasetVersion): raise TypeError("item must be DatasetVersion")
        key = (item.manifest.dataset_id, item.manifest.version)
        if item.parent_version is not None:
            parent = (item.manifest.dataset_id, item.parent_version)
            if parent not in self._versions: raise KeyError(f"unknown parent version {item.parent_version}")
        prior = self._versions.get(key)
        if prior is not None and prior != item: raise ValueError("dataset versions are immutable")
        self.registry.register(item.manifest)
        self._versions[key] = item
        return item
    def lineage(self, dataset_id: str, version: str):
        key = (dataset_id, version)
        if key not in self._versions: raise KeyError(key)
        output = []
        while key:
            item = self._versions[key]; output.append(item)
            key = (dataset_id, item.parent_version) if item.parent_version else None
        return tuple(output)

__all__ = ["DatasetVersion", "DatasetVersionGraph"]
