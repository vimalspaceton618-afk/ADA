"""Dataset manifests, deterministic preparation and quality gates."""
from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("digest must be lowercase SHA-256")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    EVALUATION = "evaluation"
    UNASSIGNED = "unassigned"


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    version: str
    content_digest: str
    split: DatasetSplit
    sample_count: int
    license_id: str
    source_id: str
    schema_version: str = "1.0.0"

    def __post_init__(self):
        for name in ("dataset_id", "version", "license_id", "source_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        _digest(self.content_digest)
        if type(self.sample_count) is not int or self.sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        if not isinstance(self.split, DatasetSplit):
            raise TypeError("split must be DatasetSplit")


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    input_count: int
    accepted_count: int
    duplicate_count: int
    rejected_count: int
    missing_fields: tuple[str, ...]
    content_digest: str

    @property
    def accepted(self) -> bool:
        return self.input_count == self.accepted_count + self.duplicate_count + self.rejected_count


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    records: tuple[Mapping[str, Any], ...]
    manifest: DatasetManifest
    quality: DataQualityReport


class DatasetRegistry:
    def __init__(self):
        self._items = {}
        self._lock = threading.RLock()

    def register(self, manifest: DatasetManifest):
        if not isinstance(manifest, DatasetManifest):
            raise TypeError("manifest must be DatasetManifest")
        key = (manifest.dataset_id, manifest.version)
        with self._lock:
            prior = self._items.get(key)
            if prior is not None and prior != manifest:
                raise ValueError("dataset version is immutable and already registered")
            self._items[key] = manifest
        return manifest

    def get(self, dataset_id: str, version: str) -> DatasetManifest:
        with self._lock:
            return self._items[(dataset_id, version)]

    @property
    def manifests(self):
        with self._lock:
            return tuple(self._items[k] for k in sorted(self._items))


class DataPipeline:
    """Canonicalizes JSON records and reports deduplication and schema rejects."""

    def prepare(self, records: Iterable[Mapping[str, Any]], *, dataset_id: str,
                version: str, split: DatasetSplit, license_id: str,
                source_id: str, required_fields: frozenset[str] = frozenset()):
        if not license_id.strip() or not source_id.strip():
            raise ValueError("license_id and source_id are mandatory")
        rows = tuple(records)
        accepted = []
        seen = set()
        duplicates = 0
        rejected = 0
        missing = set()
        for row in rows:
            if not isinstance(row, Mapping):
                rejected += 1
                continue
            absent = required_fields - set(row)
            if absent:
                missing.update(absent)
                rejected += 1
                continue
            try:
                raw = _canonical(dict(row))
            except (TypeError, ValueError):
                rejected += 1
                continue
            key = hashlib.sha256(raw).hexdigest()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            accepted.append(json.loads(raw.decode("utf-8")))
        payload = _canonical(accepted)
        content_digest = hashlib.sha256(payload).hexdigest()
        manifest = DatasetManifest(
            dataset_id, version, content_digest, split, len(accepted),
            license_id, source_id,
        )
        quality = DataQualityReport(
            len(rows), len(accepted), duplicates, rejected,
            tuple(sorted(missing)), content_digest,
        )
        return PreparedDataset(tuple(accepted), manifest, quality)



@dataclass(frozen=True, slots=True)
class DataQualityPolicy:
    minimum_accepted: int = 1
    maximum_rejection_rate: float = 0.0
    maximum_duplicate_rate: float = 1.0

    def __post_init__(self):
        if type(self.minimum_accepted) is not int or self.minimum_accepted < 1:
            raise ValueError("minimum_accepted must be positive")
        for value in (self.maximum_rejection_rate, self.maximum_duplicate_rate):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= value <= 1:
                raise ValueError("quality rates must be finite values in [0, 1]")


class DataQualityGate:
    def __init__(self, policy: DataQualityPolicy):
        if not isinstance(policy, DataQualityPolicy):
            raise TypeError("policy must be DataQualityPolicy")
        self.policy = policy

    def evaluate(self, report: DataQualityReport) -> bool:
        if not isinstance(report, DataQualityReport):
            raise TypeError("report must be DataQualityReport")
        if report.input_count <= 0 or report.accepted_count < self.policy.minimum_accepted:
            return False
        rejection = report.rejected_count / report.input_count
        duplicate = report.duplicate_count / report.input_count
        return (
            report.accepted
            and rejection <= self.policy.maximum_rejection_rate
            and duplicate <= self.policy.maximum_duplicate_rate
        )



__all__ = [
    "DatasetSplit", "DatasetManifest", "DataQualityReport",
    "PreparedDataset", "DatasetRegistry", "DataPipeline",
    "DataQualityPolicy", "DataQualityGate",
]
