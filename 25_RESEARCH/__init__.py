"""Versioned research hypotheses, protocols and experiment evidence."""
from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA-256 digest")


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class ResearchHypothesis:
    hypothesis_id: str
    statement: str
    null_hypothesis: str
    success_metric: str
    acceptance_threshold: float
    status: HypothesisStatus = HypothesisStatus.PROPOSED

    def __post_init__(self):
        for name in ("hypothesis_id", "statement", "null_hypothesis", "success_metric"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        if not math.isfinite(float(self.acceptance_threshold)):
            raise ValueError("acceptance_threshold must be finite")


@dataclass(frozen=True, slots=True)
class ExperimentProtocol:
    experiment_id: str
    hypothesis_id: str
    seed: int
    config_digest: str
    dataset_digests: tuple[str, ...]
    metric_direction: str = "higher"

    def __post_init__(self):
        if not self.experiment_id.strip() or not self.hypothesis_id.strip():
            raise ValueError("experiment and hypothesis IDs are required")
        if type(self.seed) is not int or not self.dataset_digests:
            raise ValueError("seed and dataset digests are required")
        _sha(self.config_digest)
        for value in self.dataset_digests:
            _sha(value)
        if self.metric_direction not in {"higher", "lower"}:
            raise ValueError("metric_direction must be higher or lower")


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    experiment_id: str
    protocol_digest: str
    code_digest: str
    sample_count: int
    metric_value: float
    conclusion: HypothesisStatus
    limitations: tuple[str, ...] = ()

    def __post_init__(self):
        _sha(self.protocol_digest)
        _sha(self.code_digest)
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if isinstance(self.metric_value, bool) or not math.isfinite(float(self.metric_value)):
            raise ValueError("metric_value must be finite")
        if self.conclusion is HypothesisStatus.PROPOSED:
            raise ValueError("an experiment result requires a terminal conclusion")


class ResearchLedger:
    """Append-only protocol/result ledger; hypotheses remain explicitly reviewed."""

    def __init__(self):
        self._hypotheses = {}
        self._protocols = {}
        self._results = {}
        self._lock = threading.RLock()

    def add_hypothesis(self, hypothesis: ResearchHypothesis):
        if not isinstance(hypothesis, ResearchHypothesis):
            raise TypeError("hypothesis must be ResearchHypothesis")
        with self._lock:
            if hypothesis.hypothesis_id in self._hypotheses:
                raise ValueError("hypothesis ID already exists")
            self._hypotheses[hypothesis.hypothesis_id] = hypothesis
        return hypothesis

    def add_protocol(self, protocol: ExperimentProtocol):
        if not isinstance(protocol, ExperimentProtocol):
            raise TypeError("protocol must be ExperimentProtocol")
        with self._lock:
            if protocol.hypothesis_id not in self._hypotheses:
                raise KeyError(protocol.hypothesis_id)
            if protocol.experiment_id in self._protocols:
                raise ValueError("experiment ID already exists")
            self._protocols[protocol.experiment_id] = protocol
        return protocol

    def add_result(self, result: ExperimentResult):
        if not isinstance(result, ExperimentResult):
            raise TypeError("result must be ExperimentResult")
        with self._lock:
            protocol = self._protocols[result.experiment_id]
            raw = json.dumps({
                "experiment_id": protocol.experiment_id,
                "hypothesis_id": protocol.hypothesis_id, "seed": protocol.seed,
                "config_digest": protocol.config_digest,
                "dataset_digests": protocol.dataset_digests,
                "metric_direction": protocol.metric_direction,
            }, sort_keys=True, separators=(",", ":")).encode()
            if hashlib.sha256(raw).hexdigest() != result.protocol_digest:
                raise ValueError("result protocol digest mismatch")
            if result.experiment_id in self._results:
                raise ValueError("experiment result is immutable")
            self._results[result.experiment_id] = result
        return result

    @staticmethod
    def protocol_digest(protocol: ExperimentProtocol):
        raw = json.dumps({
            "experiment_id": protocol.experiment_id,
            "hypothesis_id": protocol.hypothesis_id, "seed": protocol.seed,
            "config_digest": protocol.config_digest,
            "dataset_digests": protocol.dataset_digests,
            "metric_direction": protocol.metric_direction,
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    @property
    def results(self):
        with self._lock:
            return tuple(self._results[k] for k in sorted(self._results))


__all__ = [
    "HypothesisStatus", "ResearchHypothesis", "ExperimentProtocol",
    "ExperimentResult", "ResearchLedger",
]
