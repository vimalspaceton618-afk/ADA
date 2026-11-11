"""Reproducible training run lifecycle and immutable checkpoint registry.

Actual tensor optimization is injected by a framework-specific trainer.
"""
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


def _config_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_id: str
    dataset_digest: str
    seed: int
    max_steps: int
    learning_rate: float
    batch_size: int
    precision: str = "fp32"
    optimizer: str = "adamw"

    def __post_init__(self):
        if not self.model_id.strip():
            raise ValueError("model_id is required")
        _sha(self.dataset_digest)
        if type(self.seed) is not int or type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("seed and positive max_steps must be integers")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be positive integer")
        if isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("unsupported precision")


@dataclass(frozen=True, slots=True)
class TrainingRun:
    run_id: str
    config: TrainingConfig
    status: RunStatus = RunStatus.CREATED
    step: int = 0
    latest_metrics: Mapping[str, float] = field(default_factory=dict)
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    checkpoint_id: str
    model_id: str
    artifact_digest: str
    config_digest: str
    dataset_digest: str
    step: int
    parent_digest: str | None = None

    def __post_init__(self):
        for d in (self.artifact_digest, self.config_digest, self.dataset_digest):
            _sha(d)
        if type(self.step) is not int or self.step < 0:
            raise ValueError("checkpoint step must be non-negative")


class CheckpointRegistry:
    def __init__(self):
        self._items = {}
        self._lock = threading.RLock()

    def register(self, item: CheckpointManifest):
        if not isinstance(item, CheckpointManifest):
            raise TypeError("item must be CheckpointManifest")
        with self._lock:
            prior = self._items.get(item.artifact_digest)
            if prior is not None and prior != item:
                raise ValueError("checkpoint digest collision")
            self._items[item.artifact_digest] = item
        return item

    def get(self, artifact_digest: str):
        with self._lock:
            return self._items[artifact_digest]


class TrainingCoordinator:
    """Enforces monotonic steps and explicit run-state transitions."""

    _allowed = {
        RunStatus.CREATED: {RunStatus.RUNNING, RunStatus.FAILED},
        RunStatus.RUNNING: {RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.FAILED},
        RunStatus.PAUSED: {RunStatus.RUNNING, RunStatus.FAILED},
        RunStatus.COMPLETED: set(),
        RunStatus.FAILED: set(),
    }

    def __init__(self):
        self._runs = {}
        self._lock = threading.RLock()

    def create(self, config: TrainingConfig) -> TrainingRun:
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be TrainingConfig")
        run = TrainingRun(str(uuid.uuid4()), config)
        with self._lock:
            self._runs[run.run_id] = run
        return run

    def transition(self, run_id: str, status: RunStatus, *, reason: str | None = None):
        with self._lock:
            current = self._runs[run_id]
            if status not in self._allowed[current.status]:
                raise ValueError(f"invalid training transition {current.status.value}->{status.value}")
            updated = TrainingRun(current.run_id, current.config, status,
                                  current.step, current.latest_metrics, reason)
            self._runs[run_id] = updated
            return updated

    def record_step(self, run_id: str, step: int, metrics: Mapping[str, float]):
        with self._lock:
            current = self._runs[run_id]
            if current.status is not RunStatus.RUNNING:
                raise ValueError("training run must be running")
            if type(step) is not int or step != current.step + 1 or step > current.config.max_steps:
                raise ValueError("step must advance exactly once and stay within max_steps")
            clean = {}
            for name, value in metrics.items():
                if not name.strip() or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError("metrics must have names and finite numeric values")
                clean[name] = float(value)
            updated = TrainingRun(current.run_id, current.config, current.status,
                                  step, clean)
            self._runs[run_id] = updated
            return updated

    def get(self, run_id: str):
        with self._lock:
            return self._runs[run_id]


__all__ = [
    "RunStatus", "TrainingConfig", "TrainingRun", "CheckpointManifest",
    "CheckpointRegistry", "TrainingCoordinator",
]
