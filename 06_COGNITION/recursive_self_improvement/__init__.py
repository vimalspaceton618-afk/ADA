"""Bounded recursive self-improvement (RSI) for cognition strategies.

This module improves *versioned strategy artifacts*, never its own source code,
weights, safety policy, or execution permissions. A candidate is accepted only
when an independent evaluator reports a measurable improvement over a fixed
baseline on a held-out regression set. The evaluator and candidate generator
are injected by the caller; this module never executes generated code.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Mapping


SCHEMA_VERSION = "1.0.0"


def _digest(payload: Mapping) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StrategyArtifact:
    strategy_id: str
    version: int
    parameters: Mapping[str, object]
    parent_digest: str | None = None
    created_at: str = ""

    def __post_init__(self):
        if not self.strategy_id.strip():
            raise ValueError("strategy_id is required")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version must be a positive integer")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        # Ensure parameters are JSON-serializable, finite and detached from caller mutation.
        normalized = json.loads(json.dumps(dict(self.parameters), sort_keys=True, allow_nan=False))
        object.__setattr__(self, "parameters", normalized)
        if self.created_at and not isinstance(self.created_at, str):
            raise TypeError("created_at must be a string")

    @property
    def digest(self) -> str:
        return _digest({"strategy_id": self.strategy_id, "version": self.version,
                        "parameters": self.parameters, "parent_digest": self.parent_digest})


@dataclass(frozen=True, slots=True)
class Evaluation:
    score: float
    cases: int
    passed: int
    evaluator_id: str
    dataset_digest: str
    metrics: Mapping[str, float]

    def __post_init__(self):
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")
        if not math.isfinite(float(self.score)) or not 0 <= self.score <= 1:
            raise ValueError("score must be finite and in [0, 1]")
        if type(self.cases) is not int or self.cases < 1:
            raise ValueError("cases must be positive")
        if type(self.passed) is not int or not 0 <= self.passed <= self.cases:
            raise ValueError("passed must be within cases")
        if not self.evaluator_id.strip() or not self.dataset_digest.strip():
            raise ValueError("evaluator_id and dataset_digest are required")
        for key, value in self.metrics.items():
            if not isinstance(key, str) or not math.isfinite(float(value)):
                raise ValueError("metrics must contain finite numeric values")


@dataclass(frozen=True, slots=True)
class ImprovementRecord:
    iteration_id: str
    baseline_digest: str
    candidate_digest: str
    baseline_score: float
    candidate_score: float
    delta: float
    accepted: bool
    reason: str
    created_at: str


@dataclass(frozen=True, slots=True)
class RSIResult:
    active_strategy: StrategyArtifact
    proposed_strategy: StrategyArtifact | None
    baseline_evaluation: Evaluation
    candidate_evaluation: Evaluation | None
    record: ImprovementRecord
    history: tuple[ImprovementRecord, ...]


class BoundedRSI:
    """Propose/evaluate/promote strategy revisions with regression guards.

    Candidate generation returns a parameter mapping only. Evaluation is
    performed separately for both baseline and candidate on the same immutable
    held-out cases. Promotion is monotonic on the primary score and guarded by
    a maximum regression per named metric. Failed candidates are discarded.
    """

    def __init__(
        self,
        active_strategy: StrategyArtifact,
        *,
        evaluator: Callable[[StrategyArtifact, tuple], Evaluation],
        generator: Callable[[StrategyArtifact, tuple], Mapping],
        min_improvement: float = 0.01,
        max_metric_regression: float = 0.0,
        max_iterations: int = 3,
        allowed_parameter_keys: Iterable[str] | None = None,
    ):
        if not callable(evaluator) or not callable(generator):
            raise TypeError("evaluator and generator must be callable")
        for name, value in (("min_improvement", min_improvement),
                            ("max_metric_regression", max_metric_regression)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite numeric")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if type(max_iterations) is not int or not 1 <= max_iterations <= 20:
            raise ValueError("max_iterations must be between 1 and 20")
        self.active_strategy = active_strategy
        self.evaluator = evaluator
        self.generator = generator
        self.min_improvement = float(min_improvement)
        self.max_metric_regression = float(max_metric_regression)
        self.max_iterations = max_iterations
        self.allowed_parameter_keys = None if allowed_parameter_keys is None else frozenset(allowed_parameter_keys)
        self.history: list[ImprovementRecord] = []

    def _evaluate(self, strategy: StrategyArtifact, cases: tuple) -> Evaluation:
        result = self.evaluator(strategy, cases)
        if not isinstance(result, Evaluation):
            raise TypeError("evaluator must return Evaluation")
        return result

    def iterate(self, cases: Iterable, *, iterations: int = 1) -> tuple[RSIResult, ...]:
        cases = tuple(cases)
        if not cases:
            raise ValueError("held-out evaluation cases are required")
        if type(iterations) is not int or not 1 <= iterations <= self.max_iterations:
            raise ValueError("iterations must be between 1 and max_iterations")

        results = []
        for _ in range(iterations):
            baseline = self.active_strategy
            baseline_eval = self._evaluate(baseline, cases)
            proposed_parameters = self.generator(baseline, cases)
            if not isinstance(proposed_parameters, Mapping):
                raise TypeError("generator must return a parameter mapping")
            if self.allowed_parameter_keys is not None:
                unexpected = set(proposed_parameters) - self.allowed_parameter_keys
                if unexpected:
                    raise ValueError(f"generator proposed protected parameter keys: {sorted(unexpected)}")
            candidate = StrategyArtifact(
                strategy_id=baseline.strategy_id,
                version=baseline.version + 1,
                parameters=proposed_parameters,
                parent_digest=baseline.digest,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            candidate_eval = self._evaluate(candidate, cases)
            delta = candidate_eval.score - baseline_eval.score
            regressed = []
            for key, base_value in baseline_eval.metrics.items():
                if key in candidate_eval.metrics:
                    drop = float(base_value) - float(candidate_eval.metrics[key])
                    if drop > self.max_metric_regression:
                        regressed.append(key)
            accepted = delta >= self.min_improvement and not regressed
            reason = (
                "accepted: held-out score improved and no guarded metric regressed"
                if accepted else
                ("rejected: guarded metric regression in " + ", ".join(sorted(regressed))
                 if regressed else "rejected: improvement below minimum threshold")
            )
            record = ImprovementRecord(
                iteration_id=str(uuid.uuid4()),
                baseline_digest=baseline.digest,
                candidate_digest=candidate.digest,
                baseline_score=float(baseline_eval.score),
                candidate_score=float(candidate_eval.score),
                delta=float(delta),
                accepted=accepted,
                reason=reason,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self.history.append(record)
            if accepted:
                self.active_strategy = candidate
            results.append(RSIResult(
                active_strategy=self.active_strategy,
                proposed_strategy=candidate,
                baseline_evaluation=baseline_eval,
                candidate_evaluation=candidate_eval,
                record=record,
                history=tuple(self.history),
            ))
        return tuple(results)

    def rollback(self, artifact: StrategyArtifact) -> None:
        """Restore an explicitly retained, immutable strategy artifact."""
        if not isinstance(artifact, StrategyArtifact):
            raise TypeError("rollback target must be a StrategyArtifact")
        self.active_strategy = artifact


__all__ = [
    "SCHEMA_VERSION", "StrategyArtifact", "Evaluation",
    "ImprovementRecord", "RSIResult", "BoundedRSI",
]
