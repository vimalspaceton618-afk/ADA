"""Deterministic, direction-aware regression gates for candidate learning."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from ..contracts import LearningDecision, MetricSet


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    """How one metric is compared; higher_is_better defaults to True."""
    higher_is_better: bool = True
    max_regression: float = 0.0
    min_improvement: float = 0.0

    def __post_init__(self) -> None:
        if type(self.higher_is_better) is not bool:
            raise TypeError("higher_is_better must be bool")
        for name, value in (("max_regression", self.max_regression), ("min_improvement", self.min_improvement)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")


@dataclass(frozen=True, slots=True)
class GateResult:
    accepted: bool
    reason: str
    deltas: object


class RegressionGate:
    """Require reproducible evaluation identity and guard all declared metrics."""

    def __init__(
        self,
        policies: dict[str, MetricPolicy],
        *,
        require_improvement: bool = True,
    ) -> None:
        if not policies:
            raise ValueError("at least one metric policy is required")
        if any(not isinstance(k, str) or not k.strip() for k in policies):
            raise ValueError("metric policy names must be non-empty")
        if any(not isinstance(v, MetricPolicy) for v in policies.values()):
            raise TypeError("policies must contain MetricPolicy values")
        if type(require_improvement) is not bool:
            raise TypeError("require_improvement must be bool")
        self.policies = MappingProxyType(dict(policies))
        self.require_improvement = require_improvement

    def evaluate(self, baseline: MetricSet, candidate: MetricSet) -> GateResult:
        if not isinstance(baseline, MetricSet) or not isinstance(candidate, MetricSet):
            raise TypeError("baseline and candidate must be MetricSet")
        if baseline.evaluator_id != candidate.evaluator_id:
            return GateResult(False, "evaluator identity changed", MappingProxyType({}))
        if baseline.dataset_digest != candidate.dataset_digest:
            return GateResult(False, "evaluation dataset changed", MappingProxyType({}))
        if candidate.sample_count != baseline.sample_count:
            return GateResult(False, "evaluation sample count changed", MappingProxyType({}))
        missing = set(self.policies) - set(candidate.metrics)
        if missing:
            return GateResult(False, f"candidate is missing guarded metrics: {sorted(missing)}", MappingProxyType({}))
        missing_baseline = set(self.policies) - set(baseline.metrics)
        if missing_baseline:
            return GateResult(False, f"baseline is missing guarded metrics: {sorted(missing_baseline)}", MappingProxyType({}))

        deltas = {}
        improved = False
        for name, policy in self.policies.items():
            raw_delta = candidate.metrics[name] - baseline.metrics[name]
            oriented = raw_delta if policy.higher_is_better else -raw_delta
            deltas[name] = oriented
            if oriented < -policy.max_regression:
                return GateResult(False, f"guarded metric regressed: {name}", MappingProxyType(deltas))
            if oriented >= policy.min_improvement and oriented > 0:
                improved = True
        if self.require_improvement and not improved:
            return GateResult(False, "no guarded metric improved", MappingProxyType(deltas))
        return GateResult(True, "candidate passed all regression gates", MappingProxyType(deltas))
