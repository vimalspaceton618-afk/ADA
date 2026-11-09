"""Proposal/evaluation/promotion controller for bounded learning updates."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import threading

from ..contracts import (
    ArtifactStatus, LearningArtifact, LearningDecision, LearningProposal, MetricSet,
)
from ..learning_evaluation.gates import RegressionGate


@dataclass(frozen=True, slots=True)
class LearningAttempt:
    proposal: LearningProposal
    candidate: LearningArtifact
    baseline: MetricSet
    evaluation: MetricSet
    decision: LearningDecision


class LearningController:
    """Coordinate a learning proposal without executing arbitrary code.

    A proposal builder and evaluator are supplied by the caller. The controller
    only promotes immutable artifact descriptors after the gate passes. It does
    not load checkpoints, update model parameters, call tools, or deploy.
    """

    def __init__(
        self,
        active: LearningArtifact,
        *,
        gate: RegressionGate,
        evaluator: Callable[[LearningArtifact], MetricSet],
        proposal_builder: Callable[[LearningArtifact, tuple], LearningProposal],
    ) -> None:
        if not isinstance(active, LearningArtifact):
            raise TypeError("active must be LearningArtifact")
        if not isinstance(gate, RegressionGate):
            raise TypeError("gate must be RegressionGate")
        if not callable(evaluator) or not callable(proposal_builder):
            raise TypeError("evaluator and proposal_builder must be callable")
        self._active = active
        self._gate = gate
        self._evaluator = evaluator
        self._proposal_builder = proposal_builder
        self._history: list[LearningAttempt] = []
        self._lock = threading.RLock()

    @property
    def active(self) -> LearningArtifact:
        with self._lock:
            return self._active

    @property
    def history(self) -> tuple[LearningAttempt, ...]:
        with self._lock:
            return tuple(self._history)

    def attempt(self, evidence: Iterable) -> LearningAttempt:
        cases = tuple(evidence)
        if not cases:
            raise ValueError("learning requires non-empty evidence")
        with self._lock:
            parent = self._active
        baseline = self._evaluator(parent)
        if not isinstance(baseline, MetricSet):
            raise TypeError("evaluator must return MetricSet")
        proposal = self._proposal_builder(parent, cases)
        if not isinstance(proposal, LearningProposal):
            raise TypeError("proposal_builder must return LearningProposal")
        if proposal.parent_digest != parent.artifact_digest:
            raise ValueError("proposal parent digest does not match active artifact")
        candidate = LearningArtifact(
            target=parent.target, version=parent.version + 1,
            artifact_digest=proposal.digest, parent_digest=parent.artifact_digest,
            metadata={"proposal_id": proposal.proposal_id, "proposer_id": proposal.proposer_id},
        )
        evaluation = self._evaluator(candidate)
        if not isinstance(evaluation, MetricSet):
            raise TypeError("evaluator must return MetricSet")
        result = self._gate.evaluate(baseline, evaluation)
        decision = LearningDecision(
            accepted=result.accepted, reason=result.reason,
            parent_digest=parent.artifact_digest, candidate_digest=candidate.artifact_digest,
            baseline_metrics_digest=baseline.digest, candidate_metrics_digest=evaluation.digest,
        )
        finalized = LearningArtifact(
            target=candidate.target, version=candidate.version,
            artifact_digest=candidate.artifact_digest, parent_digest=candidate.parent_digest,
            metadata=candidate.metadata,
            status=ArtifactStatus.PROMOTED if result.accepted else ArtifactStatus.REJECTED,
            artifact_id=candidate.artifact_id, created_at=candidate.created_at,
        )
        attempt = LearningAttempt(proposal, finalized, baseline, evaluation, decision)
        with self._lock:
            if self._active.artifact_digest != parent.artifact_digest:
                raise RuntimeError("active artifact changed during evaluation; refusing stale promotion")
            self._history.append(attempt)
            if result.accepted:
                self._active = finalized
        return attempt
