from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ada.learning import (
    ArtifactStatus, ExperienceKind, ExperienceLedger, ExperienceRecord,
    ExperienceStatus, LearningArtifact, LearningController, LearningProposal,
    MetricSet, MetricPolicy, OutcomeKind, RegressionGate, ReplayBuffer,
)


NOW = datetime(2026, 11, 9, tzinfo=timezone.utc)
DIGEST = "a" * 64
DATASET = "b" * 64


def record(**kwargs):
    values = dict(
        kind=ExperienceKind.FEEDBACK, domain="coding", task_id="task-1",
        source_kind="evaluation", source_id="run-1", source_digest=DIGEST,
        payload={"prompt": "fix bug", "result": {"passed": True}},
        outcome=OutcomeKind.SUCCESS, quality_score=0.9, importance=0.8,
        status=ExperienceStatus.CANDIDATE, created_at=NOW,
    )
    values.update(kwargs)
    return ExperienceRecord(**values)


def metrics(score, safety=1.0, evaluator="fixed", dataset=DATASET):
    return MetricSet(
        evaluator_id=evaluator, dataset_digest=dataset,
        metrics={"quality": score, "safety": safety}, sample_count=10,
        evaluated_at=NOW,
    )


def test_experience_record_is_immutable_and_provenance_aware():
    item = record()
    assert len(item.digest) == 64
    with pytest.raises(TypeError):
        item.payload["new"] = "value"
    with pytest.raises(ValueError):
        record(source_digest="invalid")
    with pytest.raises(ValueError):
        record(importance=1.1)


def test_ledger_requires_explicit_validation_and_preserves_identity():
    ledger = ExperienceLedger(capacity=2)
    candidate = ledger.ingest(record())
    assert candidate.status is ExperienceStatus.CANDIDATE
    with pytest.raises(ValueError):
        ledger.ingest(record(status=ExperienceStatus.VALIDATED))
    validated = ledger.validate(candidate.experience_id)
    assert validated.status is ExperienceStatus.VALIDATED
    assert validated.digest != candidate.digest
    assert validated.experience_id == candidate.experience_id


def test_ledger_is_idempotent_and_rejects_id_collision():
    ledger = ExperienceLedger()
    item = record()
    assert ledger.ingest(item) == item
    assert ledger.ingest(item) == item
    altered = record(experience_id=item.experience_id, payload={"different": True})
    with pytest.raises(ValueError, match="collision"):
        ledger.ingest(altered)


def test_ledger_capacity_fails_closed():
    ledger = ExperienceLedger(capacity=1)
    ledger.ingest(record())
    with pytest.raises(OverflowError):
        ledger.ingest(record(task_id="task-2"))


def test_replay_only_accepts_validated_and_deduplicates():
    buffer = ReplayBuffer(capacity=2, seed=7)
    with pytest.raises(ValueError):
        buffer.add(record())
    valid = record(status=ExperienceStatus.VALIDATED)
    assert buffer.add(valid)
    assert not buffer.add(valid)
    assert len(buffer) == 1


def test_replay_sampling_is_bounded_and_domain_filtered():
    buffer = ReplayBuffer(capacity=10, seed=1)
    items = [
        record(status=ExperienceStatus.VALIDATED, experience_id=f"coding-{i}", task_id=f"t-{i}")
        for i in range(4)
    ] + [
        record(status=ExperienceStatus.VALIDATED, domain="math", experience_id=f"math-{i}", task_id=f"m-{i}")
        for i in range(2)
    ]
    for item in items:
        buffer.add(item)
    assert len(buffer.sample(3)) == 3
    assert all(x.domain == "coding" for x in buffer.sample(10, domain="coding"))
    assert buffer.sample(3, domain="unknown") == ()


def test_replay_evicts_low_importance_then_oldest():
    buffer = ReplayBuffer(capacity=2)
    low = record(status=ExperienceStatus.VALIDATED, experience_id="low", importance=0.1)
    high = record(status=ExperienceStatus.VALIDATED, experience_id="high", importance=0.9)
    new = record(status=ExperienceStatus.VALIDATED, experience_id="new", importance=0.8)
    buffer.add(low)
    buffer.add(high)
    buffer.add(new)
    assert buffer.get("low") is None
    assert buffer.get("high") is not None


def test_regression_gate_checks_identity_and_direction():
    gate = RegressionGate({
        "quality": MetricPolicy(higher_is_better=True, max_regression=0.01, min_improvement=0.02),
        "safety": MetricPolicy(higher_is_better=True, max_regression=0.0),
    })
    assert gate.evaluate(metrics(0.5), metrics(0.55)).accepted
    assert not gate.evaluate(metrics(0.5), metrics(0.51, safety=0.9)).accepted
    assert not gate.evaluate(metrics(0.5), metrics(0.6, evaluator="changed")).accepted
    assert not gate.evaluate(metrics(0.5), metrics(0.6, dataset="c" * 64)).accepted


def test_regression_gate_supports_lower_is_better_metrics():
    gate = RegressionGate({
        "loss": MetricPolicy(higher_is_better=False, max_regression=0.0, min_improvement=0.01)
    })
    baseline = MetricSet("eval", DATASET, {"loss": 1.0}, 5, NOW)
    better = MetricSet("eval", DATASET, {"loss": 0.9}, 5, NOW)
    worse = MetricSet("eval", DATASET, {"loss": 1.1}, 5, NOW)
    assert gate.evaluate(baseline, better).accepted
    assert not gate.evaluate(baseline, worse).accepted


def test_learning_controller_promotes_only_evaluated_improvements():
    active = LearningArtifact("ada-policy", 1, DIGEST, status=ArtifactStatus.PROMOTED)
    counter = {"quality": 0.5}

    def evaluator(artifact):
        score = 0.5 if artifact.artifact_digest == DIGEST else 0.7
        return metrics(score)

    def builder(parent, evidence):
        return LearningProposal(
            target=parent.target, parent_digest=parent.artifact_digest,
            proposal={"strategy": "better"}, rationale="validated feedback",
            proposer_id="test",
        )

    controller = LearningController(
        active, gate=RegressionGate({"quality": MetricPolicy(min_improvement=0.05)}),
        evaluator=evaluator, proposal_builder=builder,
    )
    attempt = controller.attempt([{"trace": "failure then fixed"}])
    assert attempt.decision.accepted
    assert controller.active.version == 2
    assert controller.active.status is ArtifactStatus.PROMOTED
    assert len(controller.history) == 1


def test_learning_controller_rejects_regression_and_keeps_parent():
    active = LearningArtifact("ada-policy", 1, DIGEST, status=ArtifactStatus.PROMOTED)

    def evaluator(artifact):
        return metrics(0.5 if artifact.artifact_digest == DIGEST else 0.4)

    def builder(parent, evidence):
        return LearningProposal(parent.target, parent.artifact_digest, {"bad": True}, "test", "test")

    controller = LearningController(
        active, gate=RegressionGate({"quality": MetricPolicy()}),
        evaluator=evaluator, proposal_builder=builder,
    )
    attempt = controller.attempt([{"evidence": 1}])
    assert not attempt.decision.accepted
    assert controller.active.artifact_digest == DIGEST
    assert controller.active.version == 1


def test_learning_controller_rejects_stale_parent_proposal():
    active = LearningArtifact("ada-policy", 1, DIGEST, status=ArtifactStatus.PROMOTED)
    controller = LearningController(
        active, gate=RegressionGate({"quality": MetricPolicy()}),
        evaluator=lambda artifact: metrics(0.5),
        proposal_builder=lambda parent, cases: LearningProposal(
            parent.target, "c" * 64, {"p": 1}, "stale", "test"
        ),
    )
    with pytest.raises(ValueError, match="parent digest"):
        controller.attempt([{"x": 1}])


def test_learning_facade_imports_without_torch():
    import ada.learning as learning
    assert learning.ExperienceLedger is ExperienceLedger
