from __future__ import annotations

import pytest

from importlib import import_module

rsi = import_module("06_COGNITION.recursive_self_improvement")
BoundedRSI = rsi.BoundedRSI
Evaluation = rsi.Evaluation
StrategyArtifact = rsi.StrategyArtifact


CASES = ({"id": "case-1"}, {"id": "case-2"})
DATASET = "heldout-sha256"


def evaluator(strategy, cases):
    assert tuple(cases) == CASES
    score = float(strategy.parameters.get("quality", 0.5))
    return Evaluation(
        score=score,
        cases=2,
        passed=int(round(score * 2)),
        evaluator_id="independent-test-evaluator-v1",
        dataset_digest=DATASET,
        metrics={"correctness": score, "safety": float(strategy.parameters.get("safety", 1.0))},
    )


def test_rsi_promotes_only_when_minimum_improvement_passes():
    active = StrategyArtifact("cognition-policy", 1, {"quality": 0.5, "safety": 1.0})
    rsi = BoundedRSI(
        active,
        evaluator=evaluator,
        generator=lambda old, cases: {"quality": 0.7, "safety": 1.0},
        min_improvement=0.1,
        allowed_parameter_keys={"quality", "safety"},
    )
    result = rsi.iterate(CASES)[0]
    assert result.record.accepted
    assert result.active_strategy.version == 2
    assert result.record.delta == pytest.approx(0.2)
    assert result.record.baseline_digest == active.digest
    assert result.active_strategy.parent_digest == active.digest


def test_rsi_rejects_metric_regression_even_when_primary_score_improves():
    active = StrategyArtifact("cognition-policy", 1, {"quality": 0.5, "safety": 1.0})
    rsi = BoundedRSI(
        active,
        evaluator=evaluator,
        generator=lambda old, cases: {"quality": 0.8, "safety": 0.7},
        min_improvement=0.1,
    )
    result = rsi.iterate(CASES)[0]
    assert not result.record.accepted
    assert "safety" in result.record.reason
    assert rsi.active_strategy.digest == active.digest


def test_rsi_rejects_missing_guard_metric():
    active = StrategyArtifact("cognition-policy", 1, {"quality": 0.5, "safety": 1.0})

    def incomplete_evaluator(strategy, cases):
        return Evaluation(
            score=0.8 if strategy.version > 1 else 0.5,
            cases=2, passed=2 if strategy.version > 1 else 1,
            evaluator_id="same", dataset_digest=DATASET,
            metrics=(
                {"correctness": 0.5, "safety": 1.0}
                if strategy.version == 1
                else {"correctness": 0.8}
            ),
        )

    controller = BoundedRSI(
        active,
        evaluator=incomplete_evaluator,
        generator=lambda old, cases: {"quality": 0.8},
        min_improvement=0.1,
    )
    result = controller.iterate(CASES)[0]
    assert not result.record.accepted


def test_rsi_rejects_evaluator_or_dataset_mismatch():
    active = StrategyArtifact("cognition-policy", 1, {"quality": 0.5})

    def evaluator_with_drift(strategy, cases):
        return Evaluation(
            score=0.5 if strategy.version == 1 else 0.8,
            cases=2, passed=1 if strategy.version == 1 else 2,
            evaluator_id="baseline" if strategy.version == 1 else "changed",
            dataset_digest=DATASET, metrics={"quality": 0.5},
        )

    controller = BoundedRSI(
        active, evaluator=evaluator_with_drift,
        generator=lambda old, cases: {"quality": 0.8},
        min_improvement=0.1,
    )
    with pytest.raises(ValueError, match="same evaluator"):
        controller.iterate(CASES)


def test_rsi_rejects_protected_parameters():
    active = StrategyArtifact("cognition-policy", 1, {"quality": 0.5})
    controller = BoundedRSI(
        active, evaluator=evaluator,
        generator=lambda old, cases: {"quality": 0.8, "execution_permission": True},
        allowed_parameter_keys={"quality"},
    )
    with pytest.raises(ValueError, match="protected parameter"):
        controller.iterate(CASES)


def test_rsi_requires_heldout_cases_and_bounded_iterations():
    active = StrategyArtifact("cognition-policy", 1, {"quality": 0.5})
    controller = BoundedRSI(active, evaluator=evaluator, generator=lambda old, cases: old.parameters)
    with pytest.raises(ValueError, match="held-out"):
        controller.iterate(())
    with pytest.raises(ValueError, match="max_iterations"):
        controller.iterate(CASES, iterations=99)


def test_rsi_rollback_restores_retained_artifact():
    old = StrategyArtifact("cognition-policy", 1, {"quality": 0.5})
    controller = BoundedRSI(
        old, evaluator=evaluator,
        generator=lambda strategy, cases: {"quality": 0.7, "safety": 1.0},
    )
    controller.iterate(CASES)
    assert controller.active_strategy.version == 2
    controller.rollback(old)
    assert controller.active_strategy.digest == old.digest


def test_strategy_artifact_digest_is_stable_and_parameters_detached():
    params = {"quality": 0.5}
    artifact = StrategyArtifact("policy", 1, params)
    digest = artifact.digest
    params["quality"] = 0.9
    assert artifact.digest == digest


def test_evaluation_rejects_nan_and_invalid_counts():
    with pytest.raises(ValueError):
        Evaluation(float("nan"), 2, 1, "eval", DATASET, {"quality": 0.5})
    with pytest.raises(ValueError):
        Evaluation(0.5, 2, 3, "eval", DATASET, {"quality": 0.5})
