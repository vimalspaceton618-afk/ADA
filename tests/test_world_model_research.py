from datetime import datetime, timedelta, timezone
import pytest

from ada.world_model import (
    CausalGraph, EmpiricalActionPrior, EpisodeTrajectory, ObservedBehaviorEvent,
    PhysicalState, mean_absolute_error, mean_squared_error,
    expected_calibration_error, horizon_mse, ModelRuntimeState,
)

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
DIGEST = "b" * 64


def test_causal_graph_topology_and_descendants():
    graph = CausalGraph(("sensor", "state", "prediction"), (("sensor", "state"), ("state", "prediction")))
    assert graph.topological_order() == ("sensor", "state", "prediction")
    assert graph.descendants("sensor") == ("state", "prediction")
    assert graph.parents("prediction") == ("state",)


def test_causal_graph_rejects_cycles_and_unknown_edges():
    with pytest.raises(ValueError):
        CausalGraph(("a", "b"), (("a", "b"), ("b", "a")))
    with pytest.raises(ValueError):
        CausalGraph(("a",), (("a", "missing"),))


def test_episode_trajectory_builds_windows_without_crossing_boundaries():
    trajectory = EpisodeTrajectory(
        "episode-1",
        ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)),
        ((0.1,), (0.2,), (0.3,)),
        (1.0, 1.0, 0.0),
        (False, False, True),
        (NOW, NOW + timedelta(seconds=1), NOW + timedelta(seconds=2), NOW + timedelta(seconds=3)),
        DIGEST,
    )
    window = trajectory.window(1, 3)
    assert window.episode_id == "episode-1"
    assert window.start_index == 1
    assert len(window.observations) == 3
    assert len(window.actions) == 2
    assert window.terminated[-1] is True
    with pytest.raises(ValueError):
        trajectory.window(2, 3)


def test_episode_trajectory_rejects_misalignment_and_non_monotonic_time():
    with pytest.raises(ValueError):
        EpisodeTrajectory("bad", ((0.0,), (1.0,)), (), (), (), (NOW, NOW + timedelta(seconds=1)), DIGEST)
    with pytest.raises(ValueError):
        EpisodeTrajectory("bad", ((0.0,), (1.0,)), ((0.1,),), (1.0,), (False,),
                          (NOW, NOW), DIGEST)


def test_physical_state_requires_units_coordinates_and_provenance():
    state = PhysicalState("object-1", "world", (0, 1, 2), (1, 0, 0), (0, 0, -9.81),
                          observed_at=NOW, source_id="imu-1", source_digest=DIGEST)
    assert state.velocity_m_s == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        PhysicalState("object-1", "world", (0, 1, 2), (1, 0, 0), (0, 0, 0),
                      observed_at=NOW, source_id="imu-1", source_digest="")


def test_behavior_frequency_counts_multiple_events_from_same_source():
    events = [
        ObservedBehaviorEvent("anon-a", "move_left", "intersection", NOW, "cam-a", DIGEST),
        ObservedBehaviorEvent("anon-b", "move_right", "intersection", NOW + timedelta(seconds=1), "cam-a", DIGEST),
        ObservedBehaviorEvent("anon-c", "move_left", "intersection", NOW + timedelta(seconds=2), "cam-b", "c" * 64),
    ]
    prior = EmpiricalActionPrior(events, smoothing=0.5)
    distribution = prior.distribution("intersection")
    assert prior.counts("intersection") == {"move_left": 2, "move_right": 1}
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_prediction_metrics_and_horizon_drift():
    assert mean_squared_error([1, 2], [2, 2]) == pytest.approx(0.5)
    assert mean_absolute_error([1, 2], [2, 2]) == pytest.approx(0.5)
    assert expected_calibration_error([0.8, 0.9], [1, 1], bins=2) == pytest.approx(0.15)
    per_horizon = horizon_mse(
        [[[0.0], [1.0]], [[1.0], [2.0]]],
        [[[0.0], [0.0]], [[1.0], [0.0]]],
    )
    assert per_horizon == pytest.approx((0.0, 2.5))


def test_runtime_state_reference_validates_digest_and_uncertainty():
    state = ModelRuntimeState("ep-1", 3, "snapshot-1", "rssm-v0", 0.25, NOW, DIGEST)
    assert state.step_index == 3
    with pytest.raises(ValueError):
        ModelRuntimeState("ep-1", 3, "snapshot-1", "rssm-v0", 1.5, NOW, DIGEST)


