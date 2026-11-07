from datetime import datetime, timedelta, timezone
import pytest
from ada.world_model import (
    FactCondition, FactEffect, PredictionStatus, TransitionRule,
    WorldAction, WorldFact, WorldModelEngine, WorldSnapshot,
)
NOW = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
DIGEST = "a" * 64

def fact(value="off", *, at=NOW, confidence=0.9, valid_until=None):
    return WorldFact("lamp", "power", value, "sensor", "sensor-1", DIGEST,
                     confidence=confidence, observed_at=at, valid_until=valid_until)

def rule(rule_id="switch-on"):
    return TransitionRule("switch_on", (FactCondition("lamp", "power", "off"),),
        (FactEffect("lamp", "power", "on"),), confidence=0.95, rule_id=rule_id)

def test_fact_validates_provenance_confidence_and_time():
    with pytest.raises(ValueError):
        WorldFact("x", "p", 1, "sensor", "s", "bad", observed_at=NOW)
    with pytest.raises(ValueError):
        fact(confidence=1.2)
    with pytest.raises(ValueError):
        WorldFact("x", "p", 1, "sensor", "s", DIGEST, observed_at=datetime(2026, 1, 1))

def test_fact_values_are_deeply_immutable_and_json_only():
    item = fact({"nested": [1, {"ok": True}]})
    with pytest.raises(TypeError):
        item.value["new"] = 2
    with pytest.raises(TypeError):
        WorldFact("x", "p", object(), "sensor", "s", DIGEST, observed_at=NOW)

def test_snapshot_rejects_duplicate_ids_and_filters_expired_facts():
    old = fact(valid_until=NOW + timedelta(seconds=1))
    assert WorldSnapshot((old,), timestamp=NOW + timedelta(seconds=2)).active_facts() == ()
    with pytest.raises(ValueError):
        WorldSnapshot((old, old), timestamp=NOW)

def test_observe_replaces_slot_and_bounds_history():
    engine = WorldModelEngine(WorldSnapshot((fact(),), timestamp=NOW), max_history=2)
    state = engine.observe((fact("on", at=NOW + timedelta(seconds=1)),),
                           at=NOW + timedelta(seconds=1))
    assert len(state.active_facts()) == 1 and state.active_facts()[0].value == "on"
    engine.observe((), at=NOW + timedelta(seconds=2))
    assert len(engine.history) == 2

def test_observe_rejects_non_monotonic_or_future_fact():
    engine = WorldModelEngine(WorldSnapshot((fact(),), timestamp=NOW))
    with pytest.raises(ValueError):
        engine.observe((), at=NOW - timedelta(seconds=1))
    with pytest.raises(ValueError):
        engine.observe((fact(at=NOW + timedelta(seconds=5)),), at=NOW)

def test_stale_observation_cannot_overwrite_newer_state():
    engine = WorldModelEngine(WorldSnapshot((fact("on", at=NOW + timedelta(seconds=2)),),
                                             timestamp=NOW + timedelta(seconds=2)))
    with pytest.raises(ValueError):
        engine.observe((fact("off", at=NOW + timedelta(seconds=1)),),
                       at=NOW + timedelta(seconds=3))

def test_prediction_requires_rule_and_satisfied_preconditions():
    engine = WorldModelEngine(WorldSnapshot((fact(),), timestamp=NOW))
    action = WorldAction("switch_on", "lamp")
    assert engine.predict(action, []).status is PredictionStatus.NO_MODEL
    wrong = TransitionRule("switch_on", (FactCondition("lamp", "power", "on"),),
                           (FactEffect("lamp", "power", "off"),))
    assert engine.predict(action, [wrong]).status is PredictionStatus.INAPPLICABLE

def test_prediction_is_provenance_linked_and_uncertainty_bounded():
    engine = WorldModelEngine(WorldSnapshot((fact(),), timestamp=NOW))
    prediction = engine.predict(WorldAction("switch_on", "lamp"), [rule()], horizon_seconds=2)
    assert prediction.status is PredictionStatus.PREDICTED
    assert prediction.predicted_facts[0].value == "on"
    assert prediction.predicted_facts[0].source_digest == engine.current.digest
    assert prediction.confidence == pytest.approx(0.9)
    assert prediction.uncertainty == pytest.approx(0.1)
    assert prediction.horizon_seconds == 2

def test_ambiguous_rules_abstain():
    engine = WorldModelEngine(WorldSnapshot((fact(),), timestamp=NOW))
    result = engine.predict(WorldAction("switch_on", "lamp"), [rule(), rule("other")])
    assert result.status is PredictionStatus.AMBIGUOUS and not result.predicted_facts

def test_simulation_isolated_from_live_state_and_history():
    engine = WorldModelEngine(WorldSnapshot((fact(),), timestamp=NOW))
    digest, history = engine.current.digest, engine.history
    result = engine.simulate(WorldAction("switch_on", "lamp"), [rule()], horizon_seconds=3)
    assert result.future_snapshot.timestamp == NOW + timedelta(seconds=3)
    future_power = [x for x in result.future_snapshot.active_facts()
                    if x.subject_id == "lamp" and x.predicate == "power"]
    assert len(future_power) == 1 and future_power[0].value == "on"
    assert engine.current.digest == digest and engine.history == history
    assert engine.current.active_facts()[0].value == "off"

def test_no_model_simulation_has_no_future_snapshot():
    result = WorldModelEngine(WorldSnapshot((fact(),), timestamp=NOW)).simulate(
        WorldAction("teleport", "lamp"), [])
    assert result.future_snapshot is None
    assert result.prediction.status is PredictionStatus.NO_MODEL

def test_invalid_horizon_is_rejected():
    engine = WorldModelEngine(WorldSnapshot((fact(),), timestamp=NOW))
    with pytest.raises(ValueError):
        engine.predict(WorldAction("switch_on", "lamp"), [rule()], horizon_seconds=0)

def test_public_facade_exports_engine():
    from ada.world_model import WorldModelEngine as PublicEngine
    assert PublicEngine is WorldModelEngine
