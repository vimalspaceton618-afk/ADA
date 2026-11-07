"""Bounded, evidence-grounded world state and counterfactual prediction."""
from datetime import timedelta
import math
from typing import Iterable
from .contracts import (
    FactCondition, PredictionStatus, SimulationResult, TransitionRule,
    WorldAction, WorldFact, WorldPrediction, WorldSnapshot, utc_now,
)

class WorldModelEngine:
    """Maintain explicit state and evaluate caller-supplied declarative dynamics.

    v1 is a symbolic baseline. It does not learn physics, infer facts from raw
    media, execute actions, or claim calibrated predictive probabilities.
    """
    def __init__(self, initial: WorldSnapshot | None = None, *, max_history: int = 256):
        if type(max_history) is not int or max_history < 1:
            raise ValueError("max_history must be a positive integer")
        self._current = initial or WorldSnapshot()
        if not isinstance(self._current, WorldSnapshot):
            raise TypeError("initial must be a WorldSnapshot")
        self._history = [self._current]
        self._max_history = max_history

    @property
    def current(self):
        return self._current

    @property
    def history(self):
        return tuple(self._history)

    def observe(self, facts: Iterable[WorldFact], *, at=None):
        """Append validated observations, replacing each subject/predicate slot."""
        timestamp = at or utc_now()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("observation time must be timezone-aware")
        if timestamp < self._current.timestamp:
            raise ValueError("world observations must be monotonic in time")
        incoming = tuple(facts)
        if any(not isinstance(x, WorldFact) for x in incoming):
            raise TypeError("facts must contain WorldFact values")
        if any(x.observed_at > timestamp for x in incoming):
            raise ValueError("a fact cannot be observed after the snapshot timestamp")
        latest = {(x.subject_id, x.predicate): x for x in self._current.active_facts()}
        for item in incoming:
            key = (item.subject_id, item.predicate)
            previous = latest.get(key)
            if previous is not None and item.observed_at < previous.observed_at:
                raise ValueError("stale observations cannot overwrite newer facts")
            latest[key] = item
        snapshot = WorldSnapshot(tuple(latest.values()), timestamp=timestamp)
        self._current = snapshot
        self._history.append(snapshot)
        self._history = self._history[-self._max_history:]
        return snapshot

    @staticmethod
    def _matching_fact(snapshot, condition: FactCondition):
        for fact in snapshot.active_facts():
            if (fact.subject_id == condition.subject_id and
                fact.predicate == condition.predicate and fact.value == condition.value):
                return fact
        return None

    def predict(self, action: WorldAction, rules: Iterable[TransitionRule], *,
                snapshot: WorldSnapshot | None = None, horizon_seconds: float = 1.0):
        """Predict one unambiguous rule outcome without mutating current state."""
        state = snapshot or self._current
        if not isinstance(state, WorldSnapshot):
            raise TypeError("snapshot must be a WorldSnapshot")
        if not isinstance(action, WorldAction):
            raise TypeError("action must be a WorldAction")
        if isinstance(horizon_seconds, bool) or not isinstance(horizon_seconds, (int, float)):
            raise TypeError("horizon_seconds must be numeric")
        if not math.isfinite(float(horizon_seconds)) or horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be finite and positive")
        candidates = tuple(x for x in rules if isinstance(x, TransitionRule)
                           and x.action_type == action.action_type)
        if not candidates:
            return WorldPrediction(PredictionStatus.NO_MODEL, state.snapshot_id, action,
                horizon_seconds=horizon_seconds, explanation="No rule is registered for this action type.")
        matches = []
        for rule in candidates:
            evidence = tuple(self._matching_fact(state, c) for c in rule.preconditions)
            if all(x is not None for x in evidence):
                matches.append((rule, evidence))
        if not matches:
            return WorldPrediction(PredictionStatus.INAPPLICABLE, state.snapshot_id, action,
                horizon_seconds=horizon_seconds, explanation="No rule has all required preconditions.")
        if len(matches) > 1:
            return WorldPrediction(PredictionStatus.AMBIGUOUS, state.snapshot_id, action,
                horizon_seconds=horizon_seconds, explanation="Multiple rules match; refusing arbitrary selection.")
        rule, evidence = matches[0]
        base_confidence = min([float(rule.confidence), *(float(x.confidence) for x in evidence)])
        future_time = state.timestamp + timedelta(seconds=float(horizon_seconds))
        predicted = tuple(WorldFact(
            subject_id=e.subject_id, predicate=e.predicate, value=e.value,
            source_kind="world_model_prediction", source_id=rule.rule_id,
            source_digest=state.digest,
            confidence=min(base_confidence, float(e.confidence)), observed_at=future_time,
        ) for e in rule.effects)
        confidence = min([base_confidence, *(float(x.confidence) for x in predicted)])
        return WorldPrediction(PredictionStatus.PREDICTED, state.snapshot_id, action,
            predicted_facts=predicted, rule_id=rule.rule_id, confidence=confidence,
            uncertainty=1.0-confidence, horizon_seconds=float(horizon_seconds),
            explanation=rule.description or "Outcome derived from one matching declarative transition rule.")

    def simulate(self, action: WorldAction, rules: Iterable[TransitionRule], *,
                 horizon_seconds: float = 1.0):
        """Build an isolated hypothetical future; never changes current state."""
        prediction = self.predict(action, rules, horizon_seconds=horizon_seconds)
        future = None
        if prediction.status is PredictionStatus.PREDICTED:
            # Predicted effects replace the same state slots in the hypothetical
            # future; the historical snapshot and live engine remain untouched.
            future_facts = {
                (item.subject_id, item.predicate): item
                for item in self._current.active_facts()
            }
            for item in prediction.predicted_facts:
                future_facts[(item.subject_id, item.predicate)] = item
            future = WorldSnapshot(
                facts=tuple(future_facts.values()),
                timestamp=self._current.timestamp + timedelta(seconds=float(horizon_seconds)),
            )
        return SimulationResult(prediction, future)
