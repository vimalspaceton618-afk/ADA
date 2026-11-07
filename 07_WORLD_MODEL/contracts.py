"""Immutable, provenance-aware contracts for ADA's bounded world model."""
from __future__ import annotations
import hashlib, json, math, uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = "1.0.0"

def utc_now():
    return datetime.now(timezone.utc)

def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")

def _aware(value, name):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")

def _confidence(value, name="confidence"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")

def freeze_value(value, path="value"):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = freeze_value(item, f"{path}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item, f"{path}[]") for item in value)
    raise TypeError(f"{path} must contain JSON-compatible values")

def to_primitive(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        _aware(value, "serialized datetime")
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {k: to_primitive(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_primitive(v) for v in value]
    return value

def digest_json(value):
    raw = json.dumps(to_primitive(value), sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True, slots=True)
class WorldFact:
    """One state assertion with explicit source provenance."""
    subject_id: str
    predicate: str
    value: Any
    source_kind: str
    source_id: str
    source_digest: str
    confidence: float = 1.0
    observed_at: datetime = field(default_factory=utc_now)
    valid_until: datetime | None = None
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        for name in ("subject_id", "predicate", "source_kind", "source_id", "fact_id"):
            _text(getattr(self, name), name)
        if not isinstance(self.source_digest, str) or len(self.source_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_digest
        ):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")
        _confidence(self.confidence)
        _aware(self.observed_at, "observed_at")
        if self.valid_until is not None:
            _aware(self.valid_until, "valid_until")
            if self.valid_until <= self.observed_at:
                raise ValueError("valid_until must be later than observed_at")
        object.__setattr__(self, "value", freeze_value(self.value, "fact.value"))

    def to_dict(self):
        return to_primitive({k: getattr(self, k) for k in (
            "fact_id", "subject_id", "predicate", "value", "source_kind",
            "source_id", "source_digest", "confidence", "observed_at", "valid_until"
        )})

@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    """Immutable point-in-time set of world facts."""
    facts: tuple[WorldFact, ...] = ()
    timestamp: datetime = field(default_factory=utc_now)
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        _aware(self.timestamp, "timestamp")
        _text(self.snapshot_id, "snapshot_id")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported world model schema: {self.schema_version}")
        facts = tuple(self.facts)
        if any(not isinstance(x, WorldFact) for x in facts):
            raise TypeError("facts must contain WorldFact values")
        ids = [x.fact_id for x in facts]
        if len(ids) != len(set(ids)):
            raise ValueError("snapshot fact IDs must be unique")
        object.__setattr__(self, "facts", facts)

    @property
    def digest(self):
        return digest_json({"schema_version": self.schema_version,
                            "timestamp": self.timestamp,
                            "facts": [x.to_dict() for x in self.facts]})

    def active_facts(self):
        return tuple(x for x in self.facts if x.observed_at <= self.timestamp
                     and (x.valid_until is None or self.timestamp < x.valid_until))

@dataclass(frozen=True, slots=True)
class WorldAction:
    """Caller-supplied hypothetical action descriptor; never executed here."""
    action_type: str
    target_entity_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        for name in ("action_type", "target_entity_id", "action_id"):
            _text(getattr(self, name), name)
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        object.__setattr__(self, "parameters", freeze_value(self.parameters, "action.parameters"))

@dataclass(frozen=True, slots=True)
class FactCondition:
    subject_id: str
    predicate: str
    value: Any
    def __post_init__(self):
        _text(self.subject_id, "subject_id")
        _text(self.predicate, "predicate")
        object.__setattr__(self, "value", freeze_value(self.value, "condition.value"))

@dataclass(frozen=True, slots=True)
class FactEffect:
    subject_id: str
    predicate: str
    value: Any
    confidence: float = 1.0
    def __post_init__(self):
        _text(self.subject_id, "subject_id")
        _text(self.predicate, "predicate")
        _confidence(self.confidence)
        object.__setattr__(self, "value", freeze_value(self.value, "effect.value"))

@dataclass(frozen=True, slots=True)
class TransitionRule:
    """Explicit declarative transition; not a learned causal guarantee."""
    action_type: str
    preconditions: tuple[FactCondition, ...]
    effects: tuple[FactEffect, ...]
    confidence: float = 1.0
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    def __post_init__(self):
        _text(self.action_type, "action_type")
        _text(self.rule_id, "rule_id")
        _confidence(self.confidence)
        conditions, effects = tuple(self.preconditions), tuple(self.effects)
        if not conditions or not effects:
            raise ValueError("transition rules require preconditions and effects")
        if any(not isinstance(x, FactCondition) for x in conditions):
            raise TypeError("preconditions must contain FactCondition values")
        if any(not isinstance(x, FactEffect) for x in effects):
            raise TypeError("effects must contain FactEffect values")
        object.__setattr__(self, "preconditions", conditions)
        object.__setattr__(self, "effects", effects)

class PredictionStatus(str, Enum):
    PREDICTED = "predicted"
    NO_MODEL = "no_model"
    INAPPLICABLE = "inapplicable"
    AMBIGUOUS = "ambiguous"

@dataclass(frozen=True, slots=True)
class WorldPrediction:
    status: PredictionStatus
    base_snapshot_id: str
    action: WorldAction
    predicted_facts: tuple[WorldFact, ...] = ()
    rule_id: str | None = None
    confidence: float = 0.0
    uncertainty: float = 1.0
    horizon_seconds: float = 1.0
    explanation: str = ""
    model_kind: str = "declarative_rule_baseline"
    def __post_init__(self):
        if not isinstance(self.status, PredictionStatus):
            raise TypeError("status must be PredictionStatus")
        _text(self.base_snapshot_id, "base_snapshot_id")
        if not isinstance(self.action, WorldAction):
            raise TypeError("action must be WorldAction")
        _confidence(self.confidence)
        _confidence(self.uncertainty, "uncertainty")
        if isinstance(self.horizon_seconds, bool) or not isinstance(self.horizon_seconds, (int, float)):
            raise TypeError("horizon_seconds must be numeric")
        if not math.isfinite(float(self.horizon_seconds)) or self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be finite and positive")
        if abs(self.confidence + self.uncertainty - 1.0) > 1e-9:
            raise ValueError("confidence and uncertainty must sum to 1")
        facts = tuple(self.predicted_facts)
        if any(not isinstance(x, WorldFact) for x in facts):
            raise TypeError("predicted_facts must contain WorldFact values")
        object.__setattr__(self, "predicted_facts", facts)
        if self.status is PredictionStatus.PREDICTED:
            if not facts or not self.rule_id:
                raise ValueError("predicted outcomes require facts and a rule_id")
        elif facts or self.rule_id is not None:
            raise ValueError("abstained predictions cannot carry facts or a rule_id")

@dataclass(frozen=True, slots=True)
class SimulationResult:
    prediction: WorldPrediction
    future_snapshot: WorldSnapshot | None
    simulated_at: datetime = field(default_factory=utc_now)
    def __post_init__(self):
        _aware(self.simulated_at, "simulated_at")
        if not isinstance(self.prediction, WorldPrediction):
            raise TypeError("prediction must be WorldPrediction")
        if self.future_snapshot is not None and not isinstance(self.future_snapshot, WorldSnapshot):
            raise TypeError("future_snapshot must be WorldSnapshot or None")
        success = self.prediction.status is PredictionStatus.PREDICTED
        if success != (self.future_snapshot is not None):
            raise ValueError("only successful simulations may contain a future snapshot")
