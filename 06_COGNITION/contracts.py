"""Auditable contracts for ADA's bounded cognitive orchestration layer."""
from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


SCHEMA_VERSION = "1.0.0"


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _strings(values, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CognitiveStatus(str, Enum):
    READY = "ready"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_CANDIDATES = "no_candidates"


class DecisionStatus(str, Enum):
    SELECTED = "selected"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class CognitiveTask:
    """A caller-defined objective; the engine does not invent user intent."""

    objective: str
    query: str | None = None
    constraints: tuple[str, ...] = ()
    candidate_actions: tuple[str, ...] = ()
    plan_steps: tuple[str, ...] = ()
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        _text(self.objective, "objective")
        _text(self.task_id, "task_id")
        if self.query is not None:
            _text(self.query, "query")
        object.__setattr__(self, "constraints", _strings(self.constraints, "constraints"))
        object.__setattr__(self, "candidate_actions", _strings(self.candidate_actions, "candidate_actions"))
        object.__setattr__(self, "plan_steps", _strings(self.plan_steps, "plan_steps"))


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    source_kind: str
    source_id: str
    source_digest: str
    content: str
    score: float = 0.0

    def __post_init__(self) -> None:
        for name in ("evidence_id", "source_kind", "source_id", "content"):
            _text(getattr(self, name), name)
        if not isinstance(self.source_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_digest):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")
        if not math.isfinite(float(self.score)) or not 0 <= self.score <= 1:
            raise ValueError("score must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class PlanStep:
    index: int
    instruction: str
    status: str = "proposed"

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be a non-negative integer")
        _text(self.instruction, "instruction")
        if self.status not in {"proposed", "blocked", "completed"}:
            raise ValueError("unsupported plan step status")


@dataclass(frozen=True, slots=True)
class CognitiveDecision:
    status: DecisionStatus
    selected_action: str | None
    confidence: float
    rationale: str
    candidate_scores: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, DecisionStatus):
            raise TypeError("status must be a DecisionStatus")
        if self.selected_action is not None:
            _text(self.selected_action, "selected_action")
        if self.status is DecisionStatus.SELECTED and self.selected_action is None:
            raise ValueError("selected decisions require selected_action")
        if self.status is DecisionStatus.ABSTAINED and self.selected_action is not None:
            raise ValueError("abstained decisions cannot select an action")
        if not math.isfinite(float(self.confidence)) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        _text(self.rationale, "rationale")
        scores = tuple(self.candidate_scores)
        for action, score in scores:
            _text(action, "candidate action")
            if not math.isfinite(float(score)) or not 0 <= score <= 1:
                raise ValueError("candidate scores must be finite and in [0, 1]")
        object.__setattr__(self, "candidate_scores", scores)


@dataclass(frozen=True, slots=True)
class CognitiveResult:
    task_id: str
    status: CognitiveStatus
    evidence: tuple[EvidenceItem, ...]
    plan: tuple[PlanStep, ...]
    decision: CognitiveDecision
    warnings: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _text(self.task_id, "task_id")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported cognition schema_version: {self.schema_version}")
        if not isinstance(self.status, CognitiveStatus):
            raise TypeError("status must be a CognitiveStatus")
        if not isinstance(self.decision, CognitiveDecision):
            raise TypeError("decision must be a CognitiveDecision")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "plan", tuple(self.plan))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
