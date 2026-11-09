"""Explicit adapters joining Memory, Cognition and World Model to Learning.

Adapters preserve upstream source identity and digest. They never validate
experience implicitly: callers must review each record before replay admission.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .contracts import (
    ExperienceKind, ExperienceRecord, ExperienceStatus, OutcomeKind,
)
from .experience_learning.ledger import ExperienceLedger
from .replay_mechanism.buffer import ReplayBuffer


def from_memory_record(
    memory_record: Any,
    *,
    domain: str,
    task_id: str,
    kind: ExperienceKind = ExperienceKind.OBSERVATION,
) -> ExperienceRecord:
    """Adapt one 05_MEMORY MemoryRecord without changing its provenance."""
    from importlib import import_module

    memory_contracts = import_module("05_MEMORY.contracts")
    if not isinstance(memory_record, memory_contracts.MemoryRecord):
        raise TypeError("memory_record must be a 05_MEMORY MemoryRecord")
    if memory_record.status is not memory_contracts.MemoryStatus.ACTIVE:
        raise ValueError("only active memory records can enter learning intake")
    return ExperienceRecord(
        kind=kind,
        domain=domain,
        task_id=task_id,
        source_kind=f"memory:{memory_record.source_kind.value}",
        source_id=memory_record.memory_id,
        source_digest=memory_record.source_digest,
        payload={
            "memory_kind": memory_record.kind.value,
            "namespace": memory_record.namespace,
            "content": memory_record.content,
            "tags": memory_record.tags,
            "memory_content_digest": memory_record.content_digest,
        },
        outcome=OutcomeKind.UNKNOWN,
        status=ExperienceStatus.CANDIDATE,
        tags=tuple(memory_record.tags),
        created_at=memory_record.created_at,
    )


def from_cognitive_evidence(
    evidence: Any,
    *,
    domain: str,
    task_id: str,
) -> ExperienceRecord:
    """Adapt one 06_COGNITION EvidenceItem; heuristic scores are not rewards."""
    from importlib import import_module

    cognition_contracts = import_module("06_COGNITION.contracts")
    if not isinstance(evidence, cognition_contracts.EvidenceItem):
        raise TypeError("evidence must be a 06_COGNITION EvidenceItem")
    return ExperienceRecord(
        kind=ExperienceKind.FEEDBACK,
        domain=domain,
        task_id=task_id,
        source_kind=f"cognition:{evidence.source_kind}",
        source_id=evidence.evidence_id,
        source_digest=evidence.source_digest,
        payload={
            "content": evidence.content,
            "evidence_score": float(evidence.score),
            "score_semantics": "cognition evidence relevance; not a reward or correctness label",
        },
        outcome=OutcomeKind.UNKNOWN,
        quality_score=None,
        importance=float(evidence.score),
        status=ExperienceStatus.CANDIDATE,
    )


def from_episode_window(
    window: Any,
    *,
    domain: str,
    task_id: str | None = None,
) -> ExperienceRecord:
    """Adapt one 07_WORLD_MODEL EpisodeWindow without crossing episode bounds."""
    from importlib import import_module

    trajectory_contracts = import_module("07_WORLD_MODEL.temporal_model.trajectory")
    if not isinstance(window, trajectory_contracts.EpisodeWindow):
        raise TypeError("window must be a 07_WORLD_MODEL EpisodeWindow")
    return ExperienceRecord(
        kind=ExperienceKind.WORLD_MODEL,
        domain=domain,
        task_id=task_id or window.episode_id,
        source_kind="world_model:episode_window",
        source_id=f"{window.episode_id}:{window.start_index}",
        source_digest=window.source_digest,
        payload={
            "episode_id": window.episode_id,
            "start_index": window.start_index,
            "observations": window.observations,
            "actions": window.actions,
            "rewards": window.rewards,
            "terminated": window.terminated,
            "timestamps": tuple(stamp.isoformat() for stamp in window.timestamps),
            "transition_count": len(window.actions),
        },
        outcome=OutcomeKind.UNKNOWN,
        status=ExperienceStatus.CANDIDATE,
        tags=("world-model", "episode-window"),
        created_at=window.timestamps[0],
    )


class LearningIntegration:
    """Small explicit handoff coordinator for validated learning evidence."""

    def __init__(
        self,
        *,
        ledger: ExperienceLedger | None = None,
        replay: ReplayBuffer | None = None,
    ) -> None:
        self.ledger = ledger or ExperienceLedger()
        self.replay = replay or ReplayBuffer()

    def ingest(self, records: Iterable[ExperienceRecord]) -> tuple[ExperienceRecord, ...]:
        accepted = []
        for record in records:
            accepted.append(self.ledger.ingest(record))
        return tuple(accepted)

    def validate_for_replay(self, experience_id: str) -> ExperienceRecord:
        """Explicit human/evaluator-reviewed transition before replay admission."""
        validated = self.ledger.validate(experience_id)
        self.replay.add(validated)
        return validated

    def reject(self, experience_id: str) -> ExperienceRecord:
        """Reject a candidate without admitting it to replay."""
        return self.ledger.reject(experience_id)
