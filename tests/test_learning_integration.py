from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib import import_module

import pytest

from ada.learning import (
    ExperienceKind, ExperienceStatus, LearningIntegration,
)
from  importlib import import_module as _import_module
from  importlib import import_module as _load


DIGEST = "a" * 64
NOW = datetime(2026, 11, 9, tzinfo=timezone.utc)


def test_memory_record_adapter_preserves_source_provenance():
    memory = _import_module("05_MEMORY.contracts")
    record = memory.MemoryRecord(
        kind=memory.MemoryKind.EPISODIC,
        content={"event": "verified"},
        source_kind=memory.MemorySourceKind.USER,
        source_id="session-4",
        source_digest=DIGEST,
        created_at=NOW,
        updated_at=NOW,
    )
    learning = _import_module("08_LEARNING.integration")
    adapted = learning.from_memory_record(record, domain="general", task_id="task-4")
    assert adapted.source_id == record.memory_id
    assert adapted.source_digest == DIGEST
    assert adapted.kind is ExperienceKind.OBSERVATION
    assert adapted.status is ExperienceStatus.CANDIDATE
    assert adapted.payload["memory_content_digest"] == record.content_digest


def test_cognition_evidence_adapter_does_not_promote_heuristic_score_to_reward():
    cognition = _import_module("06_COGNITION.contracts")
    evidence = cognition.EvidenceItem(
        evidence_id="ev-1", source_kind="memory", source_id="mem-1",
        source_digest=DIGEST, content="evidence text", score=0.8,
    )
    learning = _import_module("08_LEARNING.integration")
    adapted = learning.from_cognitive_evidence(evidence, domain="reasoning", task_id="task-1")
    assert adapted.kind is ExperienceKind.FEEDBACK
    assert adapted.quality_score is None
    assert adapted.outcome.value == "unknown"
    assert adapted.payload["evidence_score"] == 0.8


def test_world_model_window_adapter_preserves_episode_boundary_and_digest():
    trajectory = _import_module("07_WORLD_MODEL.temporal_model.trajectory")
    stamps = tuple(NOW + timedelta(seconds=i) for i in range(3))
    episode = trajectory.EpisodeTrajectory(
        episode_id="episode-9",
        observations=((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)),
        actions=((0.1,), (0.2,)),
        rewards=(0.5, 1.0),
        terminated=(False, True),
        timestamps=stamps,
        source_digest=DIGEST,
    )
    window = episode.window(0, 3)
    learning = _import_module("08_LEARNING.integration")
    adapted = learning.from_episode_window(window, domain="control")
    assert adapted.source_id == "episode-9:0"
    assert adapted.source_digest == DIGEST
    assert adapted.payload["transition_count"] == 2
    assert adapted.payload["terminated"] == (False, True)


def test_learning_integration_requires_explicit_validation_before_replay():
    learning = _import_module("08_LEARNING.integration")
    contracts = _import_module("08_LEARNING.contracts")
    integration = LearningIntegration()
    item = contracts.ExperienceRecord(
        kind=contracts.ExperienceKind.OBSERVATION,
        domain="general", task_id="task-1", source_kind="test",
        source_id="source-1", source_digest=DIGEST,
        payload={"observation": "candidate"},
    )
    integration.ingest((item,))
    assert len(integration.ledger) == 1
    assert len(integration.replay) == 0
    validated = integration.validate_for_replay(item.experience_id)
    assert validated.status is ExperienceStatus.VALIDATED
    assert len(integration.replay) == 1


def test_learning_integration_rejects_without_replay_admission():
    learning = _import_module("08_LEARNING.integration")
    contracts = _import_module("08_LEARNING.contracts")
    integration = LearningIntegration()
    item = contracts.ExperienceRecord(
        kind=contracts.ExperienceKind.OBSERVATION,
        domain="general", task_id="task-1", source_kind="test",
        source_id="source-1", source_digest=DIGEST,
        payload={"observation": "bad"},
    )
    integration.ingest((item,))
    rejected = integration.reject(item.experience_id)
    assert rejected.status is ExperienceStatus.REJECTED
    with pytest.raises(ValueError):
        integration.validate_for_replay(item.experience_id)
    assert len(integration.replay) == 0
