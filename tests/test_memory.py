from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ada.memory import (
    MemoryConflictError, MemoryEngine, MemoryKind, MemoryQuery, MemoryRecord,
    MemorySourceKind, MemoryStatus, InMemoryMemoryStore,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
DIGEST = "a" * 64


def make_record(**kwargs):
    values = dict(
        kind=MemoryKind.EPISODIC,
        content={"text": "The system observed a verified event."},
        source_kind=MemorySourceKind.USER,
        source_id="source-1",
        source_digest=DIGEST,
        created_at=NOW,
        updated_at=NOW,
    )
    values.update(kwargs)
    return MemoryRecord(**values)


def test_memory_record_is_deeply_immutable_and_serializable():
    record = make_record(content={"nested": {"items": [1, 2]}})
    with pytest.raises(TypeError):
        record.content["nested"]["new"] = 3
    assert record.to_dict()["content"]["nested"]["items"] == [1, 2]
    assert len(record.content_digest) == 64


def test_contract_rejects_bad_digest_nonfinite_and_naive_datetime():
    with pytest.raises(ValueError, match="SHA-256"):
        make_record(source_digest="bad")
    with pytest.raises(ValueError, match="non-finite"):
        make_record(content={"score": float("nan")})
    with pytest.raises(ValueError, match="timezone-aware"):
        make_record(created_at=datetime(2026, 1, 1), updated_at=NOW)


def test_namespace_isolation_and_deterministic_lexical_retrieval():
    engine = MemoryEngine()
    first = engine.remember(
        {"text": "Verified event from the research pipeline."},
        kind=MemoryKind.EPISODIC, source_kind=MemorySourceKind.USER,
        source_id="s1", source_digest=DIGEST, namespace="alpha",
    )
    engine.remember(
        {"text": "Verified event from another namespace."},
        kind=MemoryKind.SEMANTIC, source_kind=MemorySourceKind.USER,
        source_id="s2", source_digest=DIGEST, namespace="beta",
    )
    matches = engine.retrieve("verified research pipeline", MemoryQuery(namespace="alpha"))
    assert len(matches) == 1
    assert matches[0].record.memory_id == first.memory_id
    assert matches[0].matched_terms == ("verified", "research", "pipeline")
    assert matches[0].score == 1.0


def test_query_filters_kind_tags_and_source():
    engine = MemoryEngine()
    record = engine.remember(
        {"text": "Procedure for safe checkpoint recovery."},
        kind=MemoryKind.PROCEDURAL, source_kind=MemorySourceKind.SYSTEM,
        source_id="trainer", source_digest=DIGEST, tags=("training", "recovery"),
    )
    result = engine.retrieve(
        "checkpoint recovery",
        MemoryQuery(kinds=(MemoryKind.PROCEDURAL,), tags=("training",), source_id="trainer"),
    )
    assert [item.record.memory_id for item in result] == [record.memory_id]


def test_archive_is_hidden_by_default_and_can_be_included():
    engine = MemoryEngine()
    record = engine.remember(
        {"text": "Archived knowledge entry."}, kind=MemoryKind.SEMANTIC,
        source_kind=MemorySourceKind.USER, source_id="s1", source_digest=DIGEST,
    )
    engine.archive(record.memory_id)
    assert engine.retrieve("archived knowledge") == ()
    assert len(engine.retrieve("archived knowledge", MemoryQuery(include_archived=True))) == 1


def test_delete_is_terminal_and_not_retrievable():
    engine = MemoryEngine()
    record = engine.remember(
        {"text": "Sensitive temporary note."}, kind=MemoryKind.WORKING,
        source_kind=MemorySourceKind.USER, source_id="s1", source_digest=DIGEST,
    )
    engine.delete(record.memory_id)
    with pytest.raises(KeyError):
        engine.get(record.memory_id)
    with pytest.raises(ValueError, match="terminal"):
        engine.store.replace_status(record.memory_id, MemoryStatus.ACTIVE)


def test_expired_records_are_excluded_and_purge_marks_deleted():
    engine = MemoryEngine()
    record = engine.remember(
        {"text": "Expired session marker."}, kind=MemoryKind.WORKING,
        source_kind=MemorySourceKind.USER, source_id="s1", source_digest=DIGEST,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=10),
    )
    assert engine.retrieve("session marker") == ()
    assert engine.purge_expired(at=datetime.now(timezone.utc)) == 1
    with pytest.raises(KeyError):
        engine.get(record.memory_id)


def test_optimistic_version_check_prevents_lost_update():
    store = InMemoryMemoryStore()
    record = make_record()
    assert store.put(record) == 1
    assert store.replace_status(record.memory_id, MemoryStatus.ARCHIVED, expected_version=1) == 2
    with pytest.raises(MemoryConflictError):
        store.replace_status(record.memory_id, MemoryStatus.ACTIVE, expected_version=1)


def test_store_capacity_is_bounded():
    store = InMemoryMemoryStore(max_records=1)
    store.put(make_record())
    with pytest.raises(OverflowError, match="capacity"):
        store.put(make_record(memory_id="second"))


def test_memory_query_rejects_invalid_limit():
    with pytest.raises(ValueError, match="limit"):
        MemoryQuery(limit=0)


def test_representation_handoff_preserves_source_digest():
    from ada.perception import Modality, PerceptionEngine, PerceptionInput
    from ada.representation import RepresentationEngine

    source = PerceptionInput(modality=Modality.TEXT, content="ADA stores source grounded memory.")
    observed = PerceptionEngine().perceive(source)
    projected = RepresentationEngine().transform(observed)
    engine = MemoryEngine()
    stored = [engine.remember_representation(item) for item in projected.representations]
    assert stored
    assert all(item.source_digest == rep.source_content_sha256 for item, rep in zip(stored, projected.representations))
    assert all(item.metadata["source_input_id"] == source.input_id for item in stored)
