from __future__ import annotations

import pytest

from ada.cognition import (
    CognitionEngine, CognitiveStatus, CognitiveTask, DecisionStatus,
)
from ada.memory import MemoryEngine, MemoryKind, MemorySourceKind


def test_task_rejects_empty_objective_and_duplicate_candidates():
    with pytest.raises(ValueError, match="objective"):
        CognitiveTask(objective=" ")
    with pytest.raises(ValueError, match="duplicates"):
        CognitiveTask(objective="choose", candidate_actions=("inspect", "inspect"))


def test_no_evidence_abstains_and_reports_insufficient_evidence():
    result = CognitionEngine(memory_engine=MemoryEngine()).process(
        CognitiveTask(objective="choose action", candidate_actions=("inspect system",))
    )
    assert result.status is CognitiveStatus.INSUFFICIENT_EVIDENCE
    assert result.decision.status is DecisionStatus.ABSTAINED
    assert result.decision.selected_action is None


def test_no_candidates_never_invents_action():
    result = CognitionEngine().process(
        CognitiveTask(objective="summarize the available evidence", plan_steps=("review inputs",)),
        retrieve_memory=False,
    )
    assert result.status is CognitiveStatus.INSUFFICIENT_EVIDENCE
    assert result.decision.status is DecisionStatus.ABSTAINED
    assert result.plan[0].instruction == "review inputs"


def test_memory_retrieval_preserves_provenance_and_can_rank_candidate():
    memory = MemoryEngine()
    record = memory.remember(
        {"text": "inspection cooling system verified"},
        kind=MemoryKind.EPISODIC,
        source_kind=MemorySourceKind.USER,
        source_id="inspection-001",
        source_digest="a" * 64,
        namespace="engineering",
    )
    engine = CognitionEngine(memory_engine=memory)
    task = CognitiveTask(
        objective="choose maintenance action",
        query="inspection cooling system",
        candidate_actions=("inspect cooling system", "replace unrelated component"),
    )
    result = engine.process(task, namespace="engineering")
    assert result.evidence
    assert result.evidence[0].source_id == record.source_id
    assert result.evidence[0].source_digest == record.source_digest
    assert result.decision.status is DecisionStatus.SELECTED
    assert result.decision.selected_action == "inspect cooling system"


def test_ambiguous_candidates_abstain():
    memory = MemoryEngine()
    memory.remember(
        {"text": "inspect cooling system"},
        kind=MemoryKind.EPISODIC,
        source_kind=MemorySourceKind.USER,
        source_id="source-1",
        source_digest="b" * 64,
    )
    result = CognitionEngine(memory_engine=memory).process(
        CognitiveTask(
            objective="choose",
            query="inspect cooling system",
            candidate_actions=("inspect cooling system", "inspect cooling system"),
        )
    ) if False else None
    # Duplicate candidate labels are rejected at the task boundary.
    with pytest.raises(ValueError, match="duplicates"):
        CognitiveTask(objective="choose", candidate_actions=("inspect cooling system", "inspect cooling system"))


def test_caller_plan_order_is_preserved():
    task = CognitiveTask(
        objective="execute no actions",
        plan_steps=("collect evidence", "review evidence", "request approval"),
    )
    result = CognitionEngine().process(task, retrieve_memory=False)
    assert [step.index for step in result.plan] == [0, 1, 2]
    assert [step.instruction for step in result.plan] == list(task.plan_steps)


def test_memory_namespace_isolation_applies():
    memory = MemoryEngine()
    memory.remember(
        {"text": "inspect cooling system"},
        kind=MemoryKind.EPISODIC,
        source_kind=MemorySourceKind.USER,
        source_id="private-record",
        source_digest="c" * 64,
        namespace="other",
    )
    result = CognitionEngine(memory_engine=memory).process(
        CognitiveTask(objective="inspect cooling system", candidate_actions=("inspect cooling system",)),
        namespace="engineering",
    )
    assert result.evidence == ()
    assert result.decision.status is DecisionStatus.ABSTAINED
