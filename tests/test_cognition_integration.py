from __future__ import annotations

import pytest

from ada.cognition import (
    CognitionEngine, CognitiveTask, EvidenceItem, analyze_cognition,
)


def _result():
    task = CognitiveTask(objective="inspect cooling system", candidate_actions=("inspect cooling system",))
    evidence = EvidenceItem(
        evidence_id="evidence-1",
        source_kind="test",
        source_id="inspection-1",
        source_digest="a" * 64,
        content="cooling system pressure inspection pump",
    )
    from ada.cognition import CognitiveResult, CognitiveStatus, CognitiveDecision, DecisionStatus
    return CognitiveResult(
        task_id=task.task_id,
        status=CognitiveStatus.READY,
        evidence=(evidence,),
        plan=(),
        decision=CognitiveDecision(
            DecisionStatus.SELECTED, "inspect cooling system", 0.9,
            "Evidence supports inspection.",
            (("inspect cooling system", 0.9), ("replace unrelated part", 0.1)),
        ),
    )


def test_integrated_pipeline_runs_all_controller_stages_and_audits():
    report = analyze_cognition(
        _result(),
        "cooling system",
        concepts=(("cooling system", ("evidence-1",), ("mechanical",)),),
        relations=(("pump", "supports", "cooling-system", "evidence-1"),),
        causal_edges=(("low_flow", "overheating"), ("overheating", "shutdown")),
        causal_source="low_flow",
        logical_premises=((True, True), (False, True)),
        arithmetic=((6, "*", 7), (10, "/", 2)),
        hypotheses=(("pump restriction", ("evidence-1",), ("flow measurement",)),),
    )
    assert len(report.stage_trace) == 10
    assert report.focus[0][2][0] == "evidence-1"
    assert report.abstractions[0][0] == "cooling"
    assert report.concepts[0].evidence_ids == ("evidence-1",)
    assert report.relations[0].evidence_id == "evidence-1"
    assert report.causal_reachable == ("overheating", "shutdown")
    assert report.logical_results == (True, None)
    assert report.calculations == (42, 5)
    assert report.hypotheses[0].statement == "pump restriction"
    assert report.normalized_uncertainty is not None
    assert report.audit.evidence_count == 1
    assert report.audit.decision_abstained is False


def test_integrated_pipeline_does_not_invent_optional_outputs():
    report = analyze_cognition(_result(), "cooling system")
    assert report.concepts == ()
    assert report.relations == ()
    assert report.causal_reachable == ()
    assert report.logical_results == ()
    assert report.calculations == ()
    assert report.hypotheses == ()


def test_causal_edges_require_explicit_source():
    with pytest.raises(ValueError, match="causal_source"):
        analyze_cognition(_result(), "cooling system", causal_edges=(("a", "b"),))


def test_pipeline_rejects_empty_query():
    with pytest.raises(ValueError, match="query"):
        analyze_cognition(_result(), " ")
