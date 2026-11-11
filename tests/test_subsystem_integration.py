"""Cross-subsystem integration regression tests for ADA 05-14."""

from __future__ import annotations

import hashlib
from importlib import import_module

import pytest

from ada.cognition import CognitionEngine, CognitiveTask
from ada.learning import LearningIntegration, from_cognitive_result
from ada.memory import MemoryEngine, MemoryKind, MemorySourceKind
from ada.knowledge import (
    KnowledgeDocument, KnowledgeStatus, KnowledgeStore, SourceRecord,
    VerificationResult, to_cognitive_evidence,
)
from ada.development import (
    CandidateStatus, ChangeProposal, DevelopmentEngine, ModuleDescriptor,
    PromotionApproval, EvaluationBackedValidator,
)
from ada.evaluation import (
    EvaluationCase, EvaluationDataset, EvaluationGate, EvaluationReport,
    EvaluatorRegistry, GatePolicy, Metric, Verdict,
)
from ada.planning import Plan, PlanStep, PlanVerifier
from ada.actions import ActionStatus, RiskLevel, ToolGateway, ToolSpec


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_verified_knowledge_flows_into_cognition_with_chunk_provenance():
    content = "coolant pump pressure dropped below the required operating range"
    source = SourceRecord(
        "maintenance-paper-7",
        "https://example.org/maintenance",
        "Engineering Publisher",
        content_digest=sha(content),
        license_id="CC-BY-4.0",
    )
    store = KnowledgeStore()
    document = store.ingest(KnowledgeDocument(source, content, "Pump pressure note"))
    with pytest.raises(ValueError, match="different document digest"):
        store.verify(
            document.document_id,
            VerificationResult(
                KnowledgeStatus.VERIFIED, "reviewer", "reused decision",
                document_digest=sha("a different document"),
            ),
        )
    verified = store.verify(
        document.document_id,
        VerificationResult(
            KnowledgeStatus.VERIFIED, "reviewer", "publisher and digest checked",
            document_digest=document.digest,
        ),
    )
    assert verified.status is KnowledgeStatus.VERIFIED
    chunks = store.chunk(document.document_id, size=120, overlap=12)
    evidence = to_cognitive_evidence(store, chunks, score=0.8)
    assert evidence
    assert evidence[0].source_kind == "external_knowledge"
    assert evidence[0].source_id == source.source_id
    assert evidence[0].source_digest == chunks[0].digest

    task = CognitiveTask(
        objective="inspect coolant pump pressure",
        query="coolant pump pressure",
        candidate_actions=("inspect coolant pump pressure",),
    )
    result = CognitionEngine().process(
        task, additional_evidence=evidence, retrieve_memory=False
    )
    assert result.evidence[0].evidence_id == evidence[0].evidence_id
    assert result.decision.selected_action == "inspect coolant pump pressure"


def test_unverified_or_forged_knowledge_chunks_cannot_enter_cognition():
    text = "verified records only"
    store = KnowledgeStore()
    doc = store.ingest(KnowledgeDocument(
        SourceRecord("src", "https://example.org/src", "Publisher"),
        text, "Record",
    ))
    fake = import_module("13_EXTERNAL_KNOWLEDGE").KnowledgeChunk(
        doc.document_id, 0, text, sha(text)
    )
    with pytest.raises(PermissionError):
        to_cognitive_evidence(store, (fake,), score=0.5)
    store.verify(doc.document_id, VerificationResult(
        KnowledgeStatus.VERIFIED, "reviewer", "reviewed",
        document_digest=doc.digest,
    ))
    store.chunk(doc.document_id)
    forged = import_module("13_EXTERNAL_KNOWLEDGE").KnowledgeChunk(
        doc.document_id, 0, "forged text", sha("forged text")
    )
    with pytest.raises(ValueError, match="stored chunk set"):
        to_cognitive_evidence(store, (forged,), score=0.5)


def test_memory_cognition_learning_handoff_preserves_source_and_task():
    memory = MemoryEngine()
    original_digest = sha("sensor-log-raw")
    record = memory.remember(
        {"observation": "coolant pump pressure is falling"},
        kind=MemoryKind.EPISODIC,
        source_kind=MemorySourceKind.USER,
        source_id="maintenance-session-42",
        source_digest=original_digest,
    )
    task = CognitiveTask(
        objective="inspect coolant pump pressure",
        query="coolant pump pressure",
        candidate_actions=("inspect coolant pump pressure",),
        task_id="maintenance-task-42",
    )
    result = CognitionEngine(memory_engine=memory).process(task)
    assert result.evidence
    adapted = from_cognitive_result(result, domain="mechanical-diagnostics")
    assert adapted[0].task_id == task.task_id
    assert adapted[0].source_digest == original_digest
    assert adapted[0].source_id == record.memory_id

    learning = LearningIntegration()
    learning.ingest(adapted)
    assert len(learning.replay) == 0
    validated = learning.validate_for_replay(adapted[0].experience_id)
    assert validated.status.value == "validated"
    assert len(learning.replay) == 1


def test_cognition_rejects_conflicting_duplicate_evidence_identity():
    cognition = import_module("06_COGNITION")
    first = cognition.EvidenceItem(
        "same-id", "knowledge", "source-a", sha("a"), "first evidence", 0.4
    )
    second = cognition.EvidenceItem(
        "same-id", "knowledge", "source-b", sha("b"), "different evidence", 0.9
    )
    task = CognitiveTask("inspect evidence", candidate_actions=("inspect",))
    with pytest.raises(ValueError, match="conflicting evidence identity"):
        CognitionEngine().process(
            task, additional_evidence=(first, second), retrieve_memory=False
        )


def _evaluation_fixture(metric_value: float = 0.97):
    dataset = EvaluationDataset(
        "development-regression",
        tuple(
            EvaluationCase(f"case-{index}", {"input": index}, {"output": index})
            for index in range(4)
        ),
        "v1",
    )
    registry = EvaluatorRegistry()

    def evaluate(ds, subject_digest):
        return EvaluationReport(
            subject_digest=subject_digest,
            dataset_digest=ds.digest,
            evaluator_id="architecture-regression",
            metrics=(Metric("correctness", metric_value),),
            sample_count=4,
            verdict=Verdict.PASS,
        )

    registry.register("architecture-regression", evaluate)
    gate = EvaluationGate(GatePolicy(
        minimum_samples=4,
        required_metrics=frozenset({"correctness"}),
        thresholds={"correctness": ("min", 0.9)},
    ))
    return dataset, registry, gate


def test_evaluation_gate_is_used_as_development_promotion_validator():
    dataset, registry, gate = _evaluation_fixture()
    validator = EvaluationBackedValidator(
        registry=registry, evaluator_id="architecture-regression",
        dataset=dataset, gate=gate,
    )
    engine = DevelopmentEngine(validator=validator)
    base_digest = sha("planner-v1")
    engine.register_module(ModuleDescriptor("planner", "1.0.0", base_digest))
    proposal = ChangeProposal(
        "planner", base_digest,
        ({"op": "set", "path": "limits.max_steps", "value": 16},),
        "Increase the bounded planning budget", "owner",
    )
    candidate = engine.propose(proposal)
    assert engine.validate(candidate.candidate_digest).status is CandidateStatus.VALIDATED
    approval = PromotionApproval(candidate.candidate_digest, "owner", "Reviewed evaluation report")
    promoted = engine.promote(candidate.candidate_digest, approval)
    assert promoted.status is CandidateStatus.PROMOTED
    assert promoted.validation_report["subject_digest"] == candidate.candidate_digest
    assert promoted.validation_report["dataset_digest"] == dataset.digest
    assert promoted.validation_report["report_digest"]


def test_evaluation_failure_blocks_development_promotion():
    dataset, registry, gate = _evaluation_fixture(metric_value=0.4)
    validator = EvaluationBackedValidator(
        registry=registry, evaluator_id="architecture-regression",
        dataset=dataset, gate=gate,
    )
    engine = DevelopmentEngine(validator=validator)
    base_digest = sha("memory-v1")
    engine.register_module(ModuleDescriptor("memory", "1.0.0", base_digest))
    proposal = ChangeProposal(
        "memory", base_digest,
        ({"op": "set", "path": "capacity", "value": 2048},),
        "Change memory capacity", "owner",
    )
    candidate = engine.propose(proposal)
    assert engine.validate(candidate.candidate_digest).status is CandidateStatus.REJECTED


def test_only_verified_plans_reach_the_action_gateway():
    verifier = PlanVerifier()
    plan = Plan("inspect a source", (
        PlanStep("step-1", "Read the selected source", "read_source", inputs={"source_id": "source-1"}),
    ), max_steps=2)
    gateway = ToolGateway()
    calls = []
    gateway.register(ToolSpec(
        "read_source", "1.0", lambda args: calls.append(args["source_id"]) or "read",
        frozenset({"agent"}), required_arguments=frozenset({"source_id"}),
        risk=RiskLevel.LOW, requires_approval=False,
    ))

    rejected = verifier.verify(plan, available_capabilities=())
    assert not rejected.accepted
    assert gateway.audit_log == ()
    assert calls == []

    accepted = verifier.verify(plan, available_capabilities={"read_source"})
    assert accepted.accepted
    step = next(step for step in plan.steps if step.step_id == accepted.order[0])
    record = gateway.execute(
        step.capability, dict(step.inputs), principal="agent", call_id="plan-step-1"
    )
    assert record.status is ActionStatus.SUCCEEDED
    assert calls == ["source-1"]
