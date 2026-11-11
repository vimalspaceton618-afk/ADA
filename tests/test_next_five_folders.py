"""Contract and integration tests for ADA's next five subsystems."""
from __future__ import annotations

import hashlib
from importlib import import_module

import pytest

dev = import_module("10_DEVELOPMENT_ENGINE")
plan = import_module("11_REASONING_AND_PLANNING")
actions = import_module("12_ACTION_SYSTEM")
knowledge = import_module("13_EXTERNAL_KNOWLEDGE")
evaluation = import_module("14_EVALUATION")


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_development_candidate_validation_promotion_and_rollback():
    engine = dev.DevelopmentEngine(validator=lambda proposal: {"status": "pass", "tests": 3})
    engine.register_module(dev.ModuleDescriptor("planner", "1.0.0", sha("base")))
    proposal = dev.ChangeProposal(
        "planner", sha("base"),
        ({"op": "set", "path": "limits.max_steps", "value": 12},),
        "Constrain planner expansion", "researcher",
    )
    candidate = engine.propose(proposal)
    assert candidate.status is dev.CandidateStatus.PROPOSED
    approval = dev.PromotionApproval(candidate.candidate_digest, "owner", "Reviewed candidate")
    with pytest.raises(ValueError):
        engine.promote(candidate.candidate_digest, approval)
    assert engine.validate(candidate.candidate_digest).status is dev.CandidateStatus.VALIDATED
    assert engine.promote(candidate.candidate_digest, approval).status is dev.CandidateStatus.PROMOTED
    assert engine.active_digest("planner") == candidate.candidate_digest
    with pytest.raises(ValueError):
        engine.promote(candidate.candidate_digest, approval)
    assert engine.rollback("planner", expected_current=candidate.candidate_digest) == sha("base")


def test_development_rejects_failed_validation_and_stale_parent():
    engine = dev.DevelopmentEngine(validator=lambda proposal: {"status": "inconclusive"})
    engine.register_module(dev.ModuleDescriptor("memory", "1", sha("memory")))
    proposal = dev.ChangeProposal("memory", sha("memory"),
        ({"op": "set", "path": "capacity", "value": 10},), "Try memory capacity", "agent")
    record = engine.propose(proposal)
    assert engine.validate(record.candidate_digest).status is dev.CandidateStatus.REJECTED
    bad = dev.ChangeProposal("memory", sha("old"), ({"op": "remove", "path": "unused"},), "Stale", "agent")
    with pytest.raises(ValueError):
        engine.propose(bad)


def test_development_dependency_order_and_closed_operation_language():
    engine = dev.DevelopmentEngine()
    engine.register_module(dev.ModuleDescriptor("base", "1", sha("base")))
    engine.register_module(dev.ModuleDescriptor("consumer", "1", sha("consumer"), dependencies=("base",)))
    assert engine.dependency_order(["consumer"]) == ("base", "consumer")
    with pytest.raises(ValueError):
        dev.ChangeProposal("consumer", sha("consumer"), ({"op": "exec", "path": "python", "value": "bad"},), "No code", "agent")


def test_planner_validates_capabilities_dependencies_and_cycles():
    verifier = plan.PlanVerifier()
    candidate = plan.Plan("complete task", (
        plan.PlanStep("inspect", "Inspect inputs", "read"),
        plan.PlanStep("test", "Run tests", "pytest", depends_on=("inspect",)),
    ), max_steps=4)
    good = verifier.verify(candidate, {"read", "pytest"})
    assert good.accepted and good.order == ("inspect", "test")
    missing = verifier.verify(candidate, {"read"})
    assert not missing.accepted and missing.unresolved_capabilities == ("pytest",)
    cycle = plan.Plan("cycle", (
        plan.PlanStep("a", "A", "x", depends_on=("b",)),
        plan.PlanStep("b", "B", "x", depends_on=("a",)),
    ), max_steps=2)
    assert not verifier.verify(cycle, {"x"}).accepted


def test_goal_manager_and_decision_engine_abstention():
    manager = plan.GoalManager()
    goal_id = manager.add("Explicit objective")
    assert manager.get(goal_id) == "Explicit objective"
    selector = plan.DecisionEngine()
    assert selector.choose({"alpha": 0.7, "beta": 0.7}) == "beta"
    assert selector.choose({"alpha": 0.2}, min_score=0.5) is None
    with pytest.raises(ValueError):
        selector.choose({"alpha": float("nan")})


def test_action_gateway_resumes_after_approval_and_prevents_replay():
    gateway = actions.ToolGateway()
    calls = []
    gateway.register(actions.ToolSpec(
        "echo", "1.0", lambda args: calls.append(args["value"]) or {"echo": args["value"]},
        frozenset({"agent"}), required_arguments=frozenset({"value"}),
        risk=actions.RiskLevel.MEDIUM, requires_approval=True,
    ))
    args = {"value": "approved"}
    waiting = gateway.execute("echo", args, principal="agent", call_id="call-1")
    assert waiting.status is actions.ActionStatus.AWAITING_APPROVAL
    ticket = gateway.issue_approval(call_id="call-1", tool_id="echo", principal="agent", arguments=args, approver="human")
    done = gateway.execute("echo", args, principal="agent", call_id="call-1", approval=ticket)
    assert done.status is actions.ActionStatus.SUCCEEDED
    assert calls == ["approved"]
    with pytest.raises(ValueError):
        gateway.execute("echo", args, principal="agent", call_id="call-1", approval=ticket)
    replay_ticket = gateway.issue_approval(call_id="call-2", tool_id="echo", principal="agent", arguments=args, approver="human")
    gateway.execute("echo", args, principal="agent", call_id="call-2", approval=replay_ticket)
    replay = gateway.execute("echo", args, principal="agent", call_id="call-3", approval=replay_ticket)
    assert replay.status is actions.ActionStatus.DENIED


def test_action_gateway_denies_unknown_principal_and_bad_schema():
    gateway = actions.ToolGateway()
    assert gateway.execute("missing", {}, principal="agent").status is actions.ActionStatus.DENIED
    gateway.register(actions.ToolSpec("read", "1", lambda args: "ok", frozenset({"operator"}),
                                      optional_arguments=frozenset({"query"}), requires_approval=False))
    assert gateway.execute("read", {}, principal="outsider").status is actions.ActionStatus.DENIED
    assert gateway.execute("read", {"unexpected": True}, principal="operator").status is actions.ActionStatus.DENIED


def test_knowledge_is_untrusted_until_verified_and_license_is_allowlisted():
    content = "Mamba state space models process sequences with recurrent state."
    source = knowledge.SourceRecord("paper-1", "https://example.org/paper", "Research Publisher",
                                    content_digest=sha(content), license_id="CC-BY-4.0")
    document = knowledge.KnowledgeDocument(source, content, "Mamba paper")
    store = knowledge.KnowledgeStore()
    store.ingest(document)
    with pytest.raises(PermissionError):
        store.chunk(document.document_id, size=24, overlap=4)
    verified = store.verify(document.document_id, knowledge.VerificationResult(
        knowledge.KnowledgeStatus.VERIFIED, "reviewer", "publisher and digest checked",
        document_digest=document.digest))
    assert verified.status is knowledge.KnowledgeStatus.VERIFIED
    chunks = store.chunk(document.document_id, size=24, overlap=4)
    assert len(chunks) > 1
    assert store.search("Mamba recurrent", limit=1)
    assert knowledge.LicensePolicy({"CC-BY-4.0"}).check(source)
    assert not knowledge.LicensePolicy().check(source)


def test_knowledge_digest_mismatch_and_terminal_state_are_rejected():
    source = knowledge.SourceRecord("s", "https://example.org", "publisher", content_digest=sha("original"))
    with pytest.raises(ValueError):
        knowledge.KnowledgeDocument(source, "changed", "Changed")
    store = knowledge.KnowledgeStore()
    doc = knowledge.KnowledgeDocument(knowledge.SourceRecord("s2", "https://example.org/2", "publisher"), "text body", "Title")
    store.ingest(doc)
    store.verify(doc.document_id, knowledge.VerificationResult(
        knowledge.KnowledgeStatus.REJECTED, "reviewer", "not reliable",
        document_digest=doc.digest))
    with pytest.raises(PermissionError):
        store.chunk(doc.document_id)


def test_evaluation_gate_pass_fail_and_inconclusive():
    subject, dataset_digest = sha("model"), sha("dataset")
    policy = evaluation.GatePolicy(
        minimum_samples=3, required_metrics=frozenset({"accuracy", "latency"}),
        thresholds={"accuracy": ("min", 0.9), "latency": ("max", 1.0)},
    )
    gate = evaluation.EvaluationGate(policy)
    incomplete = evaluation.EvaluationReport(subject, dataset_digest, "eval",
        (evaluation.Metric("accuracy", 0.95),), 4, evaluation.Verdict.PASS)
    assert gate.evaluate(incomplete).verdict is evaluation.Verdict.INCONCLUSIVE
    too_small = evaluation.EvaluationReport(subject, dataset_digest, "eval",
        (evaluation.Metric("accuracy", 0.95), evaluation.Metric("latency", 0.2)), 2, evaluation.Verdict.PASS)
    assert gate.evaluate(too_small).verdict is evaluation.Verdict.INCONCLUSIVE
    failing = evaluation.EvaluationReport(subject, dataset_digest, "eval",
        (evaluation.Metric("accuracy", 0.85), evaluation.Metric("latency", 0.2)), 4, evaluation.Verdict.PASS)
    assert gate.evaluate(failing).verdict is evaluation.Verdict.FAIL
    passing = evaluation.EvaluationReport(subject, dataset_digest, "eval",
        (evaluation.Metric("accuracy", 0.95), evaluation.Metric("latency", 0.2, "lower")), 4, evaluation.Verdict.PASS)
    assert gate.evaluate(passing).verdict is evaluation.Verdict.PASS


def test_evaluator_registry_checks_dataset_and_subject_provenance():
    subject = sha("model")
    dataset = evaluation.EvaluationDataset("small", (
        evaluation.EvaluationCase("case-1", {"x": 1}, {"y": 2}),
    ), "v1")
    report = evaluation.EvaluationReport(subject, dataset.digest, "fixture",
        (evaluation.Metric("score", 1.0),), 1, evaluation.Verdict.PASS)
    registry = evaluation.EvaluatorRegistry()
    registry.register("fixture", lambda ds, artifact: report)
    assert registry.evaluate("fixture", dataset, subject) is report
    with pytest.raises(ValueError):
        registry.evaluate("fixture", dataset, sha("different-model"))


def test_evaluation_suite_preserves_missing_evaluator_as_inconclusive():
    subject = sha("model")
    dataset = evaluation.EvaluationDataset("small", (
        evaluation.EvaluationCase("case-1", {"x": 1}, {"y": 2}),
    ), "v1")
    suite = evaluation.EvaluationSuite()
    suite.add("regression", evaluation.EvaluatorRegistry())
    reports = suite.run([("regression", "missing-evaluator", dataset, subject)])
    assert reports[0].verdict is evaluation.Verdict.INCONCLUSIVE
    assert suite.reports == reports
