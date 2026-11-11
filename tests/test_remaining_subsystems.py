"""Executable contract tests for ADA infrastructure subsystems 15-27."""

from __future__ import annotations

import hashlib
import importlib

import pytest

from ada import systems


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_safety_policy_approval_audit_and_tool_gateway_integration():
    safety = systems.safety
    actions = importlib.import_module("12_ACTION_SYSTEM")
    principal = safety.Principal("research-agent", frozenset({"researcher"}))
    policy = safety.PolicyEngine((
        safety.PolicyRule("tool-approval", "tool.execute", "write/", safety.PolicyEffect.REQUIRE_APPROVAL,
                          frozenset({"researcher"})),
        safety.PolicyRule("explicit-deny", "tool.execute", "write/secret", safety.PolicyEffect.DENY),
        safety.PolicyRule("read-allow", "tool.execute", "read/", safety.PolicyEffect.ALLOW),
    ))
    assert policy.evaluate(principal, "tool.execute", "unknown/tool").status is safety.AuthorizationStatus.DENIED
    assert policy.evaluate(principal, "tool.execute", "write/secret").status is safety.AuthorizationStatus.DENIED

    gateway = actions.ToolGateway()
    calls = []
    gateway.register(actions.ToolSpec(
        "write/report", "1", lambda args: calls.append(args["value"]) or "stored",
        frozenset({"research-agent"}), required_arguments=frozenset({"value"}),
        risk=actions.RiskLevel.HIGH, requires_approval=True,
    ))
    authority = safety.ApprovalAuthority(b"s" * 32, frozenset({"human-reviewer"}))
    audit = safety.AuditLedger()
    controller = safety.GovernedActionController(
        policy=policy, gateway=gateway, audit=audit, approval_authority=authority,
    )
    args = {"value": "report"}
    waiting = controller.dispatch("write/report", args, principal=principal, call_id="call-a")
    assert waiting.action_record is None
    ticket = authority.issue(
        principal_id=principal.principal_id, approver_id="human-reviewer",
        action="tool.execute", resource="write/report", arguments=args,
        now=100.0,
    )
    completed = controller.dispatch(
        "write/report", args, principal=principal, call_id="call-a",
        approval=ticket, now=101.0,
    )
    assert completed.action_record.status is actions.ActionStatus.SUCCEEDED
    assert calls == ["report"]
    assert audit.verify()
    replay = controller.dispatch(
        "write/report", args, principal=principal, call_id="call-b",
        approval=ticket, now=102.0,
    )
    assert replay.action_record is None
    assert not authority.consume(ticket, principal_id=principal.principal_id,
                                 action="tool.execute", resource="write/report",
                                 arguments={"value": "tampered"}, now=102.0)


def test_safety_role_permissions_and_artifact_promotion_gates():
    safety = systems.safety
    permissions = safety.PermissionManager(safety.DEFAULT_ROLE_PERMISSIONS)
    researcher = safety.Principal("researcher-1", frozenset({"researcher"}))
    approver = safety.Principal("human-1", frozenset({"release_approver"}))
    assert permissions.has_permission(researcher, "sandbox.run")
    assert not permissions.has_permission(researcher, "release.approve")
    promotion = safety.ArtifactPromotionController()
    base, candidate = sha("base"), sha("candidate")
    promotion.register_approved("ada-model", base)
    with pytest.raises(PermissionError):
        promotion.promote("ada-model", candidate, {"integrity": True}, researcher)
    gates = {name: True for name in promotion.REQUIRED_GATES}
    gates["safety"] = False
    with pytest.raises(PermissionError):
        promotion.promote("ada-model", candidate, gates, approver)
    gates["safety"] = True
    assert promotion.promote("ada-model", candidate, gates, approver) == candidate
    assert promotion.rollback("ada-model", expected_current=candidate) == base


def test_data_pipeline_deduplication_license_and_quality_gate():
    data = systems.data
    result = data.DataPipeline().prepare(
        ({"text": "a", "label": 1}, {"text": "a", "label": 1}, {"text": "missing"}),
        dataset_id="ada-small", version="v1", split=data.DatasetSplit.TRAIN,
        license_id="CC-BY-4.0", source_id="curated-source",
        required_fields=frozenset({"text", "label"}),
    )
    assert len(result.records) == 1
    assert result.quality.duplicate_count == 1
    assert result.quality.rejected_count == 1
    assert not data.DataQualityGate(data.DataQualityPolicy(
        minimum_accepted=1, maximum_rejection_rate=0.2,
    )).evaluate(result.quality)
    accepted = data.DataPipeline().prepare(
        ({"text": "b", "label": 0}, {"text": "c", "label": 1}),
        dataset_id="ada-small", version="v2", split=data.DatasetSplit.VALIDATION,
        license_id="Apache-2.0", source_id="curated-source",
        required_fields=frozenset({"text", "label"}),
    )
    gate = data.DataQualityGate(data.DataQualityPolicy(minimum_accepted=2))
    assert gate.evaluate(accepted.quality)
    registry = data.DatasetRegistry()
    registry.register(accepted.manifest)
    assert registry.get("ada-small", "v2") == accepted.manifest


def test_training_run_state_machine_and_checkpoint_integrity():
    training = systems.training
    config = training.TrainingConfig(
        "ada-preview", sha("dataset"), seed=42, max_steps=2,
        learning_rate=1e-4, batch_size=2, precision="bf16",
    )
    coordinator = training.TrainingCoordinator()
    run = coordinator.create(config)
    coordinator.transition(run.run_id, training.RunStatus.RUNNING)
    updated = coordinator.record_step(run.run_id, 1, {"loss": 1.2})
    assert updated.step == 1
    coordinator.transition(run.run_id, training.RunStatus.PAUSED)
    with pytest.raises(ValueError):
        coordinator.record_step(run.run_id, 2, {"loss": 0.9})
    coordinator.transition(run.run_id, training.RunStatus.RUNNING)
    coordinator.record_step(run.run_id, 2, {"loss": 0.9})
    completed = coordinator.transition(run.run_id, training.RunStatus.COMPLETED)
    assert completed.step == 2
    with pytest.raises(ValueError):
        coordinator.transition(run.run_id, training.RunStatus.RUNNING)
    checkpoint = training.CheckpointManifest(
        "ckpt-1", "ada-preview", sha("weights"), sha("config"),
        sha("dataset"), 2,
    )
    registry = training.CheckpointRegistry()
    registry.register(checkpoint)
    assert registry.get(checkpoint.artifact_digest) == checkpoint


def test_inference_manifest_contract_and_no_implicit_fallback():
    inference = systems.inference
    manifest = inference.ModelManifest(
        "ada-preview", "0.1.0", sha("artifact"), "tok-v1",
        frozenset({"text"}), 128, 32, "bf16", "kv-v1",
    )
    registry = inference.ModelRegistry()
    registry.register(manifest, lambda payload, limit: {"echo": payload, "limit": limit})
    engine = inference.InferenceEngine(registry)
    result = engine.generate(inference.InferenceRequest(
        "ada-preview", {"prompt": "hello", "token_count": 10}, max_output_tokens=12,
    ))
    assert result.output["echo"]["prompt"] == "hello"
    with pytest.raises(ValueError, match="modality"):
        engine.generate(inference.InferenceRequest("ada-preview", "x", modality="audio"))
    with pytest.raises(KeyError):
        engine.generate(inference.InferenceRequest("missing-model", "x"))


def test_interaction_history_idempotency_and_feedback():
    interaction = systems.interaction
    store = interaction.ConversationStore(max_messages=3)
    first = store.append("c-1", interaction.MessageRole.USER, "hello", turn_id="turn-1")
    assert store.append("c-1", interaction.MessageRole.USER, "hello", turn_id="turn-1") == first
    with pytest.raises(ValueError):
        store.append("c-1", interaction.MessageRole.USER, "changed", turn_id="turn-1")
    reply = store.append("c-1", interaction.MessageRole.ASSISTANT, "hello back")
    feedback = interaction.FeedbackRecord(
        "c-1", reply.message_id, interaction.FeedbackLabel.HELPFUL,
    )
    feedback_store = interaction.FeedbackStore()
    feedback_store.add(feedback)
    assert len(store.get("c-1")) == 2
    assert feedback_store.records == (feedback,)


def test_distributed_scheduler_enforces_live_node_capacity_and_idempotency():
    distributed = systems.distributed
    registry = distributed.NodeRegistry(heartbeat_timeout=10)
    registry.register(distributed.NodeDescriptor(
        "node-a", frozenset({"cpu-inference"}), total_slots=1, last_heartbeat=100,
    ))
    registry.register(distributed.NodeDescriptor(
        "node-b", frozenset({"cpu-inference"}), total_slots=1, last_heartbeat=100,
    ))
    scheduler = distributed.WorkloadScheduler(registry)
    first = scheduler.lease("task-a", "cpu-inference", now=101)
    assert scheduler.lease("task-a", "cpu-inference", now=102) == first
    second = scheduler.lease("task-b", "cpu-inference", now=102)
    assert first.node_id != second.node_id
    scheduler.release(first.lease_id)
    with pytest.raises(RuntimeError):
        scheduler.lease("task-c", "gpu-inference", now=102)


def test_observability_structured_events_metrics_and_spans():
    observability = systems.observability
    store = observability.ObservabilityStore(capacity=10)
    store.record_event(observability.TelemetryEvent(
        "request.complete", observability.EventLevel.INFO, 1.0, "trace-1",
        {"model": "ada"},
    ))
    store.record_metric(observability.MetricSample("latency", 0.2, 1.0, "seconds"))
    store.record_metric(observability.MetricSample("latency", 0.4, 2.0, "seconds"))
    span = store.start_span("inference", trace_id="trace-1", now=10.0)
    closed = store.end_span(span.span_id, now=10.5)
    assert store.events(trace_id="trace-1")[0].name == "request.complete"
    assert store.metric_summary("latency")["mean"] == pytest.approx(0.3)
    assert closed.ended_at == 10.5


def test_interoperability_allowlist_and_strict_json_schema():
    interop = systems.interoperability
    manifest = interop.PluginManifest(
        "trusted-plugin", "1.0", "1", frozenset({"read"}),
        {"query": "str", "limit": "int"}, {"items": "array"},
    )
    registry = interop.PluginRegistry(frozenset({"trusted-plugin"}), api_version="1")
    registry.register(manifest)
    assert registry.resolve("trusted-plugin", "1.0") == manifest
    with pytest.raises(PermissionError):
        interop.PluginRegistry(frozenset()).register(manifest)
    schema = {"query": "str", "limit": "int"}
    raw = interop.ContractSerializer.encode({"query": "ada", "limit": 5}, schema)
    assert interop.ContractSerializer.decode(raw, schema)["limit"] == 5
    with pytest.raises(TypeError):
        interop.ContractSerializer.validate({"query": "x", "limit": True}, schema)


def test_deployment_release_gate_canary_activation_and_rollback():
    deployment = systems.deployment
    artifact = deployment.ReleaseArtifact(
        "release-1", "ada-preview", "0.1", sha("weights"),
        sha("eval-report"), sha("safety-report"),
    )
    manager = deployment.DeploymentManager()
    with pytest.raises(PermissionError):
        manager.stage(artifact, environment="staging", gate_passed=False, approver_id="human")
    staged = manager.stage(
        artifact, environment="staging", gate_passed=True, approver_id="human",
    )
    manager.transition(staged.deployment_id, deployment.DeploymentStatus.CANARY)
    active = manager.transition(staged.deployment_id, deployment.DeploymentStatus.ACTIVE)
    assert active.status is deployment.DeploymentStatus.ACTIVE
    assert manager.active_digest("ada-preview", "staging") == artifact.artifact_digest
    manager.transition(staged.deployment_id, deployment.DeploymentStatus.ROLLED_BACK)
    assert manager.active_digest("ada-preview", "staging") is None


def test_testing_harness_records_pass_failure_and_runner_error():
    testing = systems.testing
    registry = testing.TestRegistry()
    registry.register("identity", lambda payload: payload["value"])
    registry.register("broken", lambda payload: 1 / 0)
    run = testing.TestHarness(registry).run((
        testing.TestDefinition("pass", "identity", {"value": 2}, 2),
        testing.TestDefinition("fail", "identity", {"value": 2}, 3),
        testing.TestDefinition("error", "broken", {}, 0),
    ))
    assert run.counts == {"passed": 1, "failed": 1, "error": 1}
    assert not run.passed


def test_research_ledger_binds_results_to_protocol_and_dataset_digests():
    research = systems.research
    ledger = research.ResearchLedger()
    hypothesis = research.ResearchHypothesis(
        "hyp-1", "Sparse attention improves throughput", "No throughput change",
        "tokens_per_second", 1.1,
    )
    ledger.add_hypothesis(hypothesis)
    protocol = research.ExperimentProtocol(
        "exp-1", "hyp-1", 42, sha("config"), (sha("train"), sha("eval")),
    )
    ledger.add_protocol(protocol)
    result = research.ExperimentResult(
        "exp-1", ledger.protocol_digest(protocol), sha("code"), 100,
        1.12, research.HypothesisStatus.SUPPORTED,
    )
    assert ledger.add_result(result) == result
    with pytest.raises(ValueError):
        ledger.add_result(result)


def test_documentation_catalog_and_evidence_backed_adr():
    docs = systems.documentation
    catalog = docs.DocumentationCatalog()
    entry = docs.DocumentationEntry(
        "docs/inference.md", "Inference contract", "v1", sha("content"),
        "architecture", ("inference", "api"),
    )
    catalog.add_entry(entry)
    assert catalog.search("inference api") == (entry,)
    adr = docs.ArchitectureDecisionRecord(
        "ADR-001", "Keep model artifacts immutable", "Need rollback-safe releases",
        "Use digest-addressed artifacts", "More storage metadata is required",
        ("Mutable weights",), ("spec://architecture-contracts",),
        docs.DecisionStatus.ACCEPTED,
    )
    catalog.add_adr(adr)
    assert catalog.decisions == (adr,)
    with pytest.raises(ValueError):
        docs.ArchitectureDecisionRecord(
            "ADR-002", "No evidence", "Context", "Decision", "Consequences",
            (), (), docs.DecisionStatus.ACCEPTED,
        )


def test_operations_incident_backup_and_maintenance_lifecycles():
    operations = systems.operations
    incidents = operations.IncidentManager()
    incident = incidents.create("Model integrity alert", operations.IncidentSeverity.CRITICAL)
    incidents.transition(incident.incident_id, operations.IncidentStatus.ACKNOWLEDGED, owner="on-call")
    incidents.transition(incident.incident_id, operations.IncidentStatus.MITIGATED)
    resolved = incidents.transition(incident.incident_id, operations.IncidentStatus.RESOLVED)
    assert resolved.status is operations.IncidentStatus.RESOLVED
    backups = operations.BackupRegistry()
    backup = operations.BackupManifest(
        "backup-1", sha("artifact"), sha("dataset"), 10.0, "vault://backup-1", True,
    )
    backups.register(backup)
    assert backups.latest_verified() == backup
    tasks = operations.MaintenanceManager()
    task = tasks.create("Rotate logs")
    tasks.transition(task.task_id, operations.MaintenanceStatus.RUNNING)
    assert tasks.transition(task.task_id, operations.MaintenanceStatus.COMPLETED).status is operations.MaintenanceStatus.COMPLETED
