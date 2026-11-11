"""Integration and contract tests for all nested ADA packages in folders 15-27."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest
from ada import systems


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_ada_systems_exposes_every_nested_feature_package():
    for root, children in systems.__dict__.get("_CHILDREN", {}).items():
        parent = getattr(systems, root)
        for child in children:
            module = getattr(parent, child)
            assert module.__name__.endswith("." + child)
            assert module.__file__.endswith("__init__.py")


def test_safety_approval_audit_permissions_redaction_and_lockout(tmp_path):
    safety = systems.safety
    roles = safety.PermissionManager({"approver": frozenset({"action.approve"})})
    authority = safety.ApprovalAuthority(b"x" * 32, frozenset({"human-1"}))
    service = safety.action_approval.ActionApprovalService(authority, roles)
    caller = safety.Principal("agent-1")
    human = safety.Principal("human-1", frozenset({"approver"}))
    args = {"record": "report"}
    ticket = service.issue(
        principal=caller, approver=human, action="write", resource="report/1",
        arguments=args, now=100.0,
    )
    assert service.consume(ticket, principal=caller, action="write",
                           resource="report/1", arguments=args, now=101.0)
    assert not service.consume(ticket, principal=caller, action="write",
                               resource="report/1", arguments=args, now=102.0)

    ledger = safety.audit_logs.FileAuditLedger(tmp_path / "audit.jsonl")
    ledger.append("action.allowed", "agent-1", {"resource": "report/1"}, timestamp=1.0)
    ledger.append("action.denied", "agent-1", {"resource": "secret"}, timestamp=2.0)
    assert ledger.verify()
    assert safety.audit_logs.FileAuditLedger(tmp_path / "audit.jsonl").verify()

    redacted = safety.data_protection.redact_text("email vimal@example.org key sk-abcdefghijklmno")
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_SECRET]" in redacted

    monitor = safety.security_monitor.SecurityMonitor(threshold=2, window_seconds=10, lockout_seconds=20)
    event = safety.security_monitor.SecurityEvent("agent-1", "denied", 1.0)
    assert not monitor.record(event, now=1.0)
    assert monitor.record(event, now=2.0)
    assert not monitor.is_locked("agent-1", now=25.0)


def test_safety_policy_scoped_permissions_artifact_integrity_and_sandbox(tmp_path):
    safety = systems.safety
    principal = safety.Principal("researcher", frozenset({"researcher"}))
    roles = safety.PermissionManager({"reader": frozenset({"artifact.read"})})
    scoped = safety.permission_manager.ScopedPermissionManager(roles)
    grant = safety.permission_manager.PermissionGrant("researcher", "candidate.read", "candidate/",
                                                       20.0, "grant-1")
    scoped.grant(grant)
    assert scoped.allows(principal, "candidate.read", "candidate/abc", now=10.0)
    assert not scoped.allows(principal, "candidate.read", "candidate/abc", now=21.0)
    assert scoped.revoke("grant-1")

    bundle = safety.policy_engine.PolicyBundle("baseline", "v1", (
        safety.PolicyRule("allow-read", "read", "docs/", safety.PolicyEffect.ALLOW),
    ))
    assert safety.policy_engine.PolicyBundleEngine(bundle).evaluate(
        principal, "read", "docs/architecture"
    ).status is safety.AuthorizationStatus.ALLOWED

    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"immutable checkpoint")
    digest, size = safety.model_integrity.sha256_file(artifact)
    manifest = safety.model_integrity.ArtifactManifest("a1", "ada", "v1", digest, size, "v4.5")
    assert safety.model_integrity.verify_artifact(artifact, manifest)

    sandbox = safety.sandbox.SandboxRunner()
    profile = safety.sandbox.SandboxProfile("restricted", 2.0, 512, 2)
    with pytest.raises(RuntimeError, match="no isolated sandbox"):
        sandbox.run({"command": "not-run"}, profile)


def test_data_ingestion_schema_versioning_and_quality_gates(tmp_path):
    data = systems.data
    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text('{"text":"alpha","label":1}\n{"text":"beta","label":0}\n', encoding="utf-8")
    records = tuple(data.data_pipeline.JsonlReader(jsonl))
    assert len(records) == 2
    batches = tuple(data.data_pipeline.batches(records, 1))
    assert len(batches) == 2

    validator = data.data_quality.RecordValidator((
        data.data_quality.SchemaField("text", str),
        data.data_quality.SchemaField("label", int),
    ))
    assert validator.validate(records) == ()
    assert validator.validate(({"text": "bad", "label": True},))

    prepared = data.DataPipeline().prepare(
        records, dataset_id="corpus", version="v1", split=data.DatasetSplit.TRAIN,
        license_id="CC-BY-4.0", source_id="publisher",
        required_fields=frozenset({"text", "label"}),
    )
    assert data.DataQualityGate(data.DataQualityPolicy(minimum_accepted=2)).evaluate(prepared.quality)
    graph = data.dataset_versioning.DatasetVersionGraph()
    first = data.dataset_versioning.DatasetVersion(prepared.manifest)
    graph.add(first)
    next_manifest = data.DatasetManifest(
        "corpus", "v2", sha("records-v2"), data.DatasetSplit.TRAIN, 1, "CC-BY-4.0", "publisher"
    )
    graph.add(data.dataset_versioning.DatasetVersion(next_manifest, parent_version="v1"))
    assert [x.manifest.version for x in graph.lineage("corpus", "v2")] == ["v2", "v1"]


def test_data_corpora_packing_synthetic_review_and_split_leakage():
    data = systems.data
    eval_manifest = data.DatasetManifest("eval", "v1", sha("evaluation"), data.DatasetSplit.EVALUATION,
                                         1, "CC-BY-4.0", "source")
    corpus = data.evaluation_corpus.EvaluationCorpus(eval_manifest, ({"prompt": "test"},))
    access = corpus.authorize("evaluator")
    assert corpus.read(access) == ({"prompt": "test"},)
    assert corpus.revoke(access)
    with pytest.raises(PermissionError):
        corpus.read(access)

    packed = data.processed_datasets.pack_examples(
        (data.processed_datasets.TokenizedExample("doc-1", (1, 2, 3)),),
        sequence_length=4, eos_token_id=0,
    )
    assert packed[0].input_ids == (1, 2, 3, 0)
    assert all(packed[0].loss_mask)

    sample = data.synthetic_data.SyntheticSample("syn-1", "sample", "generator", sha("prompt"))
    review = data.synthetic_data.SyntheticReview("syn-1", "reviewer",
                                                 data.synthetic_data.SyntheticStatus.APPROVED, "checked")
    assert data.synthetic_data.eligible_for_training(sample, review)

    validation_manifest = data.DatasetManifest("val", "v1", sha("val"), data.DatasetSplit.VALIDATION,
                                               1, "CC-BY-4.0", "source")
    validation = data.validation_corpus.ValidationCorpus(validation_manifest, ({"x": 1},),
                                                          frozenset({sha("same")}))
    assert data.validation_corpus.check_split_leakage({sha("train"), sha("same")},
                                                       validation.sample_digests) == frozenset({sha("same")})

    train_manifest = data.DatasetManifest("train", "v1", sha("train"), data.DatasetSplit.TRAIN,
                                          1, "CC-BY-4.0", "source")
    train_view = data.training_corpus.TrainingCorpus(train_manifest, ({"x": 1},))
    assert len(train_view.records) == 1


def test_training_checkpoint_optimizer_schedule_monitor_and_budgets(tmp_path):
    training = systems.training
    store = training.checkpointing.AtomicCheckpointStore(tmp_path / "checkpoints")
    checkpoint = store.save_bytes("step-10.bin", b"weights")
    assert store.load_bytes(checkpoint) == b"weights"

    stage = training.continual_training.ContinualStage(
        "stage-1", sha("dataset"), 10, 0.2
    )
    assert training.continual_training.ContinualSchedule((stage,)).stage(0) == stage
    parallel = training.distributed_training.ParallelismConfig("dp_ep", expert_parallel_size=2)
    assert parallel.expert_parallel_size == 2
    schedule = training.optimizer.WarmupCosine(peak_lr=1e-3, min_lr=1e-4,
                                                warmup_steps=2, total_steps=10)
    assert schedule(0) == 0.0
    assert schedule(2) == pytest.approx(1e-3)
    assert schedule(10) == pytest.approx(1e-4)

    budget = training.pretraining.TokenBudget(100)
    assert budget.add(40) == 60
    assert not budget.complete
    monitor = training.training_monitor.TrainingMonitor(window=4)
    assert monitor.record(2.0, [1, 1]).finite
    assert monitor.record(float("nan")).finite is False


def test_interaction_history_feedback_multimodal_and_user_context():
    interaction = systems.interaction
    store = interaction.ConversationStore()
    user = store.append("conv-1", interaction.MessageRole.USER, "hello", turn_id="turn-1")
    assistant = store.append("conv-1", interaction.MessageRole.ASSISTANT, "hi", turn_id="turn-1")
    turn = interaction.conversation.assemble_turn("conv-1", "turn-1", (user, assistant))
    assert turn.final_message == assistant
    window = interaction.interaction_history.build_history_window(
        store.get("conv-1"), max_tokens=10
    )
    assert len(window.messages) == 2

    feedback = interaction.FeedbackRecord("conv-1", assistant.message_id,
                                          interaction.FeedbackLabel.HELPFUL)
    queue = interaction.feedback_interface.FeedbackReviewQueue()
    queue.submit(feedback)
    assert queue.training_label(feedback.feedback_id) is None
    queue.review(interaction.feedback_interface.FeedbackReview(
        feedback.feedback_id, "reviewer", interaction.feedback_interface.FeedbackDisposition.ACCEPTED,
        "verified",
    ))
    assert queue.training_label(feedback.feedback_id) is interaction.FeedbackLabel.HELPFUL

    payload = interaction.multimodal_interface.describe_payload(
        interaction.multimodal_interface.Modality.IMAGE, b"image", media_type="image/png", source_id="upload"
    )
    assert payload.byte_length == 5

    context = interaction.user_context.UserContextStore()
    context.set("user-1", interaction.user_context.ContextValue("theme", "dark", 20.0, True))
    assert context.get("user-1", "theme", now=10.0).value == "dark"
    assert context.get("user-1", "theme", now=21.0) is None


def test_distributed_node_lifecycle_storage_transport_and_fault_tolerance(tmp_path):
    distributed = systems.distributed
    managed = distributed.node_manager.ManagedNodeRegistry(distributed.NodeRegistry(heartbeat_timeout=10))
    node = distributed.NodeDescriptor("n1", frozenset({"cpu"}), 2, last_heartbeat=100.0)
    managed.register(node)
    managed.set_enabled("n1", False)
    assert managed.available("cpu", now=101.0) == ()
    managed.set_enabled("n1", True)
    snap = distributed.cluster_manager.snapshot(managed.registry, now=101.0)
    assert snap.live_nodes == 1 and snap.total_slots == 2

    devices = distributed.accelerator_manager.discover_torch_devices()
    assert devices and devices[0].backend == "cpu"

    breaker = distributed.fault_tolerance.CircuitBreaker(failure_threshold=1, reset_after=2)
    breaker.failure(now=1.0)
    assert not breaker.allow(now=2.0)
    assert breaker.allow(now=4.0)

    envelope = distributed.network_transport.TransportEnvelope.create(
        "ada", "worker", {"task": 1}, idempotency_key="task-1"
    )
    transport = distributed.network_transport.LoopbackTransport()
    transport.send(envelope)
    assert transport.receive() == envelope

    objects = distributed.storage_manager.ContentAddressedStore(tmp_path / "objects")
    object_id = objects.put(b"artifact")
    assert objects.get(object_id) == b"artifact"

    fences = distributed.synchronization.FencingTokenRegistry()
    first = fences.issue("model")
    second = fences.issue("model")
    assert not fences.valid("model", first) and fences.valid("model", second)
    barrier = distributed.synchronization.VersionBarrier(1)
    assert barrier.arrive("worker-1", version=0) == 1


def test_workload_distribution_and_node_capacity():
    distributed = systems.distributed
    registry = distributed.NodeRegistry()
    registry.register(distributed.NodeDescriptor("n1", frozenset({"inference"}), 1, last_heartbeat=100))
    registry.register(distributed.NodeDescriptor("n2", frozenset({"inference"}), 1, last_heartbeat=100))
    distributor = distributed.workload_distribution.WorkloadDistributor(
        distributed.WorkloadScheduler(registry)
    )
    leases = distributor.assign((
        distributed.workload_distribution.WorkRequest("b", "inference", 1),
        distributed.workload_distribution.WorkRequest("a", "inference", 2),
    ), now=101.0)
    assert len(leases) == 2
    assert {x.task_id for x in leases} == {"a", "b"}
