"""Tests for observability, interoperability, deployment, research, docs and operations."""

from __future__ import annotations

import hashlib
import logging
import pytest

from ada import systems


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_observability_diagnostics_logging_metrics_and_performance():
    obs = systems.observability
    runner = obs.diagnostics.DiagnosticRunner()
    runner.register("ok", lambda: True)
    runner.register("bad", lambda: False)
    report = runner.run()
    assert [x.passed for x in report] == [False, True] or {x.check_id: x.passed for x in report} == {"bad": False, "ok": True}

    registry = obs.metrics.MetricRegistry()
    registry.inc("requests")
    registry.gauge("queue", 2)
    registry.observe("latency", 0.2)
    registry.observe("latency", 0.4)
    assert registry.snapshot()["counters"]["requests"] == 1.0
    assert obs.performance_analysis.latency_report((0.1, 0.2, 0.3))["p95"] > 0.2

    logger = logging.getLogger("ada-test-json")
    formatter = obs.logging.JsonEventFormatter()
    record = logging.LogRecord("ada-test-json", logging.INFO, __file__, 1, "hello", (), None)
    record.event_fields = {"subsystem": "inference"}
    formatted = formatter.format(record)
    assert '"message": "hello"' in formatted and '"subsystem": "inference"' in formatted


def test_observability_trace_context_and_telemetry_export():
    obs = systems.observability
    store = obs.ObservabilityStore()
    class Exporter:
        def __init__(self): self.payload = None
        def export(self, events, metrics, spans): self.payload = (events, metrics, spans)
    exporter = Exporter()
    store.record_event(obs.TelemetryEvent("start", obs.EventLevel.INFO, 1.0, "trace-1"))
    store.record_metric(obs.MetricSample("tokens", 7, 1.0))
    with obs.tracing.span(store, "outer"):
        trace_id, span_id = obs.tracing.current_trace()
        assert trace_id and span_id
        with obs.tracing.span(store, "inner"):
            assert obs.tracing.current_trace()[0] == trace_id
    flushed = obs.telemetry.TelemetryFlush(store, exporter).flush()
    assert flushed == {"events": 1, "metrics": 1, "spans": 2}
    assert len(exporter.payload[2]) == 2


def test_interoperability_contract_plugins_protocol_and_public_inference_api():
    interop = systems.interoperability
    producer = interop.module_contracts.ModuleContract(
        "producer", "1", {"seed": "str"}, {"text": "str"}, frozenset({"emit"}),
    )
    consumer = interop.module_contracts.ModuleContract(
        "consumer", "1", {"text": "str"}, {"tokens": "array"}, frozenset({"consume"}),
    )
    assert interop.module_contracts.compatible(producer, consumer)

    manifest = interop.PluginManifest("echo", "1.0", "1", frozenset({"read"}),
                                      {"value": "str"}, {"value": "str"})
    registry = interop.PluginRegistry(frozenset({"echo"}), api_version="1")
    registry.register(manifest)
    class Echo:
        def invoke(self, operation, arguments): return arguments["value"]
    host = interop.plugin_interfaces.PluginHost(registry)
    host.bind(manifest, Echo())
    assert host.invoke("echo", "1.0", "read", {"value": "ADA"}) == "ADA"

    envelope = interop.protocol_definitions.ProtocolEnvelope.build(
        "ada.rpc", "1", "trace-1", {"ok": True}, "idem-1"
    )
    assert len(envelope.payload_digest) == 64
    codec = interop.serialization.StrictJsonCodec()
    raw = codec.encode({"text": "x", "count": 3}, {"text": "str", "count": "int"})
    assert codec.decode(raw, {"text": "str", "count": "int"})["count"] == 3
    with pytest.raises(ValueError, match="duplicate"):
        codec.decode(b'{"text":"a","text":"b","count":3}', {"text": "str", "count": "int"})

    infer = systems.inference
    model_registry = infer.ModelRegistry()
    model_registry.register(
        infer.ModelManifest("m", "1", sha("weights"), "tok1", frozenset({"text"}), 32, 8, "fp32", "cache1"),
        lambda payload, limit: {"output": payload["prompt"], "limit": limit},
    )
    engine = infer.InferenceEngine(model_registry)
    principal = systems.safety.Principal("user", frozenset({"user"}))
    policy = systems.safety.PolicyEngine((
        systems.safety.PolicyRule("infer", "model.infer", "m", systems.safety.PolicyEffect.ALLOW,
                                  frozenset({"user"})),
    ))
    api = interop.public_api.ADAInferenceAPI(engine, policy)
    result = api.generate(infer.InferenceRequest("m", {"prompt": "hello"}, max_output_tokens=2),
                          principal=principal)
    assert result.output["output"] == "hello"


def test_deployment_specs_gates_and_local_provider_adapter():
    deployment = systems.deployment
    config = deployment.configuration.from_mapping({
        "environment": "staging", "replicas": 2, "max_batch_tokens": 4096,
        "max_output_tokens": 1024, "request_timeout_seconds": 15.0,
    })
    assert config.replicas == 2
    plan = deployment.cluster.ClusterDeploymentPlan("d1", sha("artifact"), 2, "cpu")
    assert deployment.cluster.validate_cluster_plan(plan, {"n1": 1, "n2": 1})["accepted"]

    target = deployment.cloud.CloudTarget("dry-run", "local", "resource-1")
    assert deployment.cloud.DryRunCloudProvider().deploy(sha("artifact"), target, {})["status"] == "dry_run"
    args = deployment.containerization.docker_run_args(
        deployment.containerization.ContainerSpec("ada:test", ("python", "-m", "ada"))
    )
    assert "--network" in args and "none" in args

    class Local:
        def start(self, path, cfg): return deployment.local.LocalRuntimeStatus("running", "m", 123)
        def stop(self): return deployment.local.LocalRuntimeStatus("stopped", "m", None)
        def health(self): return deployment.local.LocalRuntimeStatus("healthy", "m", 123)
    controller = deployment.local.LocalDeploymentController(Local())
    assert controller.start("model.pt", config).state == "running"
    assert controller.health().state == "healthy"

    release = deployment.release_management.Release(
        "r1", "1.0", sha("weights"), deployment.release_management.ReleaseChannel.CANDIDATE,
        sha("eval"), sha("safety"),
    )
    registry = deployment.release_management.ReleaseRegistry()
    registry.register(release)
    with pytest.raises(PermissionError):
        registry.promote("r1", deployment.release_management.ReleaseChannel.STABLE, gates_passed=False)
    assert registry.promote("r1", deployment.release_management.ReleaseChannel.STABLE,
                            gates_passed=True).channel is deployment.release_management.ReleaseChannel.STABLE


def test_testing_framework_reliability_reproducibility_safety_and_stress():
    testing = systems.testing
    checks = (
        testing.integration.IntegrationCheck("c1", "memory", "cognition", lambda: True),
        testing.integration.IntegrationCheck("c2", "cognition", "learning", lambda: False),
    )
    report = testing.integration.run_integration_checks(checks)
    assert report.passed == ("c1",) and not report.success
    reliability = testing.reliability.reliability_report(95, 5)
    assert reliability.success_rate == 0.95 and reliability.wilson_lower_95 < 0.95
    manifest = testing.reproducibility.capture(42, {"lr": 0.001}, env_keys=("CUDA_VISIBLE_DEVICES",))
    assert len(manifest.config_digest) == 64
    assert testing.safety.default_deny_probe()
    probes = testing.safety.run_safety_probes((
        testing.safety.SafetyProbe("ok", lambda: True),
        testing.safety.SafetyProbe("raised", lambda: (_ for _ in ()).throw(RuntimeError("probe")), True),
    ))
    assert probes == (("ok", True), ("raised", False))
    stress = testing.stress.run_concurrent(lambda n: n*n, tasks=12, workers=3)
    assert stress.successes == 12 and not stress.failures
    assert testing.unit.assert_finite(1.0)
    loaded, failures = testing.system.import_all_domains()
    assert "27_OPERATIONS" in loaded and not failures


def test_research_protocol_ablation_literature_and_publication_records():
    research = systems.research
    assert len(research.ablation_studies.full_factorial(42)) == 32
    candidates = tuple(research.architecture_experiments.candidates())
    assert candidates and any(x.mod_capacity == 0.5 and x.sparse_keep == 0.75 for x in candidates)

    hypothesis = research.hypotheses.Hypothesis(
        "h1", "MoD reduces measured compute", "No compute reduction",
        "flops_per_token", 0.8,
    )
    assert hypothesis.status is research.hypotheses.ReviewStatus.PROPOSED

    metric = research.experiment_results.ExperimentMetric("loss", 1.25)
    result = research.experiment_results.ExperimentResultRecord(
        "exp-1", sha("protocol"), sha("code"), (sha("dataset"),), (metric,), ("small-scale only",)
    )
    results = research.experiment_results.ExperimentResultStore()
    results.add(result)
    assert results.get("exp-1").digest == result.digest
    assert research.mathematical_models.chinchilla_tokens(1_000_000_000) == 20_000_000_000
    assert research.mathematical_models.active_moe_parameters(100, 64, 6) == 600

    literature = research.literature_review.LiteratureCatalog()
    literature.add_reference(research.literature_review.LiteratureReference(
        "mamba2", "Transformers are SSMs", ("Dao", "Gu"), 2024, "arXiv:2405.21060"
    ))
    literature.add_claim(research.literature_review.LiteratureClaim(
        "claim-1", "SSD is an SSM formulation", ("mamba2",), "supported"
    ))
    publication = research.research_publications.PublicationArtifact(
        "ada-v4.5", "ADA v4.5", "abc1234", "MIT", "Dataset provenance documented",
        ("No scale training run",), ("eval-report-1",),
    )
    assert research.research_publications.release_ready(publication)


def test_documentation_catalog_guides_architecture_and_adr_lifecycle():
    docs = systems.documentation
    assert docs.api_reference.index_module("15_SAFETY_AND_GOVERNANCE")
    catalog = docs.architecture.ArchitectureCatalog()
    catalog.add(docs.architecture.ArchitectureComponent("memory", "Stores facts", (), ("evidence",)))
    catalog.add(docs.architecture.ArchitectureComponent("cognition", "Reasons over evidence", ("evidence",), ("decision",), ("memory",)))
    assert catalog.dependency_order() == ("memory", "cognition")

    decisions = docs.decisions.DecisionLog()
    adr = docs.decisions.Decision("ADR-1", "Use immutable manifests", "Need reproducibility",
                                  "SHA-256 manifests", (), "More metadata", ("paper://ada",),
                                  docs.decisions.DecisionState.ACCEPTED)
    decisions.add(adr)
    assert decisions.get("ADR-1") == adr

    guides = docs.developer_guides.GuideIndex()
    guide = docs.developer_guides.DeveloperGuide("g1", "Training guide", "docs/training.md", "researcher", ("training", "checkpointing"))
    guides.add(guide)
    assert guides.search("checkpointing") == (guide,)

    notebook = docs.research_notes.ResearchNotebook()
    note = notebook.add("MoD", "Measure selected token effects", "researcher", tags=("routing",))
    assert notebook.search("routing") == (note,)
    spec = docs.subsystem_specs.SubsystemSpec("inference", "1.0", "Generation",
                                               ("tokens",), ("output",), ("timeout",), ("no KV cache backend",))
    assert spec.maturity == "reference"


def test_operations_backup_incident_lifecycle_maintenance_and_updates(tmp_path):
    ops = systems.operations
    source = tmp_path / "model.bin"
    source.write_bytes(b"model data")
    backups = ops.backup_recovery.LocalBackupManager()
    artifact = backups.create(source, tmp_path / "backup", backup_id="backup-1")
    assert backups.verify(artifact)
    restored = tmp_path / "restored.bin"
    backups.restore(artifact, restored)
    assert restored.read_bytes() == source.read_bytes()

    incidents = ops.incident_management.IncidentWorkflow()
    incident = incidents.create("incident-1", "Model integrity signal", "critical", now=1.0)
    incidents.transition("incident-1", ops.incident_management.IncidentPhase.TRIAGED, now=2.0)
    incidents.transition("incident-1", ops.incident_management.IncidentPhase.CONTAINED, now=3.0)
    incidents.transition("incident-1", ops.incident_management.IncidentPhase.REMEDIATED, now=4.0)
    closed = incidents.transition("incident-1", ops.incident_management.IncidentPhase.CLOSED, now=5.0)
    assert closed.phase is ops.incident_management.IncidentPhase.CLOSED

    lifecycle = ops.lifecycle_manager.LifecycleManager()
    lifecycle.create("inference")
    lifecycle.transition("inference", ops.lifecycle_manager.ServiceState.STARTING)
    lifecycle.transition("inference", ops.lifecycle_manager.ServiceState.RUNNING)
    assert lifecycle.get("inference").state is ops.lifecycle_manager.ServiceState.RUNNING

    maintenance = ops.maintenance.MaintenanceRegistry()
    maintenance.schedule("job-1", "rotate logs", scheduled_at=1.0)
    maintenance.transition("job-1", ops.maintenance.MaintenanceState.RUNNING, now=2.0)
    result = maintenance.transition("job-1", ops.maintenance.MaintenanceState.SUCCEEDED, now=3.0)
    assert result.finished_at == 3.0

    updates = ops.system_updates.UpdateManager()
    updates.propose(ops.system_updates.UpdateManifest(
        "update-1", "inference", "1.0", "1.1", sha("payload")
    ))
    updates.transition("update-1", ops.system_updates.UpdateState.VERIFIED)
    applied = updates.transition("update-1", ops.system_updates.UpdateState.APPLIED)
    assert applied.state is ops.system_updates.UpdateState.APPLIED
