"""Train native ADA on a synthetic release produced by ADA-Datasets."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("pyarrow")
pytest.importorskip("ada_datasets.training")

from ada.atomic_checkpoint import resolve_committed_checkpoint
from ada_datasets import release_governance
from ada_datasets.cli import main as curate
from ada_datasets.release_governance import (
    initialize_registry, quarantine_release, register_release, verify_registry,
)
from ada_datasets.training import TrainingRelease
from scripts import ada_pretrain_fineweb_edu as trainer


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def curated(tmp_path):
    checkout = os.environ.get("ADA_DATASETS_CHECKOUT")
    source = Path(checkout) if checkout else Path(__file__).resolve().parents[2] / "ADA-Datasets"
    generator = source / "examples" / "make_release_fixture.py"
    if checkout is None and not generator.is_file():
        pytest.skip("ADA-Datasets checkout with the synthetic release generator is required")
    fixture = tmp_path / "fixture"
    subprocess.run(
        [sys.executable, str(generator), "--output-dir", str(fixture), "--sequence-length", "8"],
        check=True,
    )
    release = tmp_path / "release"
    assert curate([
        "release", "--recipe", str(fixture / "recipe.json"), "--work-dir", str(tmp_path / "work"),
        "--output", str(release), "--workers", "2",
    ]) == 0
    digest = hashlib.sha256((release / "manifest.json").read_bytes()).hexdigest()
    registry = tmp_path / "registry.jsonl"
    initialize_registry(registry)
    register_release(registry, digest)
    return release, digest, registry


def _args(curated, output, *extra, max_steps=3, budget=6):
    release, digest, registry = curated
    return [
        "--curated-release", str(release), "--curated-release-sha256", digest,
        "--release-status-registry", str(registry), "--output-dir", str(output),
        "--device", "cpu", "--model-profile", "smoke", "--max-steps", str(max_steps),
        "--batch-size", "1", "--eval-every", "1", "--eval-batches", "1",
        "--save-every", "1", "--warmup-steps", "0",
        "--targets-per-step", "2", "--max-train-tokens", str(budget), *extra,
    ]


def test_native_curated_training_resumes_at_exact_cursor_and_keeps_test_isolated(curated, tmp_path, monkeypatch):
    uninterrupted = tmp_path / "uninterrupted"
    interrupted = tmp_path / "interrupted"
    calls = []
    original_next = TrainingRelease.next_batch

    def track_batch(self, split, cursor, batch_size, **options):
        calls.append((split, cursor.copy()))
        return original_next(self, split, cursor, batch_size, **options)

    def unexpected_stream(*args, **kwargs):
        raise AssertionError("curated pretraining used a FineWeb-Edu stream")

    monkeypatch.setattr(TrainingRelease, "next_batch", track_batch)
    monkeypatch.setattr(trainer, "make_stream", unexpected_stream)
    assert trainer.main(_args(curated, uninterrupted)) == 0

    class SimulatedInterruption(Exception):
        pass

    def interrupt_registration(*args, **kwargs):
        raise SimulatedInterruption

    calls.clear()
    with monkeypatch.context() as one_step:
        one_step.setattr(release_governance, "register_checkpoint", interrupt_registration)
        with pytest.raises(SimulatedInterruption):
            trainer.main(_args(curated, interrupted))
    first = resolve_committed_checkpoint(interrupted)
    first_manifest = _read_json(first / "run_manifest.json")
    assert first_manifest["tokens_seen"] == 2
    assert verify_registry(curated[2])["event_count"] == 4
    with pytest.raises(RuntimeError, match="completed training checkpoint"):
        trainer.main(_args(curated, interrupted, "--resume", "--evaluate-test"))
    with pytest.raises(RuntimeError, match="max_steps differs"):
        trainer.main(_args(curated, interrupted, "--resume", max_steps=4))

    calls.clear()
    assert trainer.main(_args(curated, interrupted, "--resume")) == 0
    train_cursors = [cursor for split, cursor in calls if split == "train"]
    assert len(train_cursors) == 2
    assert train_cursors[0] == first_manifest["data_cursor"]
    assert {split for split, _ in calls} == {"train", "validation"}
    assert not (interrupted / "test_evaluation.json").exists()

    continuous_final = resolve_committed_checkpoint(uninterrupted)
    resumed_final = resolve_committed_checkpoint(interrupted)
    for checkpoint in (continuous_final, resumed_final):
        manifest = _read_json(checkpoint / "run_manifest.json")
        assert manifest["step"] == 3
        assert manifest["tokens_seen"] == 6
        assert manifest["release_manifest_sha256"] == curated[1]
    assert _read_json(continuous_final / "run_manifest.json")["data_cursor"] == (
        _read_json(resumed_final / "run_manifest.json")["data_cursor"]
    )
    assert _read_json(interrupted / "run_summary.json")["tokens_this_session"] == 4
    continuous_weights = torch.load(continuous_final / "model" / "model.pt", weights_only=True)
    resumed_weights = torch.load(resumed_final / "model" / "model.pt", weights_only=True)
    torch.testing.assert_close(continuous_weights, resumed_weights, rtol=0, atol=0)

    calls.clear()
    assert trainer.main(_args(curated, interrupted, "--resume", "--evaluate-test")) == 0
    assert {split for split, _ in calls} == {"test"}
    report = _read_json(interrupted / "test_evaluation.json")
    assert report["release_manifest_sha256"] == curated[1]
    assert report["checkpoint_manifest_sha256"] == hashlib.sha256((resumed_final / "checkpoint.json").read_bytes()).hexdigest()
    assert report["active_targets"] == _read_json(curated[0] / "manifest.json")["splits"]["test"]["active_targets"]
    assert report["test_loss"] > 0
    assert verify_registry(curated[2])["event_count"] == 7

    quarantine_release(curated[2], curated[1], "a" * 64)
    with pytest.raises(ValueError, match="quarantined"):
        trainer.main(_args(curated, interrupted, "--resume"))


def test_curated_budget_stops_on_partial_active_batch(curated, tmp_path):
    output = tmp_path / "partial"
    assert trainer.main(_args(curated, output, budget=3)) == 0
    summary = _read_json(output / "run_summary.json")
    assert summary["steps_this_session"] == 2
    assert summary["tokens_seen_total"] == 3
    assert _read_json(resolve_committed_checkpoint(output) / "run_manifest.json")["tokens_seen"] == 3


def test_masked_causal_loss_only_counts_the_selected_targets():
    from types import SimpleNamespace

    logits = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5)
    inputs = torch.tensor([[0, 4, 1, 3]])
    mask = torch.tensor([[True, False, True]])

    def model(input_ids):
        assert torch.equal(input_ids, inputs)
        return SimpleNamespace(logits=logits, auxiliary_loss=torch.tensor(0.0))

    task, _ = trainer.causal_loss(model, inputs, mask)
    expected = torch.nn.functional.cross_entropy(logits[0, [0, 2]], torch.tensor([4, 3]))
    torch.testing.assert_close(task, expected)
