"""Crash-recovery and integrity tests for ADA's published checkpoints."""
from __future__ import annotations

import json

import pytest

from ada.atomic_checkpoint import (
    REQUIRED_FILES, commit_checkpoint, resolve_committed_checkpoint,
)
from ada.pretraining_resume import validate_checkpoint_directory


def _writer(marker: str):
    def write(stage):
        for relative in REQUIRED_FILES:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"{marker}:{relative}".encode())
    return write


def test_published_checkpoints_are_versioned_and_verified(tmp_path):
    one = commit_checkpoint(tmp_path, step=1, tokens_seen=16, writer=_writer("one"))
    assert resolve_committed_checkpoint(tmp_path) == one
    assert (one / "checkpoint.json").is_file()
    two = commit_checkpoint(tmp_path, step=2, tokens_seen=32, writer=_writer("two"))
    assert one.exists() and two.exists() and one != two
    assert resolve_committed_checkpoint(tmp_path) == two
    latest = json.loads((tmp_path / "checkpoints/latest.json").read_text())
    assert latest["step"] == 2 and latest["checkpoint"] == two.name


def test_failed_writer_does_not_replace_good_snapshot(tmp_path):
    one = commit_checkpoint(tmp_path, step=2, tokens_seen=20, writer=_writer("old"))
    before = (tmp_path / "checkpoints/latest.json").read_bytes()

    def fail(stage):
        _writer("partial")(stage)
        raise OSError("simulated disk interruption")

    with pytest.raises(OSError, match="interruption"):
        commit_checkpoint(tmp_path, step=3, tokens_seen=30, writer=fail)
    assert (tmp_path / "checkpoints/latest.json").read_bytes() == before
    assert resolve_committed_checkpoint(tmp_path) == one
    assert not list((tmp_path / "checkpoints").glob(".staging-*"))


def test_crash_after_directory_rename_preserves_old_published_pointer(tmp_path, monkeypatch):
    from ada import atomic_checkpoint

    original = commit_checkpoint(tmp_path, step=1, tokens_seen=8, writer=_writer("old"))
    monkeypatch.setattr(
        atomic_checkpoint, "_atomic_pointer",
        lambda *_: (_ for _ in ()).throw(OSError("interrupted before publish")),
    )
    with pytest.raises(OSError, match="before publish"):
        commit_checkpoint(tmp_path, step=2, tokens_seen=16, writer=_writer("new"))
    assert resolve_committed_checkpoint(tmp_path) == original
    # Unreferenced snapshots are orphans, never silently treated as committed.
    assert len(list((tmp_path / "checkpoints").glob("step-000000000002-*"))) == 1


def test_incomplete_and_tampered_published_artifact_fails_closed(tmp_path):
    committed = commit_checkpoint(tmp_path, step=1, tokens_seen=8, writer=_writer("good"))
    (committed / "model/model.pt").write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        resolve_committed_checkpoint(tmp_path)


def test_pointer_traversal_and_index_tampering_are_rejected(tmp_path):
    committed = commit_checkpoint(tmp_path, step=1, tokens_seen=8, writer=_writer("good"))
    pointer = tmp_path / "checkpoints/latest.json"
    data = json.loads(pointer.read_text())
    data["checkpoint"] = "../outside"
    pointer.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="pointer"):
        resolve_committed_checkpoint(tmp_path)
    data["checkpoint"] = committed.name
    pointer.write_text(json.dumps(data))
    (committed / "checkpoint.json").write_text("{}")
    with pytest.raises(RuntimeError, match="index digest mismatch"):
        resolve_committed_checkpoint(tmp_path)


def test_reject_nonprogressing_steps_and_fresh_runs_over_snapshots(tmp_path):
    commit_checkpoint(tmp_path, step=1, tokens_seen=8, writer=_writer("good"))
    with pytest.raises(ValueError, match="advance"):
        commit_checkpoint(tmp_path, step=1, tokens_seen=8, writer=_writer("again"))
    with pytest.raises(RuntimeError, match="already contains"):
        validate_checkpoint_directory(tmp_path, resume=False)


@pytest.mark.parametrize("step,tokens", [(0, 1), (True, 1), (1, -1)])
def test_invalid_step_and_token_counters_are_rejected(tmp_path, step, tokens):
    with pytest.raises(ValueError):
        commit_checkpoint(tmp_path, step=step, tokens_seen=tokens, writer=_writer("x"))


def test_torch_optimizer_rng_payload_restores_from_published_snapshot(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(27)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    expected = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    def write(stage):
        _writer("metadata")(stage)
        torch.save(model.state_dict(), stage / "model/model.pt")
        torch.save({
            "step": 1, "tokens_seen": 8,
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        }, stage / "training_state.pt")

    committed = commit_checkpoint(tmp_path, step=1, tokens_seen=8, writer=write)
    restored = torch.nn.Linear(4, 3)
    restored.load_state_dict(
        torch.load(resolve_committed_checkpoint(tmp_path) / "model/model.pt",
                   weights_only=True)
    )
    for name, expected_tensor in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], expected_tensor)
    state = torch.load(committed / "training_state.pt", weights_only=True)
    assert state["step"] == 1 and state["tokens_seen"] == 8
    assert state["optimizer"]["state"]
