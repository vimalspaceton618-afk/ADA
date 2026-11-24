"""Regression coverage for exact data-prefix replay and resume integrity."""
from __future__ import annotations

import pytest

from ada.pretraining_resume import (
    replay_completed_batches, validate_checkpoint_directory,
    validate_resume_manifest,
)


def _complete_checkpoint(tmp_path):
    for path in (
        "tokenizer.json", "model/config.json", "model/model.pt",
        "training_state.pt", "run_manifest.json",
    ):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"test")


def test_resume_rejects_incomplete_checkpoint(tmp_path):
    with pytest.raises(RuntimeError, match="incomplete checkpoint"):
        validate_checkpoint_directory(tmp_path, resume=True)
    _complete_checkpoint(tmp_path)
    validate_checkpoint_directory(tmp_path, resume=True)
    (tmp_path / "model/model.pt").unlink()
    with pytest.raises(RuntimeError, match="model/model.pt"):
        validate_checkpoint_directory(tmp_path, resume=True)


def test_fresh_run_rejects_all_previous_run_artifacts(tmp_path):
    validate_checkpoint_directory(tmp_path, resume=False)
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(RuntimeError, match="already contains"):
        validate_checkpoint_directory(tmp_path, resume=False)


def test_resume_manifest_rejects_changes_and_accepts_legacy_fields():
    expected = {
        "dataset_id": "fineweb", "dataset_config": "sample",
        "dataset_revision": "revision", "batch_size": 2,
        "sequence_length": 8, "random_seed": 42,
        "max_document_chars": 1000, "learning_rate": 0.001,
        "weight_decay": 0.1, "warmup_steps": 10,
        "tokenizer_sha256": "abc",
    }
    validate_resume_manifest(dict(expected), expected)
    for key in expected:
        altered = dict(expected)
        altered[key] = "different"
        with pytest.raises(RuntimeError, match=key):
            validate_resume_manifest(altered, expected)
    # Legacy checkpoints did not store the extended fields.
    validate_resume_manifest(
        {k: expected[k] for k in (
            "dataset_id", "dataset_config", "dataset_revision",
            "batch_size", "sequence_length", "random_seed",
        )}, expected,
    )


def test_replay_resume_continues_at_next_batch_without_duplication():
    def batches():
        for value in range(9):
            yield (value, value * 10)
    initial = batches()
    consumed = [next(initial) for _ in range(5)]
    resumed = replay_completed_batches(batches(), len(consumed))
    assert next(resumed) == (5, 50)
    assert list(resumed) == [(6, 60), (7, 70), (8, 80)]
    assert consumed[-1] == (4, 40)


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_replay_rejects_invalid_step(value):
    with pytest.raises(ValueError, match="completed_steps"):
        replay_completed_batches(iter([]), value)


def test_replay_detects_exhausted_stream():
    with pytest.raises(RuntimeError, match="completed step 3"):
        replay_completed_batches(iter([1, 2]), 3)
