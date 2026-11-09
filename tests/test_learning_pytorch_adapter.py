"""Tests for the optional PyTorch Learning adapter."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
torch = pytest.importorskip("torch")

from ada.learning import (
    ExperienceKind,
    ExperienceRecord,
    ExperienceStatus,
    PyTorchLearningAdapter,
    ReplayBuffer,
    train_replay_step,
)

DIGEST = "b" * 64
NOW = datetime(2026, 11, 9, tzinfo=timezone.utc)


class TinyCausalModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 12) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.head = torch.nn.Linear(dim, vocab_size)
        self.aux_parameter = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids, *, attention_mask=None, labels=None):
        logits = self.head(self.embedding(input_ids))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return type("Output", (), {
            "loss": loss,
            "auxiliary_loss": self.aux_parameter.square(),
            "logits": logits,
        })()


def record(tokens, *, status=ExperienceStatus.VALIDATED, source_id="src"):
    return ExperienceRecord(
        kind=ExperienceKind.SUPERVISED,
        domain="language",
        task_id="task-1",
        source_kind="dataset",
        source_id=source_id,
        source_digest=DIGEST,
        payload={"token_ids": tokens},
        status=status,
        created_at=NOW,
    )


def test_batch_masks_padding_and_preserves_record_identity():
    adapter = PyTorchLearningAdapter(
        TinyCausalModel(), torch.optim.AdamW(TinyCausalModel().parameters(), lr=1e-3),
        pad_token_id=0, max_seq_len=8,
    )
    items = (record([1, 2, 3]), record([4, 5]))
    batch = adapter.build_batch(items)
    assert batch.input_ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch.attention_mask.tolist() == [[True, True, True], [True, True, False]]
    assert batch.labels.tolist() == [[1, 2, 3], [4, 5, -100]]
    assert batch.experience_ids == tuple(item.experience_id for item in items)


def test_candidate_experience_cannot_become_training_data():
    adapter = PyTorchLearningAdapter(
        TinyCausalModel(), torch.optim.AdamW(TinyCausalModel().parameters(), lr=1e-3),
        pad_token_id=0, max_seq_len=8,
    )
    with pytest.raises(ValueError, match="only validated"):
        adapter.build_batch((record([1, 2], status=ExperienceStatus.CANDIDATE),))


def test_invalid_or_too_long_token_sequences_are_rejected():
    adapter = PyTorchLearningAdapter(
        TinyCausalModel(), torch.optim.AdamW(TinyCausalModel().parameters(), lr=1e-3),
        pad_token_id=0, max_seq_len=3,
    )
    with pytest.raises(ValueError, match="at least two"):
        adapter.build_batch((record([1]),))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        adapter.build_batch((record([1, 2, 3, 4]),))
    with pytest.raises(ValueError, match="non-negative integers"):
        adapter.build_batch((record([1, -2]),))


def test_training_step_uses_task_plus_weighted_auxiliary_loss():
    model = TinyCausalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    adapter = PyTorchLearningAdapter(
        model, optimizer, pad_token_id=0, max_seq_len=8,
        auxiliary_loss_weight=0.25, gradient_clip_norm=1.0,
    )
    before = model.aux_parameter.detach().clone()
    result = adapter.train_step((record([1, 2, 3, 4]),))
    assert result.token_count == 4
    assert result.loss == pytest.approx(
        result.task_loss + 0.25 * result.auxiliary_loss, rel=1e-6
    )
    assert torch.isfinite(model.aux_parameter)
    assert not torch.equal(before, model.aux_parameter.detach())


def test_replay_helper_samples_only_validated_records():
    model = TinyCausalModel()
    adapter = PyTorchLearningAdapter(
        model, torch.optim.AdamW(model.parameters(), lr=1e-3),
        pad_token_id=0, max_seq_len=8,
    )
    replay = ReplayBuffer()
    replay.add(record([1, 2, 3]))
    result = train_replay_step(replay, adapter, 1, domain="language")
    assert result.token_count == 3
    assert len(result.experience_ids) == 1


def test_replay_helper_rejects_empty_or_wrong_buffer():
    model = TinyCausalModel()
    adapter = PyTorchLearningAdapter(
        model, torch.optim.AdamW(model.parameters(), lr=1e-3),
        pad_token_id=0, max_seq_len=8,
    )
    with pytest.raises(ValueError, match="no records"):
        train_replay_step(ReplayBuffer(), adapter, 1)
    with pytest.raises(TypeError, match="ReplayBuffer"):
        train_replay_step(object(), adapter, 1)


def test_training_step_rejects_non_finite_losses_before_backward():
    class BadModel(TinyCausalModel):
        def forward(self, input_ids, *, attention_mask=None, labels=None):
            logits = self.head(self.embedding(input_ids))
            return type("Output", (), {
                "loss": logits.sum() * torch.tensor(float("nan")),
                "auxiliary_loss": logits.new_zeros(()),
            })()

    model = BadModel()
    adapter = PyTorchLearningAdapter(
        model, torch.optim.AdamW(model.parameters(), lr=1e-3),
        pad_token_id=0, max_seq_len=8,
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        adapter.train_step((record([1, 2, 3]),))
