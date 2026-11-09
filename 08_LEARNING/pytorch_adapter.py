"""Optional PyTorch batch/training-step adapter for ADA Learning.

This module is deliberately opt-in and imports PyTorch only when this module
is imported. It consumes already-validated replay records, constructs padded
causal-LM batches from token IDs stored in record payloads, and performs one
optimizer step when explicitly invoked by the caller.

It does not fetch datasets, tokenize text, validate records, save checkpoints,
promote artifacts, or deploy models. Evaluation and promotion remain the
responsibility of LearningController and its RegressionGate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "PyTorchLearningAdapter requires PyTorch; install ADA with the neural extra."
    ) from exc

from .contracts import ExperienceRecord, ExperienceStatus
from .replay_mechanism.buffer import ReplayBuffer


@dataclass(frozen=True, slots=True)
class TorchBatch:
    """Padded causal-LM batch with an explicit boolean attention mask."""

    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    experience_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    """Detached scalar metrics from one explicitly requested optimizer step."""

    loss: float
    task_loss: float
    auxiliary_loss: float
    token_count: int
    experience_ids: tuple[str, ...]


class PyTorchLearningAdapter:
    """Bridge validated ADA replay records to a caller-supplied PyTorch model.

    Each record payload must contain a non-empty integer sequence under
    `token_ids`. The adapter intentionally does not infer tokenization or
    vocabularies. Padding is represented by `pad_token_id` and excluded from
    the loss with label value -100.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        pad_token_id: int,
        max_seq_len: int,
        auxiliary_loss_weight: float = 0.0,
        device: str | torch.device | None = None,
        gradient_clip_norm: float | None = 1.0,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer")
        if type(max_seq_len) is not int or max_seq_len < 2:
            raise ValueError("max_seq_len must be an integer >= 2")
        if isinstance(auxiliary_loss_weight, bool) or not isinstance(
            auxiliary_loss_weight, (int, float)
        ):
            raise TypeError("auxiliary_loss_weight must be numeric")
        if not torch.isfinite(torch.tensor(float(auxiliary_loss_weight))).item():
            raise ValueError("auxiliary_loss_weight must be finite")
        if auxiliary_loss_weight < 0:
            raise ValueError("auxiliary_loss_weight must be non-negative")
        if gradient_clip_norm is not None:
            if isinstance(gradient_clip_norm, bool) or not isinstance(
                gradient_clip_norm, (int, float)
            ):
                raise TypeError("gradient_clip_norm must be numeric or None")
            if not torch.isfinite(torch.tensor(float(gradient_clip_norm))).item():
                raise ValueError("gradient_clip_norm must be finite")
            if gradient_clip_norm <= 0:
                raise ValueError("gradient_clip_norm must be positive")
        self.model = model
        self.optimizer = optimizer
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.device = torch.device(device) if device is not None else self._model_device(model)
        self.gradient_clip_norm = (
            float(gradient_clip_norm) if gradient_clip_norm is not None else None
        )

    @staticmethod
    def _model_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def build_batch(self, records: Iterable[ExperienceRecord]) -> TorchBatch:
        """Build a padded batch; only VALIDATED experience is accepted."""
        items = tuple(records)
        if not items:
            raise ValueError("cannot build a batch from empty experience")
        if any(not isinstance(item, ExperienceRecord) for item in items):
            raise TypeError("records must contain ExperienceRecord values")
        if any(item.status is not ExperienceStatus.VALIDATED for item in items):
            raise ValueError("only validated experience can form a training batch")

        sequences: list[list[int]] = []
        for item in items:
            token_ids = item.payload.get("token_ids")
            if not isinstance(token_ids, (tuple, list)) or len(token_ids) < 2:
                raise ValueError(
                    f"experience {item.experience_id} requires at least two token_ids"
                )
            if len(token_ids) > self.max_seq_len:
                raise ValueError(
                    f"experience {item.experience_id} exceeds max_seq_len"
                )
            sequence = []
            for token in token_ids:
                if type(token) is not int or token < 0:
                    raise ValueError("token_ids must contain non-negative integers")
                sequence.append(token)
            sequences.append(sequence)

        batch_size = len(sequences)
        width = max(map(len, sequences))
        input_ids = torch.full(
            (batch_size, width), self.pad_token_id,
            dtype=torch.long, device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, width), dtype=torch.bool, device=self.device
        )
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(
                sequence, dtype=torch.long, device=self.device
            )
            attention_mask[row, :length] = True

        labels = input_ids.clone()
        labels[~attention_mask] = -100
        return TorchBatch(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            experience_ids=tuple(item.experience_id for item in items),
        )

    def train_step(self, records: Iterable[ExperienceRecord]) -> TrainingStepResult:
        """Run exactly one explicitly requested optimizer step."""
        batch = self.build_batch(records)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        task_loss = getattr(output, "loss", None)
        if task_loss is None:
            raise ValueError("model output must expose a non-None causal-LM loss")
        auxiliary_loss = getattr(output, "auxiliary_loss", None)
        if auxiliary_loss is None:
            auxiliary_loss = task_loss.new_zeros(())
        if not isinstance(task_loss, Tensor) or task_loss.ndim != 0:
            raise TypeError("model task loss must be a scalar tensor")
        if not isinstance(auxiliary_loss, Tensor) or auxiliary_loss.ndim != 0:
            raise TypeError("model auxiliary_loss must be a scalar tensor")
        if not torch.isfinite(task_loss).item() or not torch.isfinite(auxiliary_loss).item():
            raise FloatingPointError("model returned a non-finite loss")
        total_loss = task_loss + self.auxiliary_loss_weight * auxiliary_loss
        if not torch.isfinite(total_loss).item():
            raise FloatingPointError("combined training loss is non-finite")
        total_loss.backward()
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm,
                error_if_nonfinite=True,
            )
        self.optimizer.step()

        return TrainingStepResult(
            loss=float(total_loss.detach().cpu()),
            task_loss=float(task_loss.detach().cpu()),
            auxiliary_loss=float(auxiliary_loss.detach().cpu()),
            token_count=int(batch.attention_mask.sum().item()),
            experience_ids=batch.experience_ids,
        )


def train_replay_step(
    replay: ReplayBuffer,
    adapter: PyTorchLearningAdapter,
    batch_size: int,
    *,
    domain: str | None = None,
) -> TrainingStepResult:
    """Sample validated records from replay and run one PyTorch training step."""
    if not isinstance(replay, ReplayBuffer):
        raise TypeError("replay must be a ReplayBuffer")
    if not isinstance(adapter, PyTorchLearningAdapter):
        raise TypeError("adapter must be a PyTorchLearningAdapter")
    records = replay.sample(batch_size, domain=domain)
    if not records:
        raise ValueError("replay buffer contains no records matching the request")
    return adapter.train_step(records)
