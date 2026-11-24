"""Dependency-light integrity checks for resumable ADA pretraining.

Streaming batches are deterministically reconstructed by replaying the prefix.
This avoids claiming Hugging Face shuffle-buffer snapshots are exact resumes.
Replay is O(number of completed batches); large jobs should use a checkpointed,
shard-aware data loader instead.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TypeVar

_Batch = TypeVar("_Batch")

REQUIRED_CHECKPOINT_FILES = (
    "tokenizer.json",
    "model/config.json",
    "model/model.pt",
    "training_state.pt",
    "run_manifest.json",
)
RUN_ARTIFACTS = REQUIRED_CHECKPOINT_FILES + ("metrics.jsonl",)


def validate_checkpoint_directory(output_dir: Path, *, resume: bool) -> None:
    """Fail closed instead of silently starting with mismatched partial state."""
    if resume:
        missing = [
            name for name in REQUIRED_CHECKPOINT_FILES
            if not (output_dir / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                "Cannot --resume: incomplete checkpoint (missing "
                + ", ".join(missing) + "). Restore a complete run or use "
                "a new output directory."
            )
    elif (output_dir / "checkpoints").exists() or (output_dir / "run_summary.json").exists() or any(
        (output_dir / name).exists() for name in RUN_ARTIFACTS
    ):
        raise RuntimeError(
            "Output directory already contains pretraining artifacts. "
            "Use --resume for a complete checkpoint or select a new --output-dir."
        )


def validate_resume_manifest(
    manifest: Mapping[str, object], expected: Mapping[str, object],
) -> None:
    """Reject data-order and optimization hyperparameter changes on resume.

    Fields introduced after the original checkpoint format are optional when
    reading historical manifests, but enforced on all newly written runs.
    """
    required = (
        "dataset_id", "dataset_config", "dataset_revision",
        "batch_size", "sequence_length", "random_seed",
    )
    for name in required:
        if manifest.get(name) != expected[name]:
            raise RuntimeError(
                f"Cannot --resume: {name} differs from checkpoint "
                f"({manifest.get(name)!r} != {expected[name]!r})."
            )
    for name in (
        "max_document_chars", "learning_rate", "weight_decay",
        "warmup_steps", "tokenizer_sha256",
    ):
        if name in manifest and manifest[name] != expected[name]:
            raise RuntimeError(
                f"Cannot --resume: {name} differs from checkpoint "
                f"({manifest[name]!r} != {expected[name]!r})."
            )


def replay_completed_batches(
    batches: Iterator[_Batch], completed_steps: int,
) -> Iterator[_Batch]:
    """Advance a deterministic iterable to the next unseen training batch.

    This must be invoked with the same data revision, ordering seed,
    tokenizer, and packing parameters as the original run. Do not reset the
    generator or restore global training RNG *after* the next actual batch.
    """
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("completed_steps must be a non-negative integer")
    for step in range(completed_steps):
        try:
            next(batches)
        except StopIteration as exc:
            raise RuntimeError(
                f"Training stream ended while replaying completed step {step + 1}."
            ) from exc
    return batches
