#!/usr/bin/env python3
"""Pretrain ADAForCausalLM on FineWeb-Edu or a pinned local ADA-Datasets release.

This is a bounded first pretraining run, not a benchmark or a claim of broad
language capability. Dataset records are streamed; the full corpus is never
downloaded into RAM or committed to this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from ada.neural import ADAConfig, ADAForCausalLM
from ada.atomic_checkpoint import commit_checkpoint, resolve_committed_checkpoint
from ada.pretraining_resume import (
    replay_completed_batches,
    validate_checkpoint_directory,
    validate_resume_manifest,
)

if TYPE_CHECKING:
    from ada_datasets.training import TrainingRelease

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_SPLIT = "train"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DATASET_LICENSE = "ODC-By 1.0 (dataset declaration); Common Crawl terms also apply"
DATASET_CARD = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu"
DATASET_PAPER = "https://arxiv.org/abs/2406.17557"
SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sequence-length", type=int,
                   help="Defaults to 256 for FineWeb-Edu or the curated release width.")
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--tokenizer-docs", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-document-chars", type=int, default=100000)
    p.add_argument("--curated-release", type=Path)
    p.add_argument("--curated-release-sha256")
    p.add_argument("--release-status-registry", type=Path)
    p.add_argument("--model-profile", choices=("pilot", "smoke"), default="pilot")
    p.add_argument("--max-train-tokens", type=int, help="Exact curated active-target budget.")
    p.add_argument("--targets-per-step", type=int, help="Cap and require this many active targets per curated batch.")
    p.add_argument("--evaluate-test", action="store_true", help="Evaluate a completed curated checkpoint on test only.")
    a = p.parse_args(argv)
    if a.curated_release is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", a.curated_release_sha256 or ""):
            p.error("--curated-release requires a lowercase --curated-release-sha256")
        if a.release_status_registry is None:
            p.error("--curated-release requires --release-status-registry")
        if a.evaluate_test and not a.resume:
            p.error("--evaluate-test requires --resume with a completed checkpoint")
    else:
        if (a.curated_release_sha256 is not None or a.release_status_registry is not None
                or a.max_train_tokens is not None or a.targets_per_step is not None or a.evaluate_test):
            p.error("curated data flags require --curated-release")
        if a.sequence_length is None:
            a.sequence_length = 256
    for name in ("max_train_tokens", "targets_per_step"):
        value = getattr(a, name)
        if value is not None and value < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_steps", "batch_size", "vocab_size",
                 "tokenizer_docs", "eval_every", "eval_batches", "save_every",
                 "max_document_chars"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if ((a.sequence_length is not None and a.sequence_length < 2)
            or (a.curated_release is None and a.vocab_size < len(SPECIAL_TOKENS) + 256)):
        p.error("sequence length must be >=2 and vocab size >=260")
    if (not math.isfinite(a.learning_rate) or not math.isfinite(a.weight_decay)
            or a.learning_rate <= 0 or a.weight_decay < 0 or a.warmup_steps < 0):
        p.error("invalid optimizer settings")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable. Select a GPU runtime in Colab.")
    return a


def stable_bucket(row: dict) -> int:
    identity = row.get("id") or row.get("url") or row.get("text", "")[:2048]
    digest = hashlib.sha256(str(identity).encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:8], 16) % 100


def usable_record(row: dict) -> bool:
    text = row.get("text")
    return (
        isinstance(text, str)
        and len(text.strip()) >= 200
        and row.get("language", "en") == "en"
        and int(row.get("int_score", 3) or 0) >= 3
    )


def make_stream(split: str, seed: int):
    from datasets import load_dataset

    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    stream = load_dataset(
        DATASET_ID,
        name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        streaming=True,
    )
    if split == "train":
        stream = stream.filter(lambda row: usable_record(row) and stable_bucket(row) != 0)
        stream = stream.shuffle(seed=seed, buffer_size=5000)
    else:
        stream = stream.filter(lambda row: usable_record(row) and stable_bucket(row) == 0)
    return stream


def train_tokenizer(output_path: Path, args: argparse.Namespace) -> Tokenizer:
    if output_path.exists():
        print(f"[tokenizer] Reusing {output_path}", flush=True)
        return Tokenizer.from_file(str(output_path))

    print(
        f"[tokenizer] Training byte-level BPE, vocab={args.vocab_size}, "
        f"documents<={args.tokenizer_docs:,}",
        flush=True,
    )
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    sampled_stream = make_stream("train", args.seed).take(args.tokenizer_docs)

    def text_iterator() -> Iterator[str]:
        for row in sampled_stream:
            text = row["text"][: args.max_document_chars].strip()
            if text:
                yield text

    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer,
        length=args.tokenizer_docs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"[tokenizer] Saved {tokenizer.get_vocab_size():,} tokens to {output_path}", flush=True)
    return tokenizer


def packed_blocks(stream, tokenizer: Tokenizer, sequence_length: int,
                  max_document_chars: int) -> Iterator[torch.Tensor]:
    eos_id = tokenizer.token_to_id("<|eos|>")
    bos_id = tokenizer.token_to_id("<|bos|>")
    if eos_id is None or bos_id is None:
        raise RuntimeError("tokenizer is missing <|bos|> or <|eos|>")
    buffer: list[int] = []
    block_width = sequence_length + 1
    for row in stream:
        if not usable_record(row):
            continue
        text = row["text"][:max_document_chars].strip()
        if not text:
            continue
        ids = tokenizer.encode(text).ids
        if not ids:
            continue
        buffer.append(bos_id)
        buffer.extend(ids)
        buffer.append(eos_id)
        while len(buffer) >= block_width:
            block = buffer[:block_width]
            del buffer[:block_width]
            yield torch.tensor(block, dtype=torch.long)


def infinite_train_batches(tokenizer: Tokenizer, args: argparse.Namespace):
    epoch = 0
    while True:
        stream = make_stream("train", args.seed + epoch)
        count = 0
        rows: list[torch.Tensor] = []
        for block in packed_blocks(stream, tokenizer, args.sequence_length,
                                   args.max_document_chars):
            rows.append(block)
            if len(rows) == args.batch_size:
                yield torch.stack(rows)
                rows.clear()
                count += 1
        if rows:
            # Drop incomplete final batch to preserve fixed batch shapes.
            rows.clear()
        if count == 0:
            raise RuntimeError("No training batches produced; inspect dataset access/filter.")
        epoch += 1


def finite_batches(tokenizer: Tokenizer, args: argparse.Namespace):
    stream = make_stream("validation", args.seed)
    rows: list[torch.Tensor] = []
    produced = 0
    for block in packed_blocks(stream, tokenizer, args.sequence_length,
                               args.max_document_chars):
        rows.append(block)
        if len(rows) == args.batch_size:
            yield torch.stack(rows)
            rows.clear()
            produced += 1
            if produced >= args.eval_batches:
                return


def packed_release_batch(
    release: TrainingRelease, split: str, cursor: dict, batch_size: int, *,
    repeat: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    rows, next_cursor = release.next_batch(split, cursor, batch_size, repeat=repeat)
    batch = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long)
    mask = torch.tensor([row["loss_mask"] for row in rows], dtype=torch.bool)
    if not mask.any():
        raise RuntimeError(f"Curated {split} batch contains no loss targets.")
    return batch, mask, next_cursor


def curated_eval_batches(release: TrainingRelease, args: argparse.Namespace, split: str):
    cursor = release.cursor(split)
    remaining = args.eval_batches if split == "validation" else None
    while remaining is None or remaining > 0:
        try:
            batch, mask, cursor = packed_release_batch(
                release, split, cursor, args.batch_size,
            )
        except StopIteration:
            return
        yield batch, mask
        if remaining is not None:
            remaining -= 1


def causal_loss(model, input_ids: torch.Tensor,
                loss_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(input_ids)
    logits = output.logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    if loss_mask is not None:
        targets = targets.masked_fill(~loss_mask, -100)
    task_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
    total_loss = task_loss + 0.01 * output.auxiliary_loss
    return task_loss, total_loss


@torch.no_grad()
def evaluate(model, tokenizer: Tokenizer, args: argparse.Namespace,
             device: torch.device, release: TrainingRelease | None = None, *,
             split: str = "validation") -> float:
    model.eval()
    weighted_loss = 0.0
    total_weight = 0
    batches = ((batch, None) for batch in finite_batches(tokenizer, args)) if release is None else (
        curated_eval_batches(release, args, split)
    )
    try:
        for batch, mask in batches:
            weight = 1 if mask is None else int(mask.sum())
            task_loss, _ = causal_loss(
                model, batch.to(device, non_blocking=True),
                None if mask is None else mask.to(device, non_blocking=True),
            )
            value = float(task_loss.cpu())
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite {split} loss.")
            weighted_loss += value * weight
            total_weight += weight
        if not total_weight:
            raise RuntimeError(f"{split.capitalize()} split produced no batches.")
        if split == "test" and total_weight != release.manifest["splits"]["test"]["active_targets"]:
            raise RuntimeError("Test target count differs from the curated release manifest.")
        return weighted_loss / total_weight
    finally:
        model.train()


def learning_rate_at(step: int, args: argparse.Namespace) -> float:
    if args.warmup_steps and step < args.warmup_steps:
        return args.learning_rate * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_curated_release(args: argparse.Namespace) -> TrainingRelease | None:
    if args.curated_release is None:
        return None
    try:
        from ada_datasets.training import TrainingRelease
    except ImportError as exc:
        raise RuntimeError("Install ADA-Datasets with the release extra for curated pretraining.") from exc
    release = TrainingRelease(
        args.curated_release, args.curated_release_sha256,
        status_path=args.release_status_registry,
    )
    if args.sequence_length is not None and args.sequence_length != release.sequence_length:
        raise RuntimeError("--sequence-length differs from the curated release.")
    args.sequence_length = release.sequence_length
    capacity = args.batch_size * args.sequence_length
    if args.targets_per_step is not None and args.targets_per_step > capacity:
        raise RuntimeError("--targets-per-step exceeds batch capacity.")
    if (args.max_train_tokens is not None
            and args.max_train_tokens > args.max_steps * (args.targets_per_step or capacity)):
        raise RuntimeError("--max-train-tokens exceeds the configured step capacity.")
    return release


def dataset_metadata(release: TrainingRelease | None) -> dict:
    if release is not None:
        return {
            "dataset_mode": "curated",
            "dataset_id": "ADA-Datasets",
            "dataset_config": "ada-training-release",
            "dataset_revision": release.manifest_sha256,
            "release_manifest_sha256": release.manifest_sha256,
            "dataset_split": "train",
            "tokenizer_sha256": release.tokenizer_sha256,
        }
    return {
        "dataset_mode": "fineweb",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "dataset_card": DATASET_CARD,
        "dataset_paper": DATASET_PAPER,
        "split_method": "stable SHA-256 bucket from record id/url/text; bucket 0 validation, buckets 1-99 train",
        "filters": {"language": "en", "int_score_min": 3, "minimum_text_chars": 200},
    }


def make_model_config(tokenizer: Tokenizer, args: argparse.Namespace) -> ADAConfig:
    if args.model_profile == "smoke":
        return ADAConfig(
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id=tokenizer.token_to_id("<|pad|>"),
            max_seq_len=args.sequence_length + 1,
            dim=8, n_layers=2, n_heads=2, attention_rank=4,
            num_experts=2, num_slots=4, expansion_factor=1, mixer_expand=2,
            mixer_kernel_size=2, mixer_backend="mamba2_reference", mamba2_d_state=3,
            mamba2_headdim=4, mamba2_ngroups=1, mamba2_chunk_size=2,
        )
    return ADAConfig(
        vocab_size=tokenizer.get_vocab_size(),
        dim=128,
        n_layers=4,
        n_heads=4,
        attention_rank=32,
        max_seq_len=args.sequence_length + 1,
        num_experts=4,
        num_slots=64,
        expansion_factor=2,
        mixer_expand=2,
        mixer_kernel_size=4,
        temperature=0.1,
        dropout=0.0,
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
        tie_embeddings=True,
        mixer_backend="mamba2_reference",
        mamba2_d_state=16,
        mamba2_headdim=64,
        mamba2_ngroups=1,
        mamba2_chunk_size=64,
    )


def save_checkpoint(model, optimizer, output_dir: Path, step: int,
                    tokens_seen: int, args: argparse.Namespace,
                    tokenizer_path: Path, metrics_path: Path, *, dataset: dict,
                    data_cursor: dict | None = None) -> None:
    release_sha = dataset.get("release_manifest_sha256")
    tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if release_sha is not None:
        from ada_datasets.release_governance import assert_release_active, register_checkpoint

        assert_release_active(args.release_status_registry, release_sha)
        if tokenizer_sha != dataset["tokenizer_sha256"]:
            raise RuntimeError("Tokenizer differs from the pinned curated release.")
    manifest = {
        **dataset,
        "run_type": ("curated_data_causal_lm_pretraining" if release_sha else
                     "bounded_real_data_causal_lm_pretraining"),
        "tokenizer": str(tokenizer_path),
        "model_config": model.config.to_dict(),
        "model_profile": args.model_profile,
        "step": step,
        "tokens_seen": tokens_seen,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "metrics_file": str(metrics_path),
        "random_seed": args.seed,
        "max_document_chars": args.max_document_chars,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "tokenizer_sha256": tokenizer_sha,
        "precision": "float32",
        "optimizer": "AdamW",
        "checkpoint_scope": "immutable model + tokenizer + optimizer/RNG + manifest; SHA-256 verified",
    }
    if data_cursor is not None:
        manifest["data_cursor"] = data_cursor
        manifest["max_train_tokens"] = args.max_train_tokens
        manifest["targets_per_step"] = args.targets_per_step

    def write_snapshot(stage: Path) -> None:
        # Stage all artifacts before publishing the latest.json pointer.
        model.save_pretrained(stage / "model")
        torch.save(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "python_rng_state": random.getstate(),
                "seed": args.seed,
            },
            stage / "training_state.pt",
        )
        shutil.copyfile(tokenizer_path, stage / "tokenizer.json")
        if hashlib.sha256((stage / "tokenizer.json").read_bytes()).hexdigest() != tokenizer_sha:
            raise RuntimeError("Tokenizer changed while staging checkpoint.")
        (stage / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    committed = commit_checkpoint(
        output_dir, step=step, tokens_seen=tokens_seen, writer=write_snapshot
    )
    if release_sha is not None:
        registration = register_checkpoint(
            args.release_status_registry,
            hashlib.sha256((committed / "checkpoint.json").read_bytes()).hexdigest(),
            release_sha,
        )
        if not registration["active"]:
            raise RuntimeError("Curated release quarantined while publishing the checkpoint.")
    print(f"[checkpoint] step={step} tokens={tokens_seen:,} -> {committed}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    release = load_curated_release(args)
    with nullcontext() if release is None else release:
        return train(args, release)


def train(args: argparse.Namespace, release: TrainingRelease | None) -> int:
    dataset = dataset_metadata(release)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"

    # New checkpoints have a complete immutable snapshot and an atomic pointer.
    # Runs created before this format retain the previous legacy read path.
    published = resolve_committed_checkpoint(args.output_dir) if args.resume else None
    if release is not None and args.resume and published is None:
        raise RuntimeError("Curated pretraining requires a published checkpoint to resume.")
    if published is None:
        validate_checkpoint_directory(args.output_dir, resume=args.resume)
    source_dir = published if published is not None else args.output_dir
    tokenizer_path = source_dir / "tokenizer.json"
    model_dir = source_dir / "model"
    state_path = source_dir / "training_state.pt"
    data_cursor = release.cursor("train") if release is not None else None
    if args.resume:
        manifest_bytes = (source_dir / "run_manifest.json").read_bytes()
        previous_manifest = json.loads(manifest_bytes)
        checkpoint_tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        validate_resume_manifest(previous_manifest, {
            **dataset,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "model_profile": args.model_profile,
            "max_steps": args.max_steps,
            "max_train_tokens": args.max_train_tokens,
            "targets_per_step": args.targets_per_step,
            "random_seed": args.seed,
            "max_document_chars": args.max_document_chars,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "tokenizer_sha256": checkpoint_tokenizer_sha,
        })
        if release is not None:
            if checkpoint_tokenizer_sha != release.tokenizer_sha256:
                raise RuntimeError("Checkpoint tokenizer differs from the curated release.")
            saved_cursor = previous_manifest.get("data_cursor")
            if (not isinstance(saved_cursor, dict) or saved_cursor.keys() != data_cursor.keys()
                    or any(type(value) is not int or value < 0 for value in saved_cursor.values())):
                raise RuntimeError("Curated checkpoint has an invalid training data cursor.")
            data_cursor = saved_cursor

    if release is None:
        tokenizer = train_tokenizer(tokenizer_path, args)
        if tokenizer.get_vocab_size() < 260:
            raise RuntimeError(f"Tokenizer vocabulary unexpectedly small: {tokenizer.get_vocab_size()}")
    else:
        tokenizer_path = release.tokenizer_path
        tokenizer_bytes = tokenizer_path.read_bytes()
        if hashlib.sha256(tokenizer_bytes).hexdigest() != release.tokenizer_sha256:
            raise RuntimeError("Tokenizer differs from the pinned curated release.")
        tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
        if (tokenizer.get_vocab_size() != release.vocab_size
                or any(tokenizer.token_to_id(key) != value for key, value in release.special_tokens.items())):
            raise RuntimeError("Tokenizer vocabulary differs from the curated release contract.")

    config = make_model_config(tokenizer, args)
    if args.resume:
        model = ADAForCausalLM.from_pretrained(model_dir, map_location="cpu")
        if model.config != config:
            raise RuntimeError("Existing checkpoint config differs from this run's configuration.")
        print(f"[model] Resumed model weights from {model_dir}", flush=True)
    else:
        model = ADAForCausalLM(config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    tokens_seen = 0
    if args.resume:
        # Restricted unpickling prevents arbitrary classes/functions in a
        # checkpoint from being executed. Only load trusted local artifacts.
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        start_step = state["step"]
        tokens_seen = state["tokens_seen"]
        if state.get("seed") != args.seed:
            raise RuntimeError("Checkpoint seed differs from --seed")
        if type(start_step) is not int or start_step < 0 or type(tokens_seen) is not int:
            raise RuntimeError("Checkpoint step/token counters are inconsistent")
        max_tokens = start_step * args.batch_size * args.sequence_length
        if release is None:
            if tokens_seen != max_tokens:
                raise RuntimeError("Checkpoint step/token counters are inconsistent")
        else:
            if args.targets_per_step is not None:
                max_tokens = min(start_step * args.targets_per_step, args.max_train_tokens or max_tokens)
            if (not start_step <= tokens_seen <= max_tokens
                    or start_step > args.max_steps
                    or (args.targets_per_step is not None and tokens_seen != max_tokens)
                    or previous_manifest.get("step") != start_step
                    or previous_manifest.get("tokens_seen") != tokens_seen
                    or (args.max_train_tokens is not None and tokens_seen > args.max_train_tokens)):
                raise RuntimeError("Curated checkpoint step/token counters are inconsistent")
            from ada_datasets.release_governance import register_checkpoint

            # Repair an interrupted registration that followed publication.
            checkpoint_sha = hashlib.sha256((published / "checkpoint.json").read_bytes()).hexdigest()
            registration = register_checkpoint(
                args.release_status_registry, checkpoint_sha, release.manifest_sha256,
            )
            if not registration["active"]:
                raise RuntimeError("Curated release quarantined while restoring the checkpoint.")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        random.setstate(state["python_rng_state"])
        print(f"[resume] continuing at step {start_step}, tokens={tokens_seen:,}", flush=True)

    if args.max_train_tokens is None:
        already_complete = start_step >= args.max_steps
    else:
        already_complete = tokens_seen == args.max_train_tokens
    if args.evaluate_test:
        if not already_complete:
            raise RuntimeError("Test evaluation requires a completed training checkpoint.")
        report = {
            "release_manifest_sha256": release.manifest_sha256,
            "checkpoint_manifest_sha256": checkpoint_sha,
            "test_loss": evaluate(model, tokenizer, args, device, release, split="test"),
            "active_targets": release.manifest["splits"]["test"]["active_targets"],
        }
        release.check_active()
        (args.output_dir / "test_evaluation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps(report, indent=2), flush=True)
        return 0
    if already_complete:
        print("Target step count already reached." if release is not None else
              "Target step count already reached. Increase --max-steps to continue.", flush=True)
        return 0
    if start_step >= args.max_steps:
        raise RuntimeError("Checkpoint exhausted the step ceiling before reaching the target budget.")

    if release is None:
        train_batches = iter(infinite_train_batches(tokenizer, args))
        if start_step:
            print(f"[resume] replaying {start_step:,} consumed batches", flush=True)
            train_batches = replay_completed_batches(train_batches, start_step)
    model.train()
    initial_val = evaluate(model, tokenizer, args, device, release)
    session_start_tokens = tokens_seen
    print(
        f"[data] {dataset['dataset_id']}/{dataset['dataset_config']}:{dataset['dataset_split']} | "
        f"validation loss before training={initial_val:.4f}",
        flush=True,
    )
    output_dir = args.output_dir
    if not metrics_path.exists() or not args.resume:
        with metrics_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": start_step, "tokens_seen": tokens_seen,
                "train_loss": None, "validation_loss": initial_val,
                "event": "initial_validation",
            }) + "\n")

    start_time = time.time()
    running_losses: list[float] = []
    last_validation_loss = initial_val
    for step in range(start_step, args.max_steps):
        loss_mask = None
        if release is None:
            batch = next(train_batches)
            batch_tokens = args.batch_size * args.sequence_length
        else:
            try:
                batch, mask, data_cursor = packed_release_batch(
                    release, "train", data_cursor, args.batch_size, repeat=True,
                )
            except StopIteration as exc:
                raise RuntimeError("Curated release produced no training batches.") from exc
            batch_tokens = int(mask.sum())
            if args.targets_per_step is not None and batch_tokens < args.targets_per_step:
                raise RuntimeError("Curated batch has fewer active targets than --targets-per-step.")
            target_count = args.targets_per_step or batch_tokens
            if args.max_train_tokens is not None:
                target_count = min(target_count, args.max_train_tokens - tokens_seen)
            if target_count < batch_tokens:
                mask = mask & (mask.flatten().cumsum(dim=0).reshape(mask.shape) <= target_count)
            batch_tokens = target_count
            loss_mask = mask.to(device, non_blocking=True)
        lr = learning_rate_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        task_loss, total_loss = causal_loss(model, batch.to(device, non_blocking=True), loss_mask)
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {step + 1}")
        optimizer.step()

        tokens_seen += batch_tokens
        running_losses.append(float(task_loss.detach().cpu()))
        completed_step = step + 1
        budget_complete = tokens_seen == args.max_train_tokens
        last_step = completed_step == args.max_steps or budget_complete
        if completed_step % 10 == 0 or completed_step == 1:
            mean_loss = sum(running_losses[-10:]) / len(running_losses[-10:])
            elapsed = time.time() - start_time
            print(
                f"step {completed_step:>5}/{args.max_steps} | "
                f"tokens {tokens_seen:>10,} | loss {mean_loss:.4f} | "
                f"lr {lr:.2e} | {elapsed:.1f}s",
                flush=True,
            )

        should_eval = completed_step % args.eval_every == 0 or last_step
        validation_loss = None
        if should_eval:
            validation_loss = evaluate(model, tokenizer, args, device, release)
            last_validation_loss = validation_loss
            row = {
                "step": completed_step,
                "tokens_seen": tokens_seen,
                "train_loss": sum(running_losses[-args.eval_every:]) / len(running_losses[-args.eval_every:]),
                "validation_loss": validation_loss,
                "learning_rate": lr,
                "elapsed_seconds": round(time.time() - start_time, 2),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"[validation] step={completed_step} "
                f"train={row['train_loss']:.4f} val={validation_loss:.4f}",
                flush=True,
            )

        if completed_step % args.save_every == 0 or last_step:
            save_checkpoint(
                model, optimizer, output_dir, completed_step, tokens_seen,
                args, tokenizer_path, metrics_path,
                dataset=dataset, data_cursor=data_cursor,
            )
        if last_step:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print(
            f"[gpu] {torch.cuda.get_device_name(device)} | "
            f"peak allocated={torch.cuda.max_memory_allocated(device) / 1024**2:.1f} MiB",
            flush=True,
        )
    elapsed_total = time.time() - start_time
    session_tokens = tokens_seen - session_start_tokens
    complete = args.max_train_tokens is None or tokens_seen == args.max_train_tokens
    summary = {
        "status": "COMPLETED" if complete else "INCOMPLETE_TOKEN_BUDGET",
        "run_type": ("curated_data_causal_lm_pretraining" if release is not None else
                     "real_data_causal_lm_pretraining"),
        **{key: dataset[key] for key in ("dataset_id", "dataset_config", "dataset_revision")},
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "float32",
        "parameters_total": model.parameter_count(),
        "steps_this_session": completed_step - start_step,
        "target_steps": args.max_steps,
        "target_training_tokens": args.max_train_tokens,
        "tokens_this_session": session_tokens,
        "tokens_seen_total": tokens_seen,
        "elapsed_seconds": round(elapsed_total, 2),
        "tokens_per_second": round(session_tokens / max(elapsed_total, 1e-9), 2),
        "validation_loss_at_session_start": initial_val,
        "latest_validation_loss": last_validation_loss,
        "peak_cuda_memory_mib": (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
            if device.type == "cuda" else None
        ),
        "checkpoint_dir": str(resolve_committed_checkpoint(output_dir) / "model"),
        "metrics_file": str(metrics_path),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(
        f"[{'complete' if complete else 'step-ceiling'}] steps={completed_step - start_step} "
        f"tokens_this_session={session_tokens:,} "
        f"total_tokens_seen={tokens_seen:,}",
        flush=True,
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
