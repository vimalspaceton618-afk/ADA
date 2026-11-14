#!/usr/bin/env python3
"""Stream FineWeb-Edu and pretrain ADAForCausalLM from randomly initialized weights.

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
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from ada.neural import ADAConfig, ADAForCausalLM

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_SPLIT = "train"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DATASET_LICENSE = "ODC-By 1.0 (dataset declaration); Common Crawl terms also apply"
DATASET_CARD = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu"
DATASET_PAPER = "https://arxiv.org/abs/2406.17557"
SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sequence-length", type=int, default=256)
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
    a = p.parse_args()
    for name in ("max_steps", "batch_size", "sequence_length", "vocab_size",
                 "tokenizer_docs", "eval_every", "eval_batches", "save_every",
                 "max_document_chars"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if a.sequence_length < 2 or a.vocab_size < len(SPECIAL_TOKENS) + 256:
        p.error("sequence length must be >=2 and vocab size >=260")
    if a.learning_rate <= 0 or a.weight_decay < 0 or a.warmup_steps < 0:
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
    if eos_id is None:
        raise RuntimeError("tokenizer is missing <|eos|>")
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


def causal_loss(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(input_ids)
    logits = output.logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    task_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
    total_loss = task_loss + 0.01 * output.auxiliary_loss
    return task_loss, total_loss


@torch.no_grad()
def evaluate(model, tokenizer: Tokenizer, args: argparse.Namespace,
             device: torch.device) -> float:
    model.eval()
    values = []
    for batch in finite_batches(tokenizer, args):
        batch = batch.to(device, non_blocking=True)
        task_loss, _ = causal_loss(model, batch)
        values.append(float(task_loss.cpu()))
    if not values:
        raise RuntimeError("Validation split produced no batches.")
    model.train()
    return sum(values) / len(values)


def learning_rate_at(step: int, args: argparse.Namespace) -> float:
    if args.warmup_steps and step < args.warmup_steps:
        return args.learning_rate * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(model, optimizer, output_dir: Path, step: int,
                    tokens_seen: int, args: argparse.Namespace,
                    tokenizer_path: Path, metrics_path: Path) -> None:
    model_dir = output_dir / "model"
    model.save_pretrained(model_dir)
    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng_state": random.getstate(),
            "seed": args.seed,
        },
        output_dir / "training_state.pt",
    )
    manifest = {
        "run_type": "bounded_real_data_causal_lm_pretraining",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "dataset_card": DATASET_CARD,
        "dataset_paper": DATASET_PAPER,
        "split_method": "stable SHA-256 bucket from record id/url/text; bucket 0 validation, buckets 1-99 train",
        "filters": {"language": "en", "int_score_min": 3, "minimum_text_chars": 200},
        "tokenizer": str(tokenizer_path),
        "model_config": model.config.to_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "metrics_file": str(metrics_path),
        "random_seed": args.seed,
        "precision": "float32",
        "optimizer": "AdamW",
        "checkpoint_scope": "model weights + optimizer state + RNG states",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[checkpoint] step={step} tokens={tokens_seen:,} -> {output_dir}", flush=True)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.output_dir / "tokenizer.json"
    metrics_path = args.output_dir / "metrics.jsonl"

    model_dir = args.output_dir / "model"
    state_path = args.output_dir / "training_state.pt"
    if not args.resume and (model_dir / "config.json").exists():
        raise RuntimeError(
            f"A model checkpoint already exists in {args.output_dir}. "
            "Set --resume to continue it, or choose a new --output-dir to start fresh."
        )

    tokenizer = train_tokenizer(tokenizer_path, args)
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size < 260:
        raise RuntimeError(f"Tokenizer vocabulary unexpectedly small: {vocab_size}")

    config = ADAConfig(
        vocab_size=vocab_size,
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
    if args.resume and (model_dir / "config.json").exists():
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
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        tokens_seen = int(state["tokens_seen"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        random.setstate(state["python_rng_state"])
        print(f"[resume] continuing at step {start_step}, tokens={tokens_seen:,}", flush=True)

    if start_step >= args.max_steps:
        print("Target step count already reached. Increase --max-steps to continue.", flush=True)
        return 0

    train_batches = iter(infinite_train_batches(tokenizer, args))
    model.train()
    initial_val = evaluate(model, tokenizer, args, device)
    print(
        f"[data] {DATASET_ID}/{DATASET_CONFIG}:{DATASET_SPLIT} | "
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
    for step in range(start_step, args.max_steps):
        batch = next(train_batches).to(device, non_blocking=True)
        lr = learning_rate_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        task_loss, total_loss = causal_loss(model, batch)
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {step + 1}")
        optimizer.step()

        tokens_seen += args.batch_size * args.sequence_length
        running_losses.append(float(task_loss.detach().cpu()))
        completed_step = step + 1
        if completed_step % 10 == 0 or completed_step == 1:
            mean_loss = sum(running_losses[-10:]) / len(running_losses[-10:])
            elapsed = time.time() - start_time
            print(
                f"step {completed_step:>5}/{args.max_steps} | "
                f"tokens {tokens_seen:>10,} | loss {mean_loss:.4f} | "
                f"lr {lr:.2e} | {elapsed:.1f}s",
                flush=True,
            )

        should_eval = completed_step % args.eval_every == 0 or completed_step == args.max_steps
        validation_loss = None
        if should_eval:
            validation_loss = evaluate(model, tokenizer, args, device)
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

        if completed_step % args.save_every == 0 or completed_step == args.max_steps:
            save_checkpoint(
                model, optimizer, output_dir, completed_step, tokens_seen,
                args, tokenizer_path, metrics_path,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print(
            f"[gpu] {torch.cuda.get_device_name(device)} | "
            f"peak allocated={torch.cuda.max_memory_allocated(device) / 1024**2:.1f} MiB",
            flush=True,
        )
    print(
        f"[complete] steps={args.max_steps - start_step} "
        f"tokens_this_session={(args.max_steps - start_step) * args.batch_size * args.sequence_length:,} "
        f"total_tokens_seen={tokens_seen:,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
