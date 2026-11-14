#!/usr/bin/env python3
"""Fast, reproducible CUDA smoke test for ADA's integrated neural architecture.

This is an execution/gradient/update test on synthetic data. It is not a
training-quality benchmark and does not claim model capability or convergence.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import zipfile
from pathlib import Path

import torch

from ada.neural import ADAConfig, ADAForCausalLM, Mamba2ReferenceMixer, TensorNetAttention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ada_gpu_smoke_test"))
    args = parser.parse_args()
    for name in ("steps", "batch_size", "sequence_length"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2 for causal LM loss")
    return args


def require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"Numerical validation failed: {name} contains NaN/Inf")


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. In Colab select Runtime > Change runtime type > GPU, "
            "then rerun the notebook."
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(2026)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2026)

    # Small enough for a free Colab GPU, but exercises both mixer families.
    config = ADAConfig(
        vocab_size=2048,
        dim=128,
        n_layers=4,
        n_heads=4,
        attention_rank=32,
        max_seq_len=max(128, args.sequence_length),
        num_experts=4,
        num_slots=64,
        expansion_factor=2,
        mixer_expand=2,
        mixer_kernel_size=4,
        temperature=0.1,
        dropout=0.0,
        tie_embeddings=True,
        mixer_backend="mamba2_reference",
        mamba2_d_state=16,
        mamba2_headdim=64,
        mamba2_ngroups=1,
        mamba2_chunk_size=32,
    )
    model = ADAForCausalLM(config).to(device=device, dtype=torch.float32)
    model.train()

    # Assert the requested hybrid architecture was actually constructed.
    mixer_layout = []
    for index, layer in enumerate(model.layers):
        expected = Mamba2ReferenceMixer if index % 2 == 0 else TensorNetAttention
        if not isinstance(layer.mixer, expected):
            raise RuntimeError(
                f"Layer {index} mixer mismatch: expected {expected.__name__}, "
                f"got {type(layer.mixer).__name__}"
            )
        mixer_layout.append(type(layer.mixer).__name__)

    parameter_count = model.parameter_count()
    input_ids = torch.randint(
        low=0, high=config.vocab_size,
        size=(args.batch_size, args.sequence_length),
        device=device, dtype=torch.long,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    required_parameters = {
        "mamba2_ssd_input_projection": model.layers[0].mixer.in_proj.weight,
        "causal_attention_query_projection": model.layers[1].mixer.q_proj.weight,
        "moe_router": model.layers[0].ffn.W_r.weight,
        "energy_memory": next(model.layers[0].ffn.energy_memory.parameters()),
        "language_model_head": model.lm_head.weight,
    }
    mamba_before = model.layers[0].mixer.in_proj.weight.detach().clone()
    losses: list[float] = []
    start = time.perf_counter()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=input_ids)
        if output.loss is None:
            raise RuntimeError("ADA returned no causal language-model loss")
        total_loss = output.loss + 0.01 * output.auxiliary_loss
        require_finite_tensor("logits", output.logits)
        require_finite_tensor("task_loss", output.loss)
        require_finite_tensor("auxiliary_loss", output.auxiliary_loss)
        require_finite_tensor("total_loss", total_loss)
        total_loss.backward()

        if step == 0:
            for name, parameter in required_parameters.items():
                if parameter.grad is None:
                    raise RuntimeError(f"Gradient missing for critical path: {name}")
                require_finite_tensor(f"{name} gradient", parameter.grad)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        require_finite_tensor("gradient norm", grad_norm.reshape(1))
        optimizer.step()
        losses.append(float(total_loss.detach().cpu()))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    update_delta = float(
        (model.layers[0].mixer.in_proj.weight.detach() - mamba_before)
        .abs().max().cpu()
    )
    if not math.isfinite(update_delta) or update_delta <= 0:
        raise RuntimeError("Optimizer update check failed: Mamba-2 parameters did not change")

    model.eval()
    with torch.no_grad():
        inference = model(input_ids)
        require_finite_tensor("inference logits", inference.logits)
        next_token = inference.logits[:, -1, :].argmax(dim=-1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint"
    model.save_pretrained(checkpoint_dir)

    tokens_processed = args.steps * args.batch_size * args.sequence_length
    report = {
        "status": "PASS",
        "test_type": "GPU architecture smoke test on synthetic token data",
        "not_a_claim": [
            "not a convergence or model-quality benchmark",
            "not evidence of trained frontier capability",
            "reference PyTorch SSD, not a fused CUDA/Triton Mamba-2 kernel",
        ],
        "repository": "https://github.com/vimalspaceton618-afk/ADA",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "float32",
        "architecture": config.to_dict(),
        "mixer_layout": mixer_layout,
        "parameters_total": parameter_count,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "synthetic_tokens_processed": tokens_processed,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second_including_backward": round(tokens_processed / max(elapsed, 1e-9), 2),
        "last_total_loss": losses[-1],
        "losses": losses,
        "mamba2_parameter_max_update": update_delta,
        "gradient_checks": "PASS for Mamba-2, causal attention, MoE router, Energy Memory and LM head",
        "inference_logits_shape": list(inference.logits.shape),
        "next_token_ids": next_token.detach().cpu().tolist(),
        "peak_cuda_memory_mb": (
            round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
            if device.type == "cuda" else None
        ),
        "checkpoint": str(checkpoint_dir),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    bundle_path = args.output_dir / "ada_gpu_smoke_test_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(report_path, arcname="report.json")
        for file in checkpoint_dir.rglob("*"):
            if file.is_file():
                bundle.write(file, arcname=f"checkpoint/{file.relative_to(checkpoint_dir)}")

    print("\n========== ADA GPU SMOKE TEST: PASS ==========")
    print(f"Device:          {report['gpu_name'] or device}")
    print(f"PyTorch / CUDA:  {torch.__version__} / {torch.version.cuda}")
    print(f"Architecture:    {', '.join(mixer_layout)}")
    print(f"Parameters:      {parameter_count:,}")
    print(f"Steps:           {args.steps}")
    print(f"Final loss:      {losses[-1]:.6f}")
    print(f"Mamba update:    {update_delta:.8e}")
    print(f"Elapsed:         {elapsed:.2f}s")
    print(f"Throughput:      {report['tokens_per_second_including_backward']:.2f} synthetic tokens/s (train step)")
    print(f"Peak GPU memory: {report['peak_cuda_memory_mb']} MiB")
    print(f"Report:          {report_path}")
    print(f"Bundle:          {bundle_path}")
    print("==============================================")
    print("\nInterpretation: execution, selected gradient paths, optimizer update and inference passed.")
    print("This does not establish convergence, useful learned behavior, or production performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
