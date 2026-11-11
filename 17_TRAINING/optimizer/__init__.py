"""ADA v4.5 AdamW parameter groups and warmup-cosine scheduler."""
from __future__ import annotations
import math

def adamw_parameter_groups(model, *, weight_decay: float = 0.1):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        excluded = (name.endswith(".bias") or name.endswith("norm.weight")
                    or name.endswith("embed.weight") or name == "embed.weight")
        (no_decay if excluded else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]

def build_adamw(model, *, lr: float = 3e-4, weight_decay: float = 0.1,
                betas=(0.9, 0.95), eps: float = 1e-8):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for AdamW") from exc
    return torch.optim.AdamW(adamw_parameter_groups(model, weight_decay=weight_decay),
                             lr=lr, betas=betas, eps=eps)

class WarmupCosine:
    def __init__(self, *, peak_lr=3e-4, min_lr=3e-5, warmup_steps=2000, total_steps=100000):
        if peak_lr <= 0 or min_lr < 0 or min_lr > peak_lr: raise ValueError("invalid learning rates")
        if type(warmup_steps) is not int or type(total_steps) is not int or warmup_steps < 1 or total_steps <= warmup_steps:
            raise ValueError("invalid warmup/total steps")
        self.peak_lr, self.min_lr = peak_lr, min_lr
        self.warmup_steps, self.total_steps = warmup_steps, total_steps
    def __call__(self, step: int) -> float:
        if type(step) is not int or step < 0: raise ValueError("step must be non-negative integer")
        if step < self.warmup_steps: return self.peak_lr * step / self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

__all__ = ["adamw_parameter_groups", "build_adamw", "WarmupCosine"]
