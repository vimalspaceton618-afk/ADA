"""Optional representation-alignment objective for specialist experiments.

This is an extension experiment; it is not part of ADA v4.5's two-term
language-model objective unless a study explicitly adds it.
"""
from __future__ import annotations

def symmetric_contrastive_loss(left, right, *, temperature: float = 0.07):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for representation alignment") from exc
    if left.ndim != 2 or right.shape != left.shape: raise ValueError("representations must share [B,D] shape")
    if left.size(0) < 2 or temperature <= 0: raise ValueError("batch must have at least two pairs and positive temperature")
    left = F.normalize(left.float(), dim=-1); right = F.normalize(right.float(), dim=-1)
    logits = left @ right.T / temperature
    labels = torch.arange(left.size(0), device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

__all__ = ["symmetric_contrastive_loss"]
