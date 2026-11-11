"""Position-sorted Mixture-of-Depths routing for causal SSM blocks."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class MoDRouting:
    indices: object
    weights: object
    capacity: int

class MoDTokenRouter:
    def __init__(self, dim: int, *, capacity: float = 0.5):
        if dim < 1 or not 0 < capacity <= 1: raise ValueError("invalid dimension or capacity")
        try:
            import torch.nn as nn
        except ImportError as exc: raise RuntimeError("PyTorch required for MoD router") from exc
        self._module = nn.Linear(dim, 1)
        self.capacity = capacity
    @property
    def module(self): return self._module
    def forward(self, x, block):
        import torch
        if x.ndim != 3: raise ValueError("x must be [B,T,D]")
        batch, length, dim = x.shape
        k = max(1, min(length, int(self.capacity * length)))
        scores = torch.sigmoid(self._module(x).squeeze(-1))
        indices = torch.topk(scores, k, dim=1, sorted=False).indices
        indices, order = indices.sort(dim=1)
        weights = torch.sigmoid(scores.gather(1, indices))
        gathered = x.gather(1, indices.unsqueeze(-1).expand(batch, k, dim))
        transformed = block(gathered)
        if transformed.shape != gathered.shape: raise ValueError("MoD block must preserve [B,k,D]")
        output = x.clone()
        output.scatter_(1, indices.unsqueeze(-1).expand(batch, k, dim),
                        gathered + weights.unsqueeze(-1) * transformed)
        return output, MoDRouting(indices, weights, k)

__all__ = ["MoDRouting", "MoDTokenRouter"]
