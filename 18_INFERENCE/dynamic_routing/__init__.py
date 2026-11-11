"""Auxiliary-loss-free top-k expert routing with online bias correction."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ExpertRouting:
    indices: object
    weights: object
    probabilities: object

class BiasCorrectedTopKRouter:
    def __init__(self, dim: int, n_experts: int = 64, top_k: int = 6, bias_lr: float = 0.001):
        if dim < 1 or n_experts < 1 or not 1 <= top_k <= n_experts or bias_lr <= 0:
            raise ValueError("invalid routing configuration")
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc: raise RuntimeError("PyTorch required for expert routing") from exc
        self._linear = nn.Linear(dim, n_experts)
        self._bias = torch.zeros(n_experts, dtype=torch.float32)
        self.n_experts, self.top_k, self.bias_lr = n_experts, top_k, bias_lr
    @property
    def module(self): return self._linear
    @property
    def expert_bias(self): return self._bias
    def forward(self, x, *, training: bool):
        import torch
        logits = self._linear(x)
        if logits.ndim != 2: raise ValueError("router input must be flattened [tokens,dim]")
        selected = torch.topk(logits + self._bias.to(logits.device), self.top_k, dim=-1).indices
        unbiased = logits.gather(1, selected)
        weights = torch.softmax(unbiased.float(), dim=-1).to(logits.dtype)
        probs = torch.softmax(logits.float(), dim=-1)
        if training:
            with torch.no_grad():
                counts = torch.bincount(selected.reshape(-1), minlength=self.n_experts).float()
                load = counts / counts.sum().clamp_min(1.0)
                self._bias.add_(self.bias_lr * (1.0 / self.n_experts - load.cpu()))
        return ExpertRouting(selected, weights, probs)
    @staticmethod
    def dispatch(experts, tokens, routing: ExpertRouting):
        import torch
        if tokens.ndim != 2: raise ValueError("tokens must be [N,D]")
        output = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(experts):
            token_idx, slot_idx = (routing.indices == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0: continue
            values = expert(tokens.index_select(0, token_idx))
            weighted = values * routing.weights[token_idx, slot_idx].unsqueeze(-1).to(values.dtype)
            output.index_add_(0, token_idx, weighted)
        return output

__all__ = ["ExpertRouting", "BiasCorrectedTopKRouter"]
