"""Autoregressive generation loop driven by an explicit incremental runner."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    eos_token_id: int | None = None
    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1: raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0: raise ValueError("temperature must be positive")
        if type(self.top_k) is not int or self.top_k < 0: raise ValueError("top_k must be non-negative")

class IncrementalRunner:
    """Adapter contract: step(token_ids, state) -> (next_logits, next_state)."""
    def step(self, token_ids, state): raise NotImplementedError

class AutoregressiveGenerator:
    def __init__(self, runner: IncrementalRunner):
        if not callable(getattr(runner, "step", None)): raise TypeError("runner must provide step()")
        self.runner = runner
    def generate(self, input_ids, *, config: GenerationConfig, state=None, on_token=None):
        import torch
        if input_ids.ndim != 2 or input_ids.shape[1] < 1: raise ValueError("input_ids must be [B,T]")
        generated = input_ids
        current = input_ids[:, -1:]
        for _ in range(config.max_new_tokens):
            logits, state = self.runner.step(current, state)
            logits = logits[:, -1, :].float() / config.temperature
            if config.top_k:
                k = min(config.top_k, logits.size(-1))
                values, indices = torch.topk(logits, k, dim=-1)
                logits = torch.full_like(logits, float("-inf")).scatter_(1, indices, values)
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            generated = torch.cat((generated, next_token), dim=1)
            if on_token is not None: on_token(next_token)
            current = next_token
            if config.eos_token_id is not None and bool((next_token == config.eos_token_id).all()): break
        return generated, state

__all__ = ["GenerationConfig", "IncrementalRunner", "AutoregressiveGenerator"]
