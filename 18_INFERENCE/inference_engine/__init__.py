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
        # A fresh runner state has never seen the prompt: prefill with all
        # prompt tokens. Passing only the final token loses all prior context.
        # Callers supplying an already-prefilled state may continue one token.
        current = input_ids if state is None else input_ids[:, -1:]
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                next_logits, state = self.runner.step(current, state)
                if not isinstance(next_logits, torch.Tensor) or next_logits.ndim != 3:
                    raise ValueError("runner.step must return logits shaped [B,T,V]")
                if next_logits.shape[:2] != current.shape or next_logits.shape[-1] < 1:
                    raise ValueError("runner logits shape must match current tokens [B,T,V]")
                logits = next_logits[:, -1, :].float() / config.temperature
                if config.top_k:
                    k = min(config.top_k, logits.size(-1))
                    values, indices = torch.topk(logits, k, dim=-1)
                    logits = torch.full_like(logits, float("-inf")).scatter_(1, indices, values)
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                if config.eos_token_id is not None:
                    # Once a batch row finishes, keep it finished; do not
                    # sample unrelated tokens while other rows keep decoding.
                    next_token = torch.where(
                        finished.unsqueeze(1),
                        torch.full_like(next_token, config.eos_token_id),
                        next_token,
                    )
                    finished |= next_token[:, 0].eq(config.eos_token_id)
                generated = torch.cat((generated, next_token), dim=1)
                if on_token is not None: on_token(next_token)
                current = next_token
                if config.eos_token_id is not None and bool(finished.all()):
                    break
        return generated, state

__all__ = ["GenerationConfig", "IncrementalRunner", "AutoregressiveGenerator"]
