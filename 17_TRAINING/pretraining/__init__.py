"""Causal token batch validation and training token-budget accounting."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class CausalBatch:
    input_ids: tuple[tuple[int, ...], ...]
    loss_mask: tuple[tuple[bool, ...], ...]
    def __post_init__(self):
        if not self.input_ids or len(self.input_ids) != len(self.loss_mask): raise ValueError("batch dimensions mismatch")
        width = len(self.input_ids[0])
        if width < 2 or any(len(x) != width for x in self.input_ids + self.loss_mask):
            raise ValueError("sequences must have equal length >=2")
        if any(type(token) is not int or token < 0 for seq in self.input_ids for token in seq):
            raise ValueError("token IDs must be non-negative integers")
        if any(type(mask) is not bool for seq in self.loss_mask for mask in seq):
            raise ValueError("loss mask must contain booleans")

class TokenBudget:
    def __init__(self, target_tokens: int):
        if type(target_tokens) is not int or target_tokens < 1: raise ValueError("target_tokens must be positive")
        self.target_tokens, self.consumed = target_tokens, 0
    def add(self, tokens: int):
        if type(tokens) is not int or tokens < 0: raise ValueError("tokens must be non-negative")
        self.consumed += tokens
        return self.remaining
    @property
    def remaining(self): return max(0, self.target_tokens - self.consumed)
    @property
    def complete(self): return self.consumed >= self.target_tokens

__all__ = ["CausalBatch", "TokenBudget"]
