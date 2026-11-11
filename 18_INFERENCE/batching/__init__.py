"""Length-aware token-budget batching without splitting packed examples."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class BatchItem:
    item_id: str
    token_count: int
    payload: object
    def __post_init__(self):
        if not self.item_id.strip() or type(self.token_count) is not int or self.token_count < 1:
            raise ValueError("item ID and positive token_count required")

class TokenBudgetBatcher:
    def __init__(self, *, max_tokens: int, max_items: int):
        if max_tokens < 1 or max_items < 1: raise ValueError("batch limits must be positive")
        self.max_tokens, self.max_items = max_tokens, max_items
    def pack(self, items):
        batches, current, used = [], [], 0
        for item in sorted(items, key=lambda x: (x.token_count, x.item_id)):
            if not isinstance(item, BatchItem): raise TypeError("items must be BatchItem")
            if item.token_count > self.max_tokens: raise ValueError(f"item {item.item_id} exceeds batch token budget")
            if current and (used + item.token_count > self.max_tokens or len(current) >= self.max_items):
                batches.append(tuple(current)); current, used = [], 0
            current.append(item); used += item.token_count
        if current: batches.append(tuple(current))
        return tuple(batches)

__all__ = ["BatchItem", "TokenBudgetBatcher"]
