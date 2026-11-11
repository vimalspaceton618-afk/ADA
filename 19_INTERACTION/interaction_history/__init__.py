"""Token-budgeted history windows retaining the newest complete messages."""
from __future__ import annotations
from dataclasses import dataclass
from .. import ConversationMessage

@dataclass(frozen=True, slots=True)
class HistoryWindow:
    messages: tuple[ConversationMessage, ...]
    estimated_tokens: int
    dropped_messages: int

def build_history_window(messages, *, max_tokens: int, token_counter=None):
    if type(max_tokens) is not int or max_tokens < 1: raise ValueError("max_tokens must be positive")
    count = token_counter or (lambda message: max(1, (len(message.content) + 3) // 4))
    selected, total, dropped = [], 0, 0
    for message in reversed(tuple(messages)):
        if not isinstance(message, ConversationMessage): raise TypeError("messages must be ConversationMessage")
        cost = count(message)
        if type(cost) is not int or cost < 0: raise ValueError("token_counter must return non-negative int")
        if total + cost > max_tokens:
            dropped += 1
            continue
        selected.append(message); total += cost
    return HistoryWindow(tuple(reversed(selected)), total, dropped)

__all__ = ["HistoryWindow", "build_history_window"]
