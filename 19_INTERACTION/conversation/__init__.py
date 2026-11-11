"""Conversation turn assembly with ordering and tool-message invariants."""
from __future__ import annotations
from dataclasses import dataclass
from .. import ConversationMessage, MessageRole

@dataclass(frozen=True, slots=True)
class ConversationTurn:
    conversation_id: str
    turn_id: str
    messages: tuple[ConversationMessage, ...]
    def __post_init__(self):
        if not self.conversation_id.strip() or not self.turn_id.strip(): raise ValueError("conversation and turn IDs required")
        if not self.messages: raise ValueError("turn must contain messages")
        if any(x.conversation_id != self.conversation_id for x in self.messages): raise ValueError("messages cross conversations")
        if any(x.turn_id not in {None, self.turn_id} for x in self.messages): raise ValueError("message belongs to another turn")
        seq = [x.sequence for x in self.messages]
        if seq != sorted(seq) or len(seq) != len(set(seq)): raise ValueError("turn messages must have ordered unique sequence numbers")
        if self.messages[0].role is not MessageRole.USER: raise ValueError("turn must begin with a user message")
    @property
    def final_message(self):
        return self.messages[-1]

def assemble_turn(conversation_id: str, turn_id: str, messages):
    return ConversationTurn(conversation_id, turn_id, tuple(messages))

__all__ = ["ConversationTurn", "assemble_turn"]
