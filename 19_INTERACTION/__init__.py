"""Conversation history and explicit user feedback contracts."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    conversation_id: str
    sequence: int
    role: MessageRole
    content: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_id: str | None = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.conversation_id.strip() or not self.content.strip():
            raise ValueError("conversation_id and content are required")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be non-negative integer")
        if not isinstance(self.role, MessageRole):
            raise TypeError("role must be MessageRole")


class ConversationStore:
    """Thread-safe process-local transcript with idempotent turn IDs."""

    def __init__(self, max_messages: int = 100_000):
        if type(max_messages) is not int or max_messages < 1:
            raise ValueError("max_messages must be positive")
        self.max_messages = max_messages
        self._messages = {}
        self._turns = {}
        self._lock = threading.RLock()

    def append(self, conversation_id: str, role: MessageRole, content: str,
               *, turn_id: str | None = None):
        if not conversation_id.strip() or not content.strip():
            raise ValueError("conversation_id and content are required")
        with self._lock:
            if turn_id is not None:
                key = (conversation_id, turn_id, role)
                prior = self._turns.get(key)
                if prior is not None:
                    if (prior.role, prior.content) != (role, content):
                        raise ValueError("turn_id reused with different content")
                    return prior
            count = sum(len(items) for cid, items in self._messages.items() if cid == conversation_id)
            total = sum(len(items) for items in self._messages.values())
            if total >= self.max_messages:
                raise OverflowError("conversation store capacity reached")
            item = ConversationMessage(conversation_id, count, role, content, turn_id=turn_id)
            self._messages.setdefault(conversation_id, []).append(item)
            if turn_id is not None:
                self._turns[(conversation_id, turn_id, role)] = item
            return item

    def get(self, conversation_id: str):
        with self._lock:
            return tuple(self._messages.get(conversation_id, ()))


class FeedbackLabel(str, Enum):
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    INCORRECT = "incorrect"
    UNSAFE = "unsafe"


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    conversation_id: str
    message_id: str
    label: FeedbackLabel
    comment: str = ""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if not self.conversation_id.strip() or not self.message_id.strip():
            raise ValueError("conversation_id and message_id are required")
        if not isinstance(self.label, FeedbackLabel):
            raise TypeError("label must be FeedbackLabel")


class FeedbackStore:
    def __init__(self):
        self._items = []
        self._lock = threading.RLock()

    def add(self, feedback: FeedbackRecord):
        if not isinstance(feedback, FeedbackRecord):
            raise TypeError("feedback must be FeedbackRecord")
        with self._lock:
            self._items.append(feedback)
        return feedback

    @property
    def records(self):
        with self._lock:
            return tuple(self._items)


__all__ = [
    "MessageRole", "ConversationMessage", "ConversationStore",
    "FeedbackLabel", "FeedbackRecord", "FeedbackStore",
]
