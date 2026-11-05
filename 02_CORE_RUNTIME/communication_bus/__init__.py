"""Typed component-to-component communication."""

from .messages import MessageEnvelope, InMemoryMessageBus

__all__ = ["MessageEnvelope", "InMemoryMessageBus"]
