"""Transport envelope and injected send/receive backend contract."""
from __future__ import annotations
import hashlib, json, uuid
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True, slots=True)
class TransportEnvelope:
    message_id: str
    sender_id: str
    recipient_id: str
    protocol_version: str
    payload: Any
    payload_digest: str
    @classmethod
    def create(cls, sender_id, recipient_id, payload, protocol_version="1"):
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode()
        return cls(str(uuid.uuid4()), sender_id, recipient_id, protocol_version, payload, hashlib.sha256(raw).hexdigest())

class TransportBackend(Protocol):
    def send(self, envelope: TransportEnvelope): ...
    def receive(self, timeout_seconds: float | None = None): ...

class LoopbackTransport:
    def __init__(self): self._messages = []
    def send(self, envelope: TransportEnvelope):
        if not isinstance(envelope, TransportEnvelope): raise TypeError("envelope required")
        self._messages.append(envelope)
    def receive(self, timeout_seconds: float | None = None):
        return self._messages.pop(0) if self._messages else None

__all__ = ["TransportEnvelope", "TransportBackend", "LoopbackTransport"]
