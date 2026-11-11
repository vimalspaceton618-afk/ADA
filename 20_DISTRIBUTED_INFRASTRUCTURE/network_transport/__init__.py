"""Idempotent transport envelopes and injected send/receive backend contract."""
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
    idempotency_key: str

    @classmethod
    def create(cls, sender_id: str, recipient_id: str, payload: Any,
               protocol_version: str = "1", *, idempotency_key: str):
        if any(not isinstance(x, str) or not x.strip()
               for x in (sender_id, recipient_id, protocol_version, idempotency_key)):
            raise ValueError("transport identity and idempotency key are required")
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
        return cls(str(uuid.uuid4()), sender_id, recipient_id, protocol_version,
                   payload, hashlib.sha256(raw).hexdigest(), idempotency_key)

    def verify_payload(self) -> bool:
        raw = json.dumps(self.payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest() == self.payload_digest


class TransportBackend(Protocol):
    def send(self, envelope: TransportEnvelope): ...
    def receive(self, timeout_seconds: float | None = None): ...


class LoopbackTransport:
    """Local test adapter. It does not provide network delivery or encryption."""
    def __init__(self):
        self._messages = []
        self._seen = set()

    def send(self, envelope: TransportEnvelope):
        if not isinstance(envelope, TransportEnvelope):
            raise TypeError("envelope required")
        if not envelope.verify_payload():
            raise ValueError("transport payload digest mismatch")
        if envelope.idempotency_key in self._seen:
            return False
        self._seen.add(envelope.idempotency_key)
        self._messages.append(envelope)
        return True

    def receive(self, timeout_seconds: float | None = None):
        return self._messages.pop(0) if self._messages else None


__all__ = ["TransportEnvelope", "TransportBackend", "LoopbackTransport"]
