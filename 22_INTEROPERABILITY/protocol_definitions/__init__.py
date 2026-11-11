"""Protocol envelope with stable identity, version and payload digest."""
from __future__ import annotations
import hashlib, json, uuid
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class ProtocolEnvelope:
    message_id: str
    protocol: str
    version: str
    trace_id: str
    payload: Any
    payload_digest: str
    idempotency_key: str
    @classmethod
    def build(cls, protocol: str, version: str, trace_id: str, payload: Any, idempotency_key: str):
        if any(not x.strip() for x in (protocol, version, trace_id, idempotency_key)):
            raise ValueError("protocol envelope identity fields required")
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode()
        return cls(str(uuid.uuid4()), protocol, version, trace_id, payload,
                   hashlib.sha256(raw).hexdigest(), idempotency_key)

__all__ = ["ProtocolEnvelope"]
