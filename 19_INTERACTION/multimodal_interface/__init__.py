"""Validated modality envelope metadata; decoding is delegated to modality adapters."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from enum import Enum

class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

@dataclass(frozen=True, slots=True)
class ModalityPayload:
    modality: Modality
    media_type: str
    content_digest: str
    byte_length: int
    source_id: str
    def __post_init__(self):
        if not isinstance(self.modality, Modality): raise TypeError("modality must be Modality")
        if not self.media_type.strip() or not self.source_id.strip(): raise ValueError("media type and source required")
        if len(self.content_digest) != 64 or any(c not in "0123456789abcdef" for c in self.content_digest):
            raise ValueError("content_digest must be SHA-256")
        if type(self.byte_length) is not int or self.byte_length < 0: raise ValueError("byte_length must be non-negative")

def describe_payload(modality: Modality, content: bytes, *, media_type: str, source_id: str):
    if not isinstance(content, bytes): raise TypeError("content must be bytes")
    return ModalityPayload(modality, media_type, hashlib.sha256(content).hexdigest(), len(content), source_id)

__all__ = ["Modality", "ModalityPayload", "describe_payload"]
