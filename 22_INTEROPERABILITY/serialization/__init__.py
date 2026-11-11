"""Strict JSON codec rejecting duplicate keys and non-finite numeric values."""
from __future__ import annotations
import json
from .. import ContractSerializer

def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

class StrictJsonCodec:
    def __init__(self, max_bytes: int = 1_000_000):
        if type(max_bytes) is not int or max_bytes < 1: raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
    def encode(self, payload, schema):
        return ContractSerializer.encode(payload, schema, max_bytes=self.max_bytes)
    def decode(self, raw: bytes, schema):
        if not isinstance(raw, bytes) or len(raw) > self.max_bytes: raise ValueError("invalid or oversized JSON")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid number {x}")))
        ContractSerializer.validate(value, schema)
        return value

__all__ = ["StrictJsonCodec"]
