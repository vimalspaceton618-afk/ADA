"""Bounded LRU query-result cache for immutable external-memory reads."""
from __future__ import annotations
from collections import OrderedDict
import hashlib

class MemoryReadCache:
    def __init__(self, capacity: int = 2048):
        if type(capacity) is not int or capacity < 1: raise ValueError("capacity must be positive")
        self.capacity, self._cache = capacity, OrderedDict()
    @staticmethod
    def key(query_bytes: bytes, memory_digest: str):
        if len(memory_digest) != 64: raise ValueError("memory_digest must be SHA-256")
        return hashlib.sha256(memory_digest.encode() + b":" + query_bytes).hexdigest()
    def get(self, key):
        if key not in self._cache: return None
        self._cache.move_to_end(key); return self._cache[key]
    def put(self, key, value):
        self._cache[key] = value; self._cache.move_to_end(key)
        while len(self._cache) > self.capacity: self._cache.popitem(last=False)
    def invalidate(self): self._cache.clear()
    def __len__(self): return len(self._cache)

__all__ = ["MemoryReadCache"]
