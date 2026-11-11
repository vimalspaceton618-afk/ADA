"""Atomic local content-addressed object store."""
from __future__ import annotations
import hashlib, os, tempfile
from pathlib import Path

class ContentAddressedStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def put(self, content: bytes) -> str:
        if not isinstance(content, bytes): raise TypeError("content must be bytes")
        digest = hashlib.sha256(content).hexdigest()
        target = self.root / digest
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest: raise ValueError("stored object corruption")
            return digest
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".object-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        return digest
    def get(self, digest: str) -> bytes:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid content digest")
        content = (self.root / digest).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest: raise ValueError("stored object integrity failure")
        return content

__all__ = ["ContentAddressedStore"]
