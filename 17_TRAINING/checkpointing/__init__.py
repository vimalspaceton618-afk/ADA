"""Atomic checkpoint payload storage with digest manifests."""
from __future__ import annotations
import hashlib, os, tempfile
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class CheckpointFile:
    path: str
    sha256: str
    size_bytes: int

class AtomicCheckpointStore:
    def __init__(self, directory: str | os.PathLike[str]):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
    def save_bytes(self, name: str, payload: bytes) -> CheckpointFile:
        if not name or Path(name).name != name: raise ValueError("checkpoint name must be a basename")
        if not isinstance(payload, bytes): raise TypeError("payload must be bytes")
        fd, tmp = tempfile.mkstemp(prefix=".checkpoint-", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            target = self.directory / name
            os.replace(tmp, target)
            digest = hashlib.sha256(payload).hexdigest()
            return CheckpointFile(str(target), digest, len(payload))
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    def load_bytes(self, checkpoint: CheckpointFile) -> bytes:
        if not isinstance(checkpoint, CheckpointFile): raise TypeError("checkpoint must be CheckpointFile")
        data = Path(checkpoint.path).read_bytes()
        if len(data) != checkpoint.size_bytes or hashlib.sha256(data).hexdigest() != checkpoint.sha256:
            raise ValueError("checkpoint integrity verification failed")
        return data

__all__ = ["CheckpointFile", "AtomicCheckpointStore"]
