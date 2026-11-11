"""Backup creation, integrity verification and restore adapter contract."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, os, tempfile
from pathlib import Path
from typing import Protocol

@dataclass(frozen=True, slots=True)
class BackupArtifact:
    backup_id: str
    path: str
    digest: str
    size_bytes: int
    source_id: str
    def __post_init__(self):
        if any(not x.strip() for x in (self.backup_id, self.path, self.source_id)): raise ValueError("backup metadata required")
        if len(self.digest) != 64: raise ValueError("backup digest must be SHA-256")
        if self.size_bytes < 0: raise ValueError("size_bytes must be non-negative")

class BackupAdapter(Protocol):
    def restore(self, artifact: BackupArtifact, destination: str | os.PathLike[str]) -> None: ...

class LocalBackupManager:
    def create(self, source: str | os.PathLike[str], destination_dir: str | os.PathLike[str], *, backup_id: str):
        source, root = Path(source), Path(destination_dir)
        if not source.is_file(): raise FileNotFoundError(source)
        root.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        target = root / f"{backup_id}-{digest[:12]}.bak"
        fd, tmp = tempfile.mkstemp(dir=root, prefix=".backup-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        return BackupArtifact(backup_id, str(target), digest, len(data), str(source))
    def verify(self, artifact: BackupArtifact):
        data = Path(artifact.path).read_bytes()
        return len(data) == artifact.size_bytes and hashlib.sha256(data).hexdigest() == artifact.digest
    def restore(self, artifact: BackupArtifact, destination: str | os.PathLike[str]):
        if not self.verify(artifact): raise ValueError("backup integrity verification failed")
        target = Path(destination); target.parent.mkdir(parents=True, exist_ok=True)
        data = Path(artifact.path).read_bytes()
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".restore-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        return target

__all__ = ["BackupArtifact", "BackupAdapter", "LocalBackupManager"]
