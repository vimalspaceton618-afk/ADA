"""Content-addressed model artifact manifests and verification."""
from __future__ import annotations
import hashlib, os
from dataclasses import dataclass
from pathlib import Path

def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> tuple[str, int]:
    if chunk_size < 1: raise ValueError("chunk_size must be positive")
    h, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            h.update(chunk); size += len(chunk)
    return h.hexdigest(), size

@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    artifact_id: str
    model_id: str
    version: str
    sha256: str
    size_bytes: int
    architecture_version: str
    def __post_init__(self):
        for name in ("artifact_id", "model_id", "version", "architecture_version"):
            if not getattr(self, name).strip(): raise ValueError(f"{name} is required")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("sha256 must be lowercase SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")

def verify_artifact(path: str | os.PathLike[str], manifest: ArtifactManifest) -> bool:
    if not isinstance(manifest, ArtifactManifest): raise TypeError("manifest must be ArtifactManifest")
    actual, size = sha256_file(path)
    return actual == manifest.sha256 and size == manifest.size_bytes

__all__ = ["sha256_file", "ArtifactManifest", "verify_artifact"]
