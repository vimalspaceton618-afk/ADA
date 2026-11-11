"""Deterministic experiment environment capture and config hashing."""
from __future__ import annotations
import hashlib, json, os, platform, random
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True, slots=True)
class ReproducibilityManifest:
    seed: int
    python_version: str
    platform: str
    config_digest: str
    environment: tuple[tuple[str, str], ...]

def canonical_digest(config: Mapping[str, Any]):
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

def seed_python(seed: int):
    if type(seed) is not int: raise TypeError("seed must be int")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError: pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError: pass

def capture(seed: int, config: Mapping[str, Any], env_keys=()):
    values = tuple(sorted((key, os.environ.get(key, "")) for key in env_keys))
    return ReproducibilityManifest(seed, platform.python_version(), platform.platform(), canonical_digest(config), values)

__all__ = ["ReproducibilityManifest", "canonical_digest", "seed_python", "capture"]
