"""Digest-verified, strict PyTorch state-dict loading."""
from __future__ import annotations
from pathlib import Path
import hashlib

def load_state_dict_verified(model, path, expected_sha256: str, *, map_location="cpu"):
    if len(expected_sha256) != 64: raise ValueError("expected_sha256 must be SHA-256")
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256: raise ValueError("model artifact digest mismatch")
    try:
        import torch
    except ImportError as exc: raise RuntimeError("PyTorch is required to load this artifact") from exc
    state = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict): raise ValueError("checkpoint must contain a state dictionary")
    model.load_state_dict(state, strict=True)
    return model

__all__ = ["load_state_dict_verified"]
