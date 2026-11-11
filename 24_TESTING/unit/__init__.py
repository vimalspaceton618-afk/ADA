"""Reusable unit assertion primitives."""
from __future__ import annotations
import math

def assert_finite(value, name="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AssertionError(f"{name} must be finite numeric")
    return True

def assert_digest(value, name="digest"):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise AssertionError(f"{name} must be lowercase SHA-256")
    return True

__all__ = ["assert_finite", "assert_digest"]
