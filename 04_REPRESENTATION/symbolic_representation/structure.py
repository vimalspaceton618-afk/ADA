"""Shape-only symbolic conversion for structured perception observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_shape_tree(value: Any, *, depth: int = 0, max_depth: int = 12) -> None:
    """Validate a JSON-like structural shape without inventing field values."""
    if depth > max_depth:
        raise ValueError("symbolic shape exceeds maximum depth")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("shape object keys must be strings")
            validate_shape_tree(child, depth=depth + 1, max_depth=max_depth)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            validate_shape_tree(child, depth=depth + 1, max_depth=max_depth)
        return
    raise TypeError(f"unsupported symbolic shape value: {type(value).__name__}")


def shape_payload(attributes: Mapping[str, Any]) -> dict[str, Any]:
    shape = attributes.get("shape")
    if not isinstance(shape, Mapping):
        raise ValueError("structured observation is missing a shape mapping")
    validate_shape_tree(shape)
    keys = attributes.get("top_level_keys", ())
    count = attributes.get("top_level_field_count")
    if type(count) is not int or count < 0:
        raise ValueError("structured observation has an invalid top-level field count")
    if not isinstance(keys, (list, tuple)):
        raise TypeError("top_level_keys must be a sequence")
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("top_level_keys must contain strings")
    return {
        "structure": shape,
        "top_level_keys": list(keys),
        "top_level_field_count": count,
        "interpretation": "shape_only_no_semantic_interpretation",
    }
