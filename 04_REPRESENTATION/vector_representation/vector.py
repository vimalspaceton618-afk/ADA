"""Validated numeric vector payload helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable


def validate_vector(
    values: Iterable[float],
    *,
    expected_dimension: int | None = None,
) -> tuple[tuple[float, ...], float]:
    """Return finite float values and their L2 norm without renormalizing."""
    if isinstance(values, (str, bytes)):
        raise TypeError("vector values must be a numeric iterable")
    try:
        vector = tuple(values)
    except TypeError as exc:
        raise TypeError("vector values must be iterable") from exc
    if not vector:
        raise ValueError("vector must contain at least one value")
    if expected_dimension is not None:
        if type(expected_dimension) is not int or expected_dimension < 1:
            raise ValueError("expected_dimension must be a positive integer")
        if len(vector) != expected_dimension:
            raise ValueError(
                f"vector dimension mismatch ({len(vector)} != {expected_dimension})"
            )
    normalized_values: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"vector[{index}] must be numeric")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"vector[{index}] must be finite")
        normalized_values.append(converted)
    l2_norm = math.sqrt(sum(value * value for value in normalized_values))
    return tuple(normalized_values), l2_norm
