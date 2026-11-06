"""Numerically stable entropy summaries for non-negative score distributions."""
from __future__ import annotations

import math


def entropy(scores):
    values = tuple(scores)
    if not values:
        raise ValueError("scores must be non-empty")
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("scores must be numeric")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("scores must be finite and non-negative") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("scores must be finite and non-negative")
        normalized.append(value)
    scale = max(normalized)
    if scale <= 0:
        raise ValueError("score sum must be positive")
    scaled = [value / scale for value in normalized]
    total = math.fsum(scaled)
    return -math.fsum((value / total) * math.log(value / total)
                      for value in scaled if value > 0)


def normalized_entropy(scores):
    values = tuple(scores)
    if len(values) <= 1:
        return 0.0
    result = entropy(values) / math.log(len(values))
    return min(1.0, max(0.0, result))


__all__ = ["entropy", "normalized_entropy"]
