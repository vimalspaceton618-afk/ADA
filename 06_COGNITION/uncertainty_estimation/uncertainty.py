"""Uncertainty summaries for normalized candidate score distributions."""
from __future__ import annotations
import math

def entropy(scores):
    values = tuple(float(x) for x in scores)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("scores must be a non-empty sequence of finite non-negative values")
    total = sum(values)
    if total <= 0:
        raise ValueError("score sum must be positive")
    probs = [x / total for x in values if x > 0]
    return -sum(p * math.log(p) for p in probs)

def normalized_entropy(scores):
    values = tuple(scores)
    if len(values) <= 1:
        return 0.0
    return entropy(values) / math.log(len(values))
