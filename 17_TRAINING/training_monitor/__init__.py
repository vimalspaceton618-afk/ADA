"""Finite training metric validation and expert-utilisation diagnostics."""
from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class TrainingHealth:
    finite: bool
    loss_mean: float | None
    loss_increasing: bool
    expert_entropy: float | None
    warnings: tuple[str, ...]

class TrainingMonitor:
    def __init__(self, window: int = 100):
        if type(window) is not int or window < 2: raise ValueError("window must be >=2")
        self.losses = deque(maxlen=window)
    def record(self, loss: float, expert_load=None) -> TrainingHealth:
        finite = isinstance(loss, (int, float)) and math.isfinite(float(loss))
        warnings = []
        if not finite:
            warnings.append("non-finite loss")
        else:
            self.losses.append(float(loss))
        mean = sum(self.losses) / len(self.losses) if self.losses else None
        values = list(self.losses)
        increasing = len(values) >= 4 and sum(values[-2:]) / 2 > sum(values[:2]) / 2
        if increasing: warnings.append("loss window increased")
        entropy = None
        if expert_load is not None:
            values = [float(x) for x in expert_load]
            total = sum(values)
            if any(x < 0 or not math.isfinite(x) for x in values) or total <= 0:
                warnings.append("invalid expert load distribution")
            else:
                probs = [x / total for x in values if x > 0]
                entropy = -sum(x * math.log(x) for x in probs)
        return TrainingHealth(finite, mean, increasing, entropy, tuple(warnings))

__all__ = ["TrainingHealth", "TrainingMonitor"]
