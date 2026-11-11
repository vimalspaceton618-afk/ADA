"""Reliability trial aggregation with Wilson 95% confidence bounds."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True, slots=True)
class ReliabilityReport:
    trials: int
    successes: int
    failures: int
    success_rate: float
    wilson_lower_95: float
    wilson_upper_95: float

def reliability_report(successes: int, failures: int):
    if type(successes) is not int or type(failures) is not int or min(successes, failures) < 0:
        raise ValueError("successes and failures must be non-negative integers")
    n = successes + failures
    if n == 0: raise ValueError("at least one trial required")
    p, z = successes / n, 1.959963984540054
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / denom
    return ReliabilityReport(n, successes, failures, p, max(0.0, center-margin), min(1.0, center+margin))

__all__ = ["ReliabilityReport", "reliability_report"]
