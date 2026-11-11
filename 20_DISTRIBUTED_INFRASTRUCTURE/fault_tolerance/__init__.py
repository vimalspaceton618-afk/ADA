"""Bounded retry policy and circuit breaker for transient distributed failures."""
from __future__ import annotations
import time
from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 5.0
    def __post_init__(self):
        if type(self.max_attempts) is not int or self.max_attempts < 1: raise ValueError("max_attempts must be positive")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < self.base_delay_seconds: raise ValueError("invalid delay bounds")
    def delay(self, failed_attempt: int):
        if type(failed_attempt) is not int or failed_attempt < 1: raise ValueError("failed_attempt must be >=1")
        return min(self.max_delay_seconds, self.base_delay_seconds * (2 ** (failed_attempt - 1)))

class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, *, failure_threshold: int = 5, reset_after: float = 30.0):
        if failure_threshold < 1 or reset_after <= 0: raise ValueError("invalid circuit breaker limits")
        self.failure_threshold, self.reset_after = failure_threshold, reset_after
        self.failures, self.opened_at, self.state = 0, None, BreakerState.CLOSED
    def allow(self, *, now: float | None = None):
        current = time.monotonic() if now is None else float(now)
        if self.state is BreakerState.OPEN and current - self.opened_at >= self.reset_after:
            self.state = BreakerState.HALF_OPEN
        return self.state is not BreakerState.OPEN
    def success(self):
        self.failures, self.opened_at, self.state = 0, None, BreakerState.CLOSED
    def failure(self, *, now: float | None = None):
        current = time.monotonic() if now is None else float(now)
        self.failures += 1
        if self.state is BreakerState.HALF_OPEN or self.failures >= self.failure_threshold:
            self.state, self.opened_at = BreakerState.OPEN, current

__all__ = ["RetryPolicy", "BreakerState", "CircuitBreaker"]
