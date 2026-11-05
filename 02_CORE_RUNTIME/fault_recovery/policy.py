"""Bounded exponential retry policy; never retries by default."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Type


@dataclass(frozen=True, slots=True)
class RetryDecision:
    retry: bool
    next_attempt: int
    delay_seconds: float
    reason: str


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 1
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 8.0
    retryable_exceptions: tuple[Type[BaseException], ...] = ()

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        for name, value in (
            ("base_delay_seconds", self.base_delay_seconds),
            ("max_delay_seconds", self.max_delay_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be finite and non-negative")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds cannot be below base delay")
        if not isinstance(self.retryable_exceptions, tuple) or any(
            not isinstance(item, type) or not issubclass(item, BaseException)
            for item in self.retryable_exceptions
        ):
            raise TypeError("retryable_exceptions must be a tuple of exception types")

    def decide(self, *, completed_attempts: int, error: BaseException) -> RetryDecision:
        if type(completed_attempts) is not int or completed_attempts < 1:
            raise ValueError("completed_attempts must be a positive integer")
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        next_attempt = completed_attempts + 1
        if not self.retryable_exceptions or not isinstance(error, self.retryable_exceptions):
            return RetryDecision(False, next_attempt, 0.0, "error type is not explicitly retryable")
        if completed_attempts >= self.max_attempts:
            return RetryDecision(False, next_attempt, 0.0, "maximum attempts reached")
        delay = min(self.base_delay_seconds * (2 ** (completed_attempts - 1)), self.max_delay_seconds)
        return RetryDecision(True, next_attempt, delay, "retryable error within attempt budget")
