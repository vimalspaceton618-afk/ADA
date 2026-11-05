"""Explicit fault classification and retry policy."""

from .policy import RetryPolicy, RetryDecision

__all__ = ["RetryPolicy", "RetryDecision"]
