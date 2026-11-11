"""Deployment configuration validation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Any

@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    environment: str
    replicas: int
    max_batch_tokens: int
    max_output_tokens: int
    request_timeout_seconds: float
    enable_external_tools: bool = False
    def __post_init__(self):
        if self.environment not in {"development", "staging", "production"}: raise ValueError("invalid environment")
        for name in ("replicas", "max_batch_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1: raise ValueError(f"{name} must be a positive integer")
        if self.max_output_tokens > self.max_batch_tokens: raise ValueError("output limit cannot exceed batch token budget")
        if self.request_timeout_seconds <= 0: raise ValueError("request timeout must be positive")
        if type(self.enable_external_tools) is not bool: raise TypeError("enable_external_tools must be bool")

def from_mapping(value: Mapping[str, Any]) -> DeploymentConfig:
    if not isinstance(value, Mapping): raise TypeError("configuration must be a mapping")
    unknown = set(value) - set(DeploymentConfig.__dataclass_fields__)
    if unknown: raise ValueError(f"unknown deployment configuration keys: {sorted(unknown)}")
    return DeploymentConfig(**dict(value))

__all__ = ["DeploymentConfig", "from_mapping"]
