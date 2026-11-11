"""Subsystem interface specifications with explicit maturity and limitations."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SubsystemSpec:
    subsystem_id: str
    version: str
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    failure_modes: tuple[str, ...]
    limitations: tuple[str, ...]
    maturity: str = "reference"
    def __post_init__(self):
        if any(not x.strip() for x in (self.subsystem_id, self.version, self.purpose)):
            raise ValueError("subsystem spec identity and purpose required")
        if not self.failure_modes or not self.limitations: raise ValueError("failure modes and limitations must be documented")
        if self.maturity not in {"planned", "reference", "experimental", "production"}: raise ValueError("invalid maturity")

__all__ = ["SubsystemSpec"]
