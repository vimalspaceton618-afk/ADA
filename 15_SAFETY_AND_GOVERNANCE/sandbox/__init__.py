"""Fail-closed sandbox adapter contracts.

No in-process Python function can provide a security boundary. A production
backend must enforce OS/container isolation, egress policy and resource limits.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True, slots=True)
class SandboxProfile:
    profile_id: str
    timeout_seconds: float
    memory_mb: int
    cpu_seconds: int
    network_enabled: bool = False
    read_only_root: bool = True
    def __post_init__(self):
        if not self.profile_id.strip(): raise ValueError("profile_id is required")
        if self.timeout_seconds <= 0 or self.memory_mb < 64 or self.cpu_seconds < 1:
            raise ValueError("sandbox resource limits are invalid")
        if self.network_enabled: raise ValueError("network access is denied by default")

@dataclass(frozen=True, slots=True)
class SandboxResult:
    exit_code: int
    stdout: bytes
    stderr: bytes
    duration_seconds: float
    backend_id: str

class SandboxBackend(Protocol):
    def execute(self, request: Any, profile: SandboxProfile) -> SandboxResult: ...

class SandboxRunner:
    def __init__(self, backend: SandboxBackend | None = None):
        self.backend = backend
    def run(self, request: Any, profile: SandboxProfile) -> SandboxResult:
        if not isinstance(profile, SandboxProfile): raise TypeError("profile must be SandboxProfile")
        if self.backend is None: raise RuntimeError("no isolated sandbox backend configured; execution denied")
        result = self.backend.execute(request, profile)
        if not isinstance(result, SandboxResult): raise TypeError("sandbox backend returned an invalid result")
        if result.duration_seconds > profile.timeout_seconds:
            raise TimeoutError("sandbox backend exceeded the declared timeout")
        return result

__all__ = ["SandboxProfile", "SandboxResult", "SandboxBackend", "SandboxRunner"]
