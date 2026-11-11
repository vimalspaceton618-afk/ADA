"""Local runtime start/stop/health adapter contract."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any

@dataclass(frozen=True, slots=True)
class LocalRuntimeStatus:
    state: str
    model_id: str | None
    process_id: int | None
    detail: str = ""

class LocalRuntime(Protocol):
    def start(self, artifact_path: str, config: Any) -> LocalRuntimeStatus: ...
    def stop(self) -> LocalRuntimeStatus: ...
    def health(self) -> LocalRuntimeStatus: ...

class LocalDeploymentController:
    def __init__(self, runtime: LocalRuntime):
        if not all(callable(getattr(runtime, name, None)) for name in ("start", "stop", "health")):
            raise TypeError("runtime must implement start, stop and health")
        self.runtime = runtime
    def _check(self, status):
        if not isinstance(status, LocalRuntimeStatus): raise TypeError("runtime returned invalid status")
        return status
    def start(self, artifact_path: str, config: Any): return self._check(self.runtime.start(artifact_path, config))
    def stop(self): return self._check(self.runtime.stop())
    def health(self): return self._check(self.runtime.health())

__all__ = ["LocalRuntimeStatus", "LocalRuntime", "LocalDeploymentController"]
