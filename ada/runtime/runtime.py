"""Compatibility facade for the canonical Core Runtime orchestrator."""

from importlib import import_module

_core = import_module("02_CORE_RUNTIME.runtime")
Runtime = _core.Runtime
TaskContext = _core.TaskContext
__all__ = ["Runtime", "TaskContext"]
