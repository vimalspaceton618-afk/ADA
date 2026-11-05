"""Compatibility facade for the canonical runtime monitor."""

from importlib import import_module

_core = import_module("02_CORE_RUNTIME.runtime_monitor.health")
RuntimeHealthSnapshot = _core.RuntimeHealthSnapshot
RuntimeMonitor = _core.RuntimeMonitor
__all__ = ["RuntimeHealthSnapshot", "RuntimeMonitor"]
