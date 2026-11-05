"""Compatibility facade for the canonical runtime state manager."""

from importlib import import_module

RuntimeStateStore = import_module("02_CORE_RUNTIME.state_manager.store").RuntimeStateStore
__all__ = ["RuntimeStateStore"]
