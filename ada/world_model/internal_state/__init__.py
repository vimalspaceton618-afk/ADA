"""Stable facade for ADA World Model internal_state APIs."""
from importlib import import_module
_module = import_module("07_WORLD_MODEL.internal_state")
__all__ = ["ModelRuntimeState"]
def __getattr__(name: str):
    return getattr(_module, name)
def __dir__():
    return sorted(set(globals()) | set(__all__))
