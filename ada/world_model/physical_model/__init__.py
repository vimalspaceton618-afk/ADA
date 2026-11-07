"""Stable facade for ADA World Model physical_model APIs."""
from importlib import import_module
_module = import_module("07_WORLD_MODEL.physical_model")
__all__ = ["PhysicalState"]
def __getattr__(name: str):
    return getattr(_module, name)
def __dir__():
    return sorted(set(globals()) | set(__all__))
