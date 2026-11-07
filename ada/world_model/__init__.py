"""Stable, lazy public import facade for ADA World Model."""
from importlib import import_module

_module = import_module("07_WORLD_MODEL")
for _name in _module.__all__:
    globals()[_name] = getattr(_module, _name)

__all__ = list(_module.__all__)

def __getattr__(name: str):
    return getattr(_module, name)

def __dir__():
    return sorted(set(globals()) | set(dir(_module)))
