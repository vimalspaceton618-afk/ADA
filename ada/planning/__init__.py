"""Stable public import facade for ADA Reasoning and Planning."""
from importlib import import_module
_module = import_module("11_REASONING_AND_PLANNING")
for _name in _module.__all__:
    globals()[_name] = getattr(_module, _name)
__all__ = list(_module.__all__)
def __getattr__(name):
    return getattr(_module, name)
def __dir__():
    return sorted(set(globals()) | set(dir(_module)))
