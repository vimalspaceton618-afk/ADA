"""Stable public import facade for ADA External Knowledge."""
from importlib import import_module
_module = import_module("13_EXTERNAL_KNOWLEDGE")
for _name in _module.__all__:
    globals()[_name] = getattr(_module, _name)
__all__ = list(_module.__all__)
def __getattr__(name):
    return getattr(_module, name)
def __dir__():
    return sorted(set(globals()) | set(dir(_module)))
