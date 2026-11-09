"""Stable public facade for ADA Learning."""
from importlib import import_module

_module = import_module("08_LEARNING")
__all__ = list(_module.__all__)


def __getattr__(name: str):
    return getattr(_module, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
