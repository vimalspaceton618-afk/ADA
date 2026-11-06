"""Stable Python import facade for ADA's canonical 04_REPRESENTATION tree."""

from importlib import import_module

_core = import_module("04_REPRESENTATION")
for _name in _core.__all__:
    globals()[_name] = getattr(_core, _name)

__all__ = list(_core.__all__)
