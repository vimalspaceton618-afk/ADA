"""Stable import facade for ADA's infrastructure and governance subsystems."""

from importlib import import_module

_MODULES = {
    "safety": "15_SAFETY_AND_GOVERNANCE",
    "data": "16_DATA_INFRASTRUCTURE",
    "training": "17_TRAINING",
    "inference": "18_INFERENCE",
    "interaction": "19_INTERACTION",
    "distributed": "20_DISTRIBUTED_INFRASTRUCTURE",
    "observability": "21_OBSERVABILITY",
    "interoperability": "22_INTEROPERABILITY",
    "deployment": "23_DEPLOYMENT",
    "testing": "24_TESTING",
    "research": "25_RESEARCH",
    "documentation": "26_DOCUMENTATION",
    "operations": "27_OPERATIONS",
}

__all__ = []
for _alias, _path in _MODULES.items():
    _module = import_module(_path)
    globals()[_alias] = _module
    __all__.append(_alias)
    for _name in getattr(_module, "__all__", ()):
        if _name not in globals():
            globals()[_name] = getattr(_module, _name)
        if _name not in __all__:
            __all__.append(_name)
