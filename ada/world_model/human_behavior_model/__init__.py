"""Stable facade for ADA World Model human_behavior_model APIs."""
from importlib import import_module
_module = import_module("07_WORLD_MODEL.human_behavior_model")
__all__ = ["ObservedBehaviorEvent","EmpiricalActionPrior"]
def __getattr__(name: str):
    return getattr(_module, name)
def __dir__():
    return sorted(set(globals()) | set(__all__))
