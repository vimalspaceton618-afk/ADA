"""Stable facade for ADA World Model simulation_engine APIs."""
from importlib import import_module
_module = import_module("07_WORLD_MODEL.simulation_engine")
__all__ = ["LatentRollout","imagine_rollout","compare_rollouts"]
def __getattr__(name: str):
    return getattr(_module, name)
def __dir__():
    return sorted(set(globals()) | set(__all__))
