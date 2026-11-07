"""Stable facade for ADA World Model temporal_model APIs."""
from importlib import import_module
_module = import_module("07_WORLD_MODEL.temporal_model")
__all__ = ["EpisodeTrajectory","EpisodeWindow"]
def __getattr__(name: str):
    return getattr(_module, name)
def __dir__():
    return sorted(set(globals()) | set(__all__))
