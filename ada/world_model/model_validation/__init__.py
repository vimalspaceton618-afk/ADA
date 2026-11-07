"""Stable facade for ADA World Model model_validation APIs."""
from importlib import import_module
_module = import_module("07_WORLD_MODEL.model_validation")
__all__ = ["mean_squared_error","mean_absolute_error","expected_calibration_error","horizon_mse"]
def __getattr__(name: str):
    return getattr(_module, name)
def __dir__():
    return sorted(set(globals()) | set(__all__))
