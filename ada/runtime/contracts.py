"""Compatibility facade for the canonical 02_CORE_RUNTIME contracts."""

from importlib import import_module

_core = import_module("02_CORE_RUNTIME.contracts")
for _name in (
    "TaskStatus", "RuntimeErrorBase", "InvalidTransitionError", "DuplicateTaskError",
    "UnknownTaskError", "QueueFullError", "HandlerNotFoundError", "TaskCancelledError",
    "TaskTimedOutError", "RetryExhaustedError", "TaskSpec", "TaskSnapshot", "RuntimeEvent",
    "freeze_json", "utc_now",
):
    globals()[_name] = getattr(_core, _name)

__all__ = [name for name in globals() if not name.startswith("_")]
