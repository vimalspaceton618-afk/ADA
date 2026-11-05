"""Compatibility facade for the canonical runtime event system."""

from importlib import import_module

_core = import_module("02_CORE_RUNTIME.event_system.bus")
InMemoryEventBus = _core.InMemoryEventBus
EventDeliveryFailure = _core.EventDeliveryFailure
__all__ = ["InMemoryEventBus", "EventDeliveryFailure"]
