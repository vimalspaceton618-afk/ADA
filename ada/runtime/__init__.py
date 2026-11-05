"""Stable public import surface for ADA Core Runtime.

Canonical implementations live in the architecturally named
02_CORE_RUNTIME directory. This package provides normal Python imports.
"""

from importlib import import_module

_contracts = import_module("02_CORE_RUNTIME.contracts")
_events = import_module("02_CORE_RUNTIME.event_system.bus")
_monitor = import_module("02_CORE_RUNTIME.runtime_monitor.health")
_runtime = import_module("02_CORE_RUNTIME.runtime")
_scheduler = import_module("02_CORE_RUNTIME.task_scheduler.fifo")
_state = import_module("02_CORE_RUNTIME.state_manager.store")
_recovery = import_module("02_CORE_RUNTIME.fault_recovery.policy")
_messages = import_module("02_CORE_RUNTIME.communication_bus.messages")
_processes = import_module("02_CORE_RUNTIME.process_manager.supervisor")
_concurrency = import_module("02_CORE_RUNTIME.concurrency.primitives")

for _module, _names in (
    (_contracts, (
        "DuplicateTaskError", "HandlerNotFoundError", "InvalidTransitionError",
        "QueueFullError", "RetryExhaustedError", "RuntimeErrorBase", "RuntimeEvent",
        "TaskCancelledError", "TaskSnapshot", "TaskSpec", "TaskStatus",
        "TaskTimedOutError", "UnknownTaskError",
    )),
    (_events, ("EventDeliveryFailure", "InMemoryEventBus")),
    (_monitor, ("RuntimeHealthSnapshot", "RuntimeMonitor")),
    (_runtime, ("Runtime", "TaskContext")),
    (_scheduler, ("FIFOScheduler",)),
    (_state, ("RuntimeStateStore",)),
    (_recovery, ("RetryDecision", "RetryPolicy")),
    (_messages, ("InMemoryMessageBus", "MessageEnvelope")),
    (_processes, ("ProcessRecord", "ProcessSupervisor")),
    (_concurrency, ("CancellationToken", "CapacityLimiter")),
):
    for _name in _names:
        globals()[_name] = getattr(_module, _name)

__all__ = [
    "CancellationToken", "CapacityLimiter", "DuplicateTaskError",
    "EventDeliveryFailure", "FIFOScheduler", "HandlerNotFoundError",
    "InMemoryEventBus", "InMemoryMessageBus", "InvalidTransitionError",
    "MessageEnvelope", "ProcessRecord", "ProcessSupervisor", "QueueFullError",
    "RetryDecision", "RetryPolicy", "Runtime", "RuntimeErrorBase", "RuntimeEvent",
    "RuntimeHealthSnapshot", "RuntimeMonitor", "RuntimeStateStore", "TaskCancelledError",
    "TaskContext", "TaskSnapshot", "TaskSpec", "TaskStatus", "TaskTimedOutError",
    "UnknownTaskError",
]
