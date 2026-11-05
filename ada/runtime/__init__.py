"""ADA's CPU-first core runtime foundation."""

from .contracts import (
    DuplicateTaskError,
    HandlerNotFoundError,
    InvalidTransitionError,
    QueueFullError,
    RuntimeErrorBase,
    RuntimeEvent,
    TaskCancelledError,
    TaskSnapshot,
    TaskSpec,
    TaskStatus,
    TaskTimedOutError,
    UnknownTaskError,
)
from .events import EventDeliveryFailure, InMemoryEventBus
from .monitor import RuntimeHealthSnapshot, RuntimeMonitor
from .runtime import Runtime, TaskContext
from .scheduler import FIFOScheduler
from .state import RuntimeStateStore

__all__ = [
    "DuplicateTaskError",
    "EventDeliveryFailure",
    "FIFOScheduler",
    "HandlerNotFoundError",
    "InMemoryEventBus",
    "InvalidTransitionError",
    "QueueFullError",
    "Runtime",
    "RuntimeErrorBase",
    "RuntimeEvent",
    "RuntimeHealthSnapshot",
    "RuntimeMonitor",
    "RuntimeStateStore",
    "TaskCancelledError",
    "TaskContext",
    "TaskSnapshot",
    "TaskSpec",
    "TaskStatus",
    "TaskTimedOutError",
    "UnknownTaskError",
]
