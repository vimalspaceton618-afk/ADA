"""ADA Core Runtime implementation package.

The directory name is retained as an architectural namespace. Consumers should
import the stable public API from ada.runtime; that facade loads this package
through Python's importlib API.
"""

from .contracts import RuntimeEvent, TaskSnapshot, TaskSpec, TaskStatus
from .runtime import Runtime, TaskContext

__all__ = ["Runtime", "TaskContext", "RuntimeEvent", "TaskSnapshot", "TaskSpec", "TaskStatus"]
