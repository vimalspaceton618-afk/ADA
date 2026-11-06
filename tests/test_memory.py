from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from ada.memory import (
    MemoryConflictError, MemoryEngine, MemoryKind, MemoryQuery, MemoryRecord,
    MemorySourceKind, MemoryStatus, utc_now if False else MemoryEngine,
)
