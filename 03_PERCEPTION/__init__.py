"""ADA's canonical perception subsystem.

The importable public facade is ada.perception. This numbered directory is
kept as the architectural source tree and loaded through importlib.
"""

from .contracts import (
    Modality, PerceptionInput, Observation, PerceptionResult,
    PerceptionStatus, EvidenceSpan,
)
from .engine import PerceptionEngine

__all__ = [
    "Modality", "PerceptionInput", "Observation", "PerceptionResult",
    "PerceptionStatus", "EvidenceSpan", "PerceptionEngine",
]
