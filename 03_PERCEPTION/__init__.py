"""ADA's canonical perception subsystem.

This package exposes modality-agnostic contracts and concrete optional
adapters from the numbered architecture source tree.
"""

from .contracts import (
    Modality,
    PerceptionInput,
    Observation,
    PerceptionResult,
    PerceptionStatus,
    EvidenceSpan,
)
from .engine import PerceptionEngine
from .visual_perception import (
    DEFAULT_SIGLIP_MODEL,
    DEFAULT_SIGLIP_REVISION,
    SigLIPVisionAdapter,
    VisualAdapterCapabilities,
    ZeroShotLabelScore,
)

__all__ = [
    "Modality",
    "PerceptionInput",
    "Observation",
    "PerceptionResult",
    "PerceptionStatus",
    "EvidenceSpan",
    "PerceptionEngine",
    "DEFAULT_SIGLIP_MODEL",
    "DEFAULT_SIGLIP_REVISION",
    "SigLIPVisionAdapter",
    "VisualAdapterCapabilities",
    "ZeroShotLabelScore",
]
