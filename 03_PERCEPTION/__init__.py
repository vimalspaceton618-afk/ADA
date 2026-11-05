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
from .audio_perception import (
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_REVISION,
    TranscriptionSegment,
    WhisperAudioAdapter,
)
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
    "DEFAULT_WHISPER_MODEL",
    "DEFAULT_WHISPER_REVISION",
    "TranscriptionSegment",
    "WhisperAudioAdapter",
    "PerceptionEngine",
    "DEFAULT_SIGLIP_MODEL",
    "DEFAULT_SIGLIP_REVISION",
    "SigLIPVisionAdapter",
    "VisualAdapterCapabilities",
    "ZeroShotLabelScore",
    "DecodedVideo",
    "SampledVideoFrame",
    "SigLIPVideoAdapter",
    "VideoAdapterCapabilities",
    "VideoDecodeError",
    "VideoPerceptionAdapter",
    "decode_video",
]
