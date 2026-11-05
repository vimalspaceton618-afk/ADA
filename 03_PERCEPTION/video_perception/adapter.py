"""Video adapter capability contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..contracts import Modality, Observation, PerceptionInput


@dataclass(frozen=True, slots=True)
class VideoAdapterCapabilities:
    supports_frame_sampling: bool = False
    supports_frame_embeddings: bool = False
    supports_temporal_reasoning: bool = False
    supports_audio_extraction: bool = False


class VideoPerceptionAdapter(Protocol):
    modality: Modality
    capabilities: VideoAdapterCapabilities

    def perceive(self, item: PerceptionInput) -> Observation:
        """Return a source-grounded observation for the supplied video."""
        ...
