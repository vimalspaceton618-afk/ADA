"""Contract for future image/video visual encoders and interpreters.

No visual model is bundled here. Implementations must return source-grounded
Observations and report model/version metadata in observation attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..contracts import Modality, Observation, PerceptionInput


@dataclass(frozen=True, slots=True)
class VisualAdapterCapabilities:
    supports_images: bool = True
    supports_video_frames: bool = False
    supports_ocr: bool = False
    supports_object_detection: bool = False
    supports_spatial_grounding: bool = False
    supports_image_embeddings: bool = False
    supports_zero_shot_label_scoring: bool = False


class VisualPerceptionAdapter(Protocol):
    modality: Modality
    capabilities: VisualAdapterCapabilities

    def perceive(self, item: PerceptionInput) -> Observation:
        """Return an evidence-grounded visual observation for the supplied input."""
        ...
