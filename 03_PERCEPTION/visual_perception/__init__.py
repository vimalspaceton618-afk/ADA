"""Visual perception adapters and capability declarations."""

from .adapter import VisualAdapterCapabilities, VisualPerceptionAdapter
from .preprocessing import DecodedImage, ImageDecodeError, decode_image
from .siglip import (
    DEFAULT_SIGLIP_MODEL,
    DEFAULT_SIGLIP_REVISION,
    SigLIPVisionAdapter,
    ZeroShotLabelScore,
)

__all__ = [
    "DEFAULT_SIGLIP_MODEL",
    "DEFAULT_SIGLIP_REVISION",
    "DecodedImage",
    "ImageDecodeError",
    "SigLIPVisionAdapter",
    "VisualAdapterCapabilities",
    "VisualPerceptionAdapter",
    "ZeroShotLabelScore",
    "decode_image",
]
