"""Safe image decoding before visual model preprocessing."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


_SUPPORTED_MIME_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


class ImageDecodeError(ValueError):
    """Image bytes, declared content type, or dimensions are invalid."""


@dataclass(frozen=True, slots=True)
class DecodedImage:
    image: Image.Image
    detected_format: str
    original_width: int
    original_height: int
    oriented_width: int
    oriented_height: int


def decode_image(
    raw: bytes,
    content_type: str | None,
    *,
    max_pixels: int = 40_000_000,
) -> DecodedImage:
    """Decode JPEG/PNG/WebP bytes, apply EXIF orientation, and return RGB.

    The image is decoded in memory only. SVG, GIF, TIFF, remote URLs, and
    arbitrary image plugins are intentionally not accepted by this adapter.
    """
    if not isinstance(raw, bytes) or not raw:
        raise ImageDecodeError("image content must be non-empty bytes")
    if type(max_pixels) is not int or max_pixels < 1:
        raise ValueError("max_pixels must be a positive integer")
    if not isinstance(content_type, str):
        raise ImageDecodeError("image content_type is required")
    mime = content_type.split(";", 1)[0].strip().lower()
    expected_format = _SUPPORTED_MIME_FORMATS.get(mime)
    if expected_format is None:
        raise ImageDecodeError(f"unsupported image content_type: {mime}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as source:
                detected = (source.format or "").upper()
                width, height = source.size
                if width < 1 or height < 1:
                    raise ImageDecodeError("image dimensions must be positive")
                if width * height > max_pixels:
                    raise ImageDecodeError(
                        f"image exceeds pixel limit ({width * height} > {max_pixels})"
                    )
                if detected != expected_format:
                    raise ImageDecodeError(
                        f"declared MIME type {mime} does not match decoded format {detected}"
                    )
                source.verify()

            with Image.open(BytesIO(raw)) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                rgb = oriented.convert("RGB")
                oriented_width, oriented_height = rgb.size
    except ImageDecodeError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageDecodeError(f"unable to safely decode image: {exc}") from exc

    return DecodedImage(
        image=rgb,
        detected_format=detected,
        original_width=width,
        original_height=height,
        oriented_width=oriented_width,
        oriented_height=oriented_height,
    )
