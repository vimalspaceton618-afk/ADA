"""Bounded, timestamp-preserving video frame sampling with PyAV.

Only video frames are decoded here. Audio tracks are deliberately ignored;
audio extraction and ASR remain separate, explicitly registered operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable


_SUPPORTED_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-matroska",
    "video/webm",
    "video/mpeg",
}


class VideoDecodeError(ValueError):
    """Video bytes, metadata, or decoded frames violate configured limits."""


@dataclass(frozen=True, slots=True)
class SampledVideoFrame:
    """A sampled decoded frame; the AV frame remains internal to perception."""

    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    frame: Any


@dataclass(frozen=True, slots=True)
class DecodedVideo:
    frames: tuple[SampledVideoFrame, ...]
    duration_seconds: float | None
    container_format: str | None
    decoded_frame_count: int
    effective_sample_interval_seconds: float


def _positive_finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _frame_timestamp(frame: Any, stream: Any) -> float | None:
    value = getattr(frame, "time", None)
    if value is not None:
        timestamp = float(value)
        return timestamp if math.isfinite(timestamp) else None

    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None) or getattr(stream, "time_base", None)
    if pts is None or not time_base:
        return None
    timestamp = float(pts * time_base)
    return timestamp if math.isfinite(timestamp) else None


def _known_duration_seconds(container: Any, stream: Any) -> float | None:
    duration = getattr(container, "duration", None)
    if duration is not None and duration > 0:
        # FFmpeg AV_TIME_BASE is one million ticks per second.
        return float(duration) / 1_000_000.0

    stream_duration = getattr(stream, "duration", None)
    stream_time_base = getattr(stream, "time_base", None)
    if stream_duration is not None and stream_duration > 0 and stream_time_base:
        result = float(stream_duration * stream_time_base)
        if math.isfinite(result) and result > 0:
            return result
    return None


def decode_video(
    raw: bytes,
    content_type: str | None,
    *,
    sample_interval_seconds: float = 5.0,
    max_frames: int = 12,
    max_duration_seconds: float = 600.0,
    max_video_bytes: int = 50 * 1024 * 1024,
    max_frame_pixels: int = 40_000_000,
    max_decoded_frames: int = 100_000,
    container_opener: Callable[..., Any] | None = None,
) -> DecodedVideo:
    """Decode and sparsely sample video frames under explicit resource caps.

    Sampling starts at the first decodable frame and then selects the first
    frame at or after each interval boundary. Frames retain presentation-time
    timestamps in seconds. This function never extracts audio or writes files.
    """
    interval = _positive_finite_number("sample_interval_seconds", sample_interval_seconds)
    duration_limit = _positive_finite_number("max_duration_seconds", max_duration_seconds)
    for name, value in (
        ("max_frames", max_frames),
        ("max_video_bytes", max_video_bytes),
        ("max_frame_pixels", max_frame_pixels),
        ("max_decoded_frames", max_decoded_frames),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    if not isinstance(raw, bytes) or not raw:
        raise VideoDecodeError("video content must be non-empty bytes")
    if len(raw) > max_video_bytes:
        raise VideoDecodeError(
            f"video exceeds byte limit ({len(raw)} > {max_video_bytes})"
        )
    if not isinstance(content_type, str):
        raise VideoDecodeError("video content_type is required")
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime not in _SUPPORTED_VIDEO_MIME_TYPES:
        raise VideoDecodeError(f"unsupported video content_type: {mime}")

    if container_opener is None:
        try:
            import av
        except ImportError as exc:
            raise ImportError(
                "Video decoding requires PyAV; install the perception-video extra"
            ) from exc
        container_opener = av.open

    container = None
    try:
        container = container_opener(BytesIO(raw), mode="r")
        streams = getattr(container, "streams", None)
        video_streams = getattr(streams, "video", None)
        if not video_streams:
            raise VideoDecodeError("container does not contain a video stream")
        stream = video_streams[0]

        known_duration = _known_duration_seconds(container, stream)
        if known_duration is not None and known_duration > duration_limit:
            raise VideoDecodeError(
                f"video duration exceeds limit ({known_duration:.3f}s > "
                f"{duration_limit:.3f}s)"
            )

        selected: list[SampledVideoFrame] = []
        # Treat the configured interval as a minimum. For long, known-duration
        # clips, widen the interval so the bounded sample spans the whole clip.
        effective_interval = interval
        if known_duration is not None and max_frames > 1:
            effective_interval = max(interval, known_duration / max_frames)
        next_sample_at = 0.0
        previous_timestamp = -1.0
        decoded_frame_count = 0
        container_format_obj = getattr(container, "format", None)
        format_name = getattr(container_format_obj, "name", None)

        for frame_index, frame in enumerate(container.decode(video=stream.index)):
            decoded_frame_count = frame_index + 1
            if decoded_frame_count > max_decoded_frames:
                raise VideoDecodeError(
                    f"decoded frame count exceeds limit ({max_decoded_frames})"
                )
            width = int(getattr(frame, "width", 0))
            height = int(getattr(frame, "height", 0))
            if width < 1 or height < 1:
                raise VideoDecodeError("decoded video frame has invalid dimensions")
            if width * height > max_frame_pixels:
                raise VideoDecodeError(
                    f"decoded frame exceeds pixel limit ({width * height} > "
                    f"{max_frame_pixels})"
                )

            timestamp = _frame_timestamp(frame, stream)
            if timestamp is None or timestamp < 0:
                continue
            if timestamp > duration_limit:
                raise VideoDecodeError(
                    f"decoded frame timestamp exceeds duration limit ({timestamp:.3f}s)"
                )
            if timestamp + 1e-9 < previous_timestamp:
                raise VideoDecodeError("decoded frame timestamps are not monotonic")
            if timestamp <= previous_timestamp + 1e-9:
                continue
            previous_timestamp = timestamp

            if timestamp + 1e-9 >= next_sample_at:
                selected.append(
                    SampledVideoFrame(
                        frame_index=frame_index,
                        timestamp_seconds=timestamp,
                        width=width,
                        height=height,
                        frame=frame,
                    )
                )
                while next_sample_at <= timestamp + 1e-9:
                    next_sample_at += interval
                if len(selected) >= max_frames:
                    break

        if not selected:
            raise VideoDecodeError("video contains no sampleable timestamped frames")

        return DecodedVideo(
            frames=tuple(selected),
            duration_seconds=known_duration,
            container_format=format_name,
            decoded_frame_count=decoded_frame_count,
            effective_sample_interval_seconds=effective_interval,
        )
    except VideoDecodeError:
        raise
    except Exception as exc:
        raise VideoDecodeError(f"unable to decode video: {type(exc).__name__}: {exc}") from exc
    finally:
        if container is not None:
            close = getattr(container, "close", None)
            if callable(close):
                close()
