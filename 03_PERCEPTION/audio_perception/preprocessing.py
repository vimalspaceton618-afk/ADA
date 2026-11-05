"""Audio decoding and normalization for speech perception."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

_SUPPORTED_AUDIO_TYPES = {
    "audio/wav": "WAV",
    "audio/x-wav": "WAV",
    "audio/flac": "FLAC",
}


class AudioDecodeError(ValueError):
    """Audio bytes, content type, sample rate, or duration are invalid."""


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    samples: Any
    original_sample_rate: int
    target_sample_rate: int
    original_channels: int
    duration_seconds: float
    resampled: bool


def decode_audio(
    raw: bytes,
    content_type: str | None,
    *,
    target_sample_rate: int = 16_000,
    max_duration_seconds: float = 1_800.0,
) -> DecodedAudio:
    """Decode WAV/FLAC to mono float32 at the Whisper 16 kHz input rate.

    Decoding is in-memory and uses SoundFile/libsndfile. Compressed MP3/AAC,
    arbitrary URLs, and video containers are intentionally not accepted here.
    """
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ImportError(
            "Audio decoding requires soundfile and scipy; install the perception-audio extra"
        ) from exc

    if not isinstance(raw, bytes) or not raw:
        raise AudioDecodeError("audio content must be non-empty bytes")
    if type(target_sample_rate) is not int or target_sample_rate != 16_000:
        raise ValueError("target_sample_rate must be 16000 Hz for this Whisper adapter")
    if isinstance(max_duration_seconds, bool) or not isinstance(max_duration_seconds, (int, float)):
        raise TypeError("max_duration_seconds must be a positive number")
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be positive")
    if not isinstance(content_type, str):
        raise AudioDecodeError("audio content_type is required")
    mime = content_type.split(";", 1)[0].strip().lower()
    expected_format = _SUPPORTED_AUDIO_TYPES.get(mime)
    if expected_format is None:
        raise AudioDecodeError(f"unsupported audio content_type: {mime}")

    try:
        with sf.SoundFile(BytesIO(raw)) as audio_file:
            detected_format = audio_file.format.upper()
            original_sample_rate = int(audio_file.samplerate)
            original_channels = int(audio_file.channels)
            frames = int(audio_file.frames)
            if detected_format != expected_format:
                raise AudioDecodeError(
                    f"declared MIME type {mime} does not match decoded format {detected_format}"
                )
            if original_sample_rate < 1 or original_channels < 1 or frames < 1:
                raise AudioDecodeError("audio must have positive sample rate, channels and frames")
            duration = frames / original_sample_rate
            if duration > max_duration_seconds:
                raise AudioDecodeError(
                    f"audio duration {duration:.2f}s exceeds limit {max_duration_seconds:.2f}s"
                )
            samples = audio_file.read(dtype="float32", always_2d=True)
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(f"unable to decode audio: {exc}") from exc

    if samples.size == 0 or not np.isfinite(samples).all():
        raise AudioDecodeError("audio contains no samples or non-finite values")

    # Deterministic arithmetic downmix. This does not perform source separation.
    mono = samples.mean(axis=1, dtype=np.float32)
    resampled = original_sample_rate != target_sample_rate
    if resampled:
        import math

        divisor = math.gcd(original_sample_rate, target_sample_rate)
        mono = resample_poly(
            mono,
            target_sample_rate // divisor,
            original_sample_rate // divisor,
        ).astype(np.float32, copy=False)

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak == 0.0:
        raise AudioDecodeError("audio is silent")
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    return DecodedAudio(
        samples=mono,
        original_sample_rate=original_sample_rate,
        target_sample_rate=target_sample_rate,
        original_channels=original_channels,
        duration_seconds=float(duration),
        resampled=resampled,
    )
