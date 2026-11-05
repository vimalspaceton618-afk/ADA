"""Audio perception contracts and optional Whisper implementation."""

from .adapter import AudioAdapterCapabilities, AudioPerceptionAdapter
from .preprocessing import AudioDecodeError, DecodedAudio, decode_audio
from .whisper import (
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_REVISION,
    TranscriptionSegment,
    WhisperAudioAdapter,
)

__all__ = [
    "AudioAdapterCapabilities",
    "AudioDecodeError",
    "AudioPerceptionAdapter",
    "DEFAULT_WHISPER_MODEL",
    "DEFAULT_WHISPER_REVISION",
    "DecodedAudio",
    "TranscriptionSegment",
    "WhisperAudioAdapter",
    "decode_audio",
]
