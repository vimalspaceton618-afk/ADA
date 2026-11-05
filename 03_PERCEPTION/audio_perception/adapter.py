"""Contract for future audio encoders, ASR and acoustic event models.

No transcription or audio classification is fabricated by this interface.
A concrete adapter must identify its model/version and preserve timestamps
and source offsets for any transcript or event evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..contracts import Modality, Observation, PerceptionInput


@dataclass(frozen=True, slots=True)
class AudioAdapterCapabilities:
    supports_transcription: bool = False
    supports_speaker_diarization: bool = False
    supports_audio_event_detection: bool = False
    supports_timestamps: bool = False
    supports_language_conditioning: bool = False


class AudioPerceptionAdapter(Protocol):
    modality: Modality
    capabilities: AudioAdapterCapabilities

    def perceive(self, item: PerceptionInput) -> Observation:
        """Return a source-grounded audio observation."""
        ...
