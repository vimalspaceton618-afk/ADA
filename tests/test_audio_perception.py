from types import SimpleNamespace

import numpy as np
import pytest

from ada.perception import (
    Modality,
    PerceptionEngine,
    PerceptionInput,
    PerceptionStatus,
    WhisperAudioAdapter,
)
from importlib import import_module


whisper_module = import_module("03_PERCEPTION.audio_perception.whisper")


class FakeASR:
    def __call__(self, audio, *, return_timestamps, generate_kwargs):
        assert return_timestamps is True
        assert audio["sampling_rate"] == 16_000
        assert generate_kwargs["task"] == "transcribe"
        return {
            "text": "Hello from ADA.",
            "chunks": [
                {"text": "Hello", "timestamp": (0.0, 0.4)},
                {"text": "from ADA.", "timestamp": (0.4, 1.2)},
            ],
        }


def test_whisper_adapter_transcription_preserves_provenance_and_timestamps(monkeypatch):
    decoded = SimpleNamespace(
        samples=np.zeros(16_000, dtype=np.float32),
        original_sample_rate=48_000,
        target_sample_rate=16_000,
        original_channels=2,
        duration_seconds=1.0,
        resampled=True,
    )
    monkeypatch.setattr(whisper_module, "decode_audio", lambda *args, **kwargs: decoded)

    adapter = WhisperAudioAdapter(
        asr_pipeline=FakeASR(),
        model_id="openai/whisper-base",
        revision="test-revision",
    )
    engine = PerceptionEngine()
    engine.register_adapter(adapter)

    source = PerceptionInput(
        modality=Modality.AUDIO,
        content=b"synthetic-test-audio",
        content_type="audio/wav",
        metadata={"language": "english"},
        correlation_id="audio-test-001",
    )
    result = engine.perceive(source)

    assert result.status is PerceptionStatus.OBSERVED
    assert result.observation is not None
    assert result.observation.extracted_text == "Hello from ADA."
    assert result.observation.source_input_id == source.input_id
    assert result.observation.content_sha256 == result.content_sha256
    assert result.observation.attributes["model_id"] == "openai/whisper-base"
    assert result.observation.attributes["requested_language"] == "english"
    assert result.observation.attributes["segments"] == (
        {
            "text": "Hello",
            "start_seconds": 0.0,
            "end_seconds": 0.4,
        },
        {
            "text": " from ADA.",
            "start_seconds": 0.4,
            "end_seconds": 1.2,
        },
    )


def test_whisper_adapter_rejects_empty_transcript(monkeypatch):
    decoded = SimpleNamespace(
        samples=np.zeros(16_000, dtype=np.float32),
        original_sample_rate=16_000,
        target_sample_rate=16_000,
        original_channels=1,
        duration_seconds=1.0,
        resampled=False,
    )
    monkeypatch.setattr(whisper_module, "decode_audio", lambda *args, **kwargs: decoded)

    class EmptyASR:
        def __call__(self, audio, *, return_timestamps, generate_kwargs):
            return {"text": "   ", "chunks": []}

    adapter = WhisperAudioAdapter(
        asr_pipeline=EmptyASR(),
        model_id="openai/whisper-base",
        revision="test-revision",
    )
    engine = PerceptionEngine()
    engine.register_adapter(adapter)
    source = PerceptionInput(
        modality=Modality.AUDIO,
        content=b"synthetic-test-audio",
        content_type="audio/wav",
    )

    result = engine.perceive(source)
    assert result.status is PerceptionStatus.FAILED
    assert result.observation is None
    assert "empty transcript" in result.errors[0]


def test_whisper_capabilities_do_not_claim_diarization_or_event_detection():
    adapter = WhisperAudioAdapter(
        asr_pipeline=FakeASR(),
        model_id="openai/whisper-base",
        revision="test-revision",
    )
    assert adapter.capabilities.supports_transcription is True
    assert adapter.capabilities.supports_timestamps is True
    assert adapter.capabilities.supports_language_conditioning is True
    assert adapter.capabilities.supports_speaker_diarization is False
    assert adapter.capabilities.supports_audio_event_detection is False
