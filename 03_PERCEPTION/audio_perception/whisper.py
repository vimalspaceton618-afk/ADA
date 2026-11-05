"""Optional pretrained Whisper speech-to-text adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..contracts import Modality, Observation, PerceptionInput, PerceptionStatus, content_digest
from .adapter import AudioAdapterCapabilities
from .preprocessing import DecodedAudio, decode_audio


DEFAULT_WHISPER_MODEL = "openai/whisper-base"
DEFAULT_WHISPER_REVISION = "main"


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """Transcript segment with timestamps in seconds relative to the source clip."""

    text: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("segment text must be a string")
        if not math.isfinite(self.start_seconds) or not math.isfinite(self.end_seconds):
            raise ValueError("segment timestamps must be finite")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("segment timestamps must be a valid non-negative interval")


class WhisperAudioAdapter:
    """Multilingual Whisper ASR with explicit transcription-only semantics.

    Accepts WAV/FLAC bytes, converts them to mono 16 kHz float32, and invokes
    the Transformers automatic-speech-recognition pipeline. Audio timestamps
    are retained when the pipeline returns chunk boundaries. No speaker
    diarization or acoustic event classification is claimed.
    """

    modality = Modality.AUDIO
    capabilities = AudioAdapterCapabilities(
        supports_transcription=True,
        supports_speaker_diarization=False,
        supports_audio_event_detection=False,
        supports_timestamps=True,
        supports_language_conditioning=True,
    )

    def __init__(
        self,
        *,
        asr_pipeline: Any,
        model_id: str,
        revision: str,
        max_duration_seconds: float = 1_800.0,
        torch_module: Any | None = None,
    ) -> None:
        if not callable(asr_pipeline):
            raise TypeError("asr_pipeline must be callable")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be non-empty")
        if isinstance(max_duration_seconds, bool) or not isinstance(max_duration_seconds, (int, float)):
            raise TypeError("max_duration_seconds must be a positive number")
        if not math.isfinite(float(max_duration_seconds)) or max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be finite and positive")
        self.asr_pipeline = asr_pipeline
        self.model_id = model_id
        self.requested_revision = revision
        self.max_duration_seconds = float(max_duration_seconds)
        self._torch = torch_module
        model = getattr(asr_pipeline, "model", None)
        resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
        self.resolved_revision = resolved or revision

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_WHISPER_MODEL,
        *,
        revision: str = DEFAULT_WHISPER_REVISION,
        device: str | None = None,
        local_files_only: bool = True,
        max_duration_seconds: float = 1_800.0,
    ) -> "WhisperAudioAdapter":
        """Load a local/cached Whisper checkpoint; remote downloads are opt-in."""
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            raise ImportError(
                "Install ADA's optional audio dependencies: "
                "pip install -e '.[perception-audio]'"
            ) from exc

        target_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float16 if target_device.startswith("cuda") else torch.float32,
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        device_index: int | str = 0 if target_device.startswith("cuda") else -1
        asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device_index,
            chunk_length_s=30,
            stride_length_s=(5, 5),
        )
        return cls(
            asr_pipeline=asr,
            model_id=model_id,
            revision=revision,
            max_duration_seconds=max_duration_seconds,
            torch_module=torch,
        )

    def _decode(self, item: PerceptionInput) -> DecodedAudio:
        if item.modality is not Modality.AUDIO:
            raise TypeError("WhisperAudioAdapter only accepts AUDIO inputs")
        if not isinstance(item.content, bytes):
            raise TypeError("AUDIO input content must be bytes")
        return decode_audio(
            item.content,
            item.content_type,
            max_duration_seconds=self.max_duration_seconds,
        )

    def _transcribe(self, decoded: DecodedAudio, language: str | None) -> tuple[str, tuple[TranscriptionSegment, ...]]:
        generate_kwargs: dict[str, Any] = {"task": "transcribe"}
        if language is not None:
            if not isinstance(language, str) or not language.strip():
                raise ValueError("language must be a non-empty language name/code or None")
            generate_kwargs["language"] = language.strip()

        output = self.asr_pipeline(
            {
                "raw": decoded.samples,
                "sampling_rate": decoded.target_sample_rate,
            },
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
        if not isinstance(output, dict) or not isinstance(output.get("text"), str):
            raise RuntimeError("Whisper pipeline returned no transcript text")
        transcript = output["text"].strip()
        if not transcript:
            raise RuntimeError("Whisper returned an empty transcript")

        segments: list[TranscriptionSegment] = []
        for chunk in output.get("chunks") or ():
            if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
                continue
            timestamp = chunk.get("timestamp")
            if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
                continue
            start, end = timestamp
            if start is None or end is None:
                continue
            start_value, end_value = float(start), float(end)
            if not math.isfinite(start_value) or not math.isfinite(end_value):
                continue
            if start_value < 0 or end_value < start_value:
                continue
            segments.append(
                TranscriptionSegment(
                    text=chunk["text"].strip(),
                    start_seconds=start_value,
                    end_seconds=end_value,
                )
            )
        return transcript, tuple(segments)

    def perceive(self, item: PerceptionInput) -> Observation:
        decoded = self._decode(item)
        requested_language = item.metadata.get("language")
        transcript, segments = self._transcribe(decoded, requested_language)

        attributes: dict[str, Any] = {
            "model_family": "Whisper",
            "model_id": self.model_id,
            "model_revision": self.resolved_revision,
            "requested_revision": self.requested_revision,
            "task": "transcribe",
            "audio_format": item.content_type.split(";", 1)[0].strip().lower(),
            "original_sample_rate": decoded.original_sample_rate,
            "sample_rate": decoded.target_sample_rate,
            "original_channels": decoded.original_channels,
            "duration_seconds": decoded.duration_seconds,
            "resampled": decoded.resampled,
            "timestamp_unit": "seconds",
            "timestamp_origin": "start of supplied audio input",
            "timestamp_granularity": "pipeline chunk boundaries",
            "speaker_diarization": False,
            "audio_event_detection": False,
        }
        if requested_language is not None:
            attributes["requested_language"] = requested_language
        if segments:
            attributes["segments"] = [
                {
                    "text": segment.text,
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                }
                for segment in segments
            ]

        warnings = [
            "Transcript is model-generated and may contain recognition errors; verify important content against the audio.",
            "Chunk timestamps are segment-level boundaries, not word-level alignment.",
            "Speaker diarization and acoustic event classification are not implemented.",
        ]
        if not segments:
            warnings.append("The ASR pipeline did not return usable timestamped chunks.")

        return Observation(
            modality=Modality.AUDIO,
            status=PerceptionStatus.OBSERVED,
            summary=f"Whisper transcribed {decoded.duration_seconds:.2f} seconds of audio",
            source_input_id=item.input_id,
            content_sha256=content_digest(item.content),
            extracted_text=transcript,
            attributes=attributes,
            warnings=tuple(warnings),
        )
