"""Sparse video-frame feature extraction using the existing SigLIP image encoder.

This adapter is intentionally not a temporal video model. It samples frames,
encodes each frame independently, and preserves timestamps and provenance.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from ..contracts import Modality, Observation, PerceptionInput, PerceptionStatus, content_digest
from ..visual_perception.siglip import SigLIPVisionAdapter
from .adapter import VideoAdapterCapabilities
from .preprocessing import DecodedVideo, VideoDecodeError, decode_video


class SigLIPVideoAdapter:
    """Sample video frames and encode them independently with SigLIP.

    The injected vision adapter must be a working image adapter. Model weights
    are not downloaded or duplicated by this class.
    """

    modality = Modality.VIDEO
    capabilities = VideoAdapterCapabilities(
        supports_frame_sampling=True,
        supports_frame_embeddings=True,
        supports_temporal_reasoning=False,
        supports_audio_extraction=False,
    )

    def __init__(
        self,
        *,
        vision_adapter: SigLIPVisionAdapter,
        sample_interval_seconds: float = 5.0,
        max_frames: int = 12,
        max_duration_seconds: float = 600.0,
        max_video_bytes: int = 50 * 1024 * 1024,
        max_frame_pixels: int = 40_000_000,
        max_decoded_frames: int = 100_000,
        container_opener: Any | None = None,
    ) -> None:
        if getattr(vision_adapter, "modality", None) is not Modality.IMAGE:
            raise TypeError("vision_adapter must process IMAGE inputs")
        if not callable(getattr(vision_adapter, "perceive", None)):
            raise TypeError("vision_adapter must provide perceive(input)")
        self.vision_adapter = vision_adapter
        self.sample_interval_seconds = sample_interval_seconds
        self.max_frames = max_frames
        self.max_duration_seconds = max_duration_seconds
        self.max_video_bytes = max_video_bytes
        self.max_frame_pixels = max_frame_pixels
        self.max_decoded_frames = max_decoded_frames
        self.container_opener = container_opener

    @staticmethod
    def _encode_frame(frame: Any) -> bytes:
        """Encode a decoded AV frame as bounded JPEG bytes for the image adapter."""
        try:
            image = frame.to_image()
        except ImportError as exc:
            raise ImportError(
                "Video frame encoding requires Pillow; install perception-vision"
            ) from exc
        rgb_image = image.convert("RGB")
        try:
            buffer = BytesIO()
            rgb_image.save(
                buffer,
                format="JPEG",
                quality=90,
                optimize=True,
            )
            encoded = buffer.getvalue()
        finally:
            for candidate in (rgb_image, image):
                close = getattr(candidate, "close", None)
                if callable(close):
                    close()
        if not encoded:
            raise VideoDecodeError("video frame encoder produced empty image bytes")
        return encoded

    def _decode(self, item: PerceptionInput) -> DecodedVideo:
        if item.modality is not Modality.VIDEO:
            raise TypeError("SigLIPVideoAdapter only accepts VIDEO inputs")
        if not isinstance(item.content, bytes):
            raise TypeError("VIDEO input content must be bytes")
        return decode_video(
            item.content,
            item.content_type,
            sample_interval_seconds=self.sample_interval_seconds,
            max_frames=self.max_frames,
            max_duration_seconds=self.max_duration_seconds,
            max_video_bytes=self.max_video_bytes,
            max_frame_pixels=self.max_frame_pixels,
            max_decoded_frames=self.max_decoded_frames,
            container_opener=self.container_opener,
        )

    def perceive(self, item: PerceptionInput) -> Observation:
        decoded = self._decode(item)
        candidate_labels = item.metadata.get("candidate_labels")
        frame_records: list[dict[str, Any]] = []
        model_id: str | None = None
        model_revision: str | None = None

        for sampled in decoded.frames:
            image_bytes = self._encode_frame(sampled.frame)
            metadata = {}
            if candidate_labels is not None:
                metadata["candidate_labels"] = list(candidate_labels)
            frame_input = PerceptionInput(
                modality=Modality.IMAGE,
                content=image_bytes,
                source=item.source,
                input_id=f"{item.input_id}:frame:{sampled.frame_index}",
                correlation_id=item.correlation_id,
                content_type="image/jpeg",
                received_at=item.received_at,
                metadata=metadata,
            )
            frame_observation = self.vision_adapter.perceive(frame_input)
            if not isinstance(frame_observation, Observation):
                raise TypeError("vision adapter must return Observation")
            if frame_observation.modality is not Modality.IMAGE:
                raise ValueError("vision adapter returned a non-image observation")
            if frame_observation.source_input_id != frame_input.input_id:
                raise ValueError("vision adapter returned mismatched frame provenance")
            if frame_observation.content_sha256 != content_digest(image_bytes):
                raise ValueError("vision adapter returned a mismatched frame digest")

            attributes = frame_observation.attributes
            embedding = attributes.get("embedding")
            if not isinstance(embedding, (list, tuple)) or not embedding:
                raise ValueError("vision adapter must return an image embedding per sampled frame")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in embedding
            ):
                raise ValueError("frame embedding must contain numeric values")

            model_id = str(attributes.get("model_id", model_id or "unspecified"))
            model_revision = str(
                attributes.get("model_revision", model_revision or "unspecified")
            )
            frame_record: dict[str, Any] = {
                "frame_index": sampled.frame_index,
                "timestamp_seconds": sampled.timestamp_seconds,
                "width": sampled.width,
                "height": sampled.height,
                "embedding": [float(value) for value in embedding],
            }
            scores = attributes.get("candidate_scores")
            if scores is not None:
                frame_record["candidate_scores"] = scores
            frame_records.append(frame_record)

        if not frame_records:
            raise VideoDecodeError("no video frames were encoded")

        warnings = [
            "Video is represented by sparse independent frame embeddings; no temporal model is used.",
            "The audio track is ignored. Register audio perception separately for transcription.",
            "Frame sampling can miss brief events between selected timestamps.",
        ]
        if decoded.duration_seconds is None:
            warnings.append(
                "Container duration was unavailable; bounded sampling may cover only the initial portion."
            )

        return Observation(
            modality=Modality.VIDEO,
            status=PerceptionStatus.PARTIAL,
            summary=(
                f"Extracted independent SigLIP embeddings from "
                f"{len(frame_records)} sampled video frames"
            ),
            source_input_id=item.input_id,
            content_sha256=content_digest(item.content),
            attributes={
                "model_family": "SigLIP frame encoder",
                "model_id": model_id or "unspecified",
                "model_revision": model_revision or "unspecified",
                "container_format": decoded.container_format,
                "video_duration_seconds": decoded.duration_seconds,
                "requested_sampling_interval_seconds": float(self.sample_interval_seconds),
                "effective_sampling_interval_seconds": decoded.effective_sample_interval_seconds,
                "decoded_frame_count": decoded.decoded_frame_count,
                "sampled_frame_count": len(frame_records),
                "sampled_frames": frame_records,
                "frame_embedding_semantics": "independent L2-normalized image features",
                "temporal_reasoning": False,
                "audio_extracted": False,
            },
            warnings=tuple(warnings),
        )
