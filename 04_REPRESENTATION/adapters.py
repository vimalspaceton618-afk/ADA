"""Deterministic adapters from perception observations to typed representations."""

from __future__ import annotations

import math

from collections.abc import Mapping
from typing import Any, Protocol

from .contracts import (
    Representation,
    RepresentationKind,
    RepresentationStatus,
)
from .symbolic_representation.structure import shape_payload
from .temporal_representation.timeline import validate_timed_records
from .token_representation.tokenizer import tokens_to_payload
from .vector_representation.vector import validate_vector


class RepresentationAdapter(Protocol):
    modality_value: str

    def represent(self, result: Any) -> tuple[Representation, ...]:
        """Create zero or more provenance-preserving representations."""
        ...


def _representation_status(result: Any) -> RepresentationStatus:
    if getattr(result.status, "value", None) == "partial":
        return RepresentationStatus.PARTIAL
    return RepresentationStatus.COMPLETE




def _validate_candidate_scores(value: Any) -> tuple[dict[str, Any], ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError("candidate_scores must be a sequence")
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"candidate_scores[{index}] must be a mapping")
        label = item.get("label")
        score = item.get("sigmoid_score")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"candidate_scores[{index}] label must be non-empty")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(f"candidate_scores[{index}] sigmoid_score must be numeric")
        score_value = float(score)
        if not math.isfinite(score_value) or not 0.0 <= score_value <= 1.0:
            raise ValueError(f"candidate_scores[{index}] sigmoid_score must be finite in [0, 1]")
        validated.append({"label": label, "sigmoid_score": score_value})
    return tuple(validated)

def _new_representation(
    result: Any,
    *,
    kind: RepresentationKind,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Representation:
    observation = result.observation
    return Representation(
        kind=kind,
        modality=result.modality.value,
        source_input_id=result.input_id,
        source_content_sha256=result.content_sha256,
        payload=payload,
        metadata={
            "source_observation_summary": observation.summary,
            "source_observed_at": observation.observed_at.isoformat(),
            **dict(metadata or {}),
        },
        status=_representation_status(result),
    )


class TextRepresentationAdapter:
    """Creates lexical surface tokens while preserving character offsets."""

    modality_value = "text"

    def represent(self, result: Any) -> tuple[Representation, ...]:
        text = result.observation.extracted_text
        if not isinstance(text, str):
            raise ValueError("text observation is missing extracted_text")
        return (_new_representation(
            result,
            kind=RepresentationKind.TOKEN,
            payload=tokens_to_payload(text),
            metadata={"transform": "ada.lexical.v1", "semantic_analysis": "not_performed"},
        ),)


class StructuredRepresentationAdapter:
    """Preserves the perceived structural shape without inventing values."""

    modality_value = "structured"

    def represent(self, result: Any) -> tuple[Representation, ...]:
        payload = shape_payload(result.observation.attributes)
        return (_new_representation(
            result,
            kind=RepresentationKind.SYMBOLIC,
            payload=payload,
            metadata={"transform": "ada.symbolic_shape.v1"},
        ),)


class ImageRepresentationAdapter:
    """Validates and carries a learned image feature vector."""

    modality_value = "image"

    def represent(self, result: Any) -> tuple[Representation, ...]:
        attributes = result.observation.attributes
        embedding = attributes.get("embedding")
        expected = attributes.get("embedding_dimension")
        vector, norm = validate_vector(
            embedding if isinstance(embedding, (list, tuple)) else (),
            expected_dimension=expected,
        )
        payload: dict[str, Any] = {
            "values": list(vector),
            "dimension": len(vector),
            "l2_norm": norm,
            "vector_semantics": attributes.get(
                "embedding_type", "model_feature_vector"
            ),
        }
        candidate_scores = _validate_candidate_scores(attributes.get("candidate_scores"))
        if candidate_scores is not None:
            payload["candidate_scores"] = list(candidate_scores)
            payload["candidate_score_semantics"] = attributes.get(
                "candidate_score_semantics",
                "sigmoid_image_text_matching_scores_not_calibrated_probabilities",
            )
        return (_new_representation(
            result,
            kind=RepresentationKind.VECTOR,
            payload=payload,
            metadata={
                "model_id": attributes.get("model_id"),
                "model_revision": attributes.get("model_revision"),
                "transform": "passthrough_validated_embedding",
            },
        ),)


class AudioRepresentationAdapter:
    """Creates lexical and segment-timeline views from Whisper observations."""

    modality_value = "audio"

    def represent(self, result: Any) -> tuple[Representation, ...]:
        observation = result.observation
        output: list[Representation] = []
        if isinstance(observation.extracted_text, str):
            output.append(_new_representation(
                result,
                kind=RepresentationKind.TOKEN,
                payload=tokens_to_payload(observation.extracted_text),
                metadata={
                    "transform": "ada.lexical.v1",
                    "text_source": "whisper_transcript",
                    "model_id": observation.attributes.get("model_id"),
                    "model_revision": observation.attributes.get("model_revision"),
                },
            ))

        raw_segments = observation.attributes.get("segments")
        if raw_segments:
            if not isinstance(raw_segments, (list, tuple)):
                raise TypeError("audio segments must be a sequence")
            segments = validate_timed_records(
                raw_segments,
                start_key="start_seconds",
                end_key="end_seconds",
                duration_seconds=observation.attributes.get("duration_seconds"),
            )
            output.append(_new_representation(
                result,
                kind=RepresentationKind.TEMPORAL,
                payload={
                    "time_unit": "seconds",
                    "time_origin": "start_of_supplied_audio",
                    "timestamp_granularity": "pipeline_chunk_boundaries",
                    "segments": list(segments),
                },
                metadata={
                    "transform": "whisper_segment_timeline.v1",
                    "model_id": observation.attributes.get("model_id"),
                    "model_revision": observation.attributes.get("model_revision"),
                },
            ))
        if not output:
            raise ValueError("audio observation contains no transcript or timed segments")
        return tuple(output)


class VideoRepresentationAdapter:
    """Creates per-frame vectors and a sparse timestamped frame timeline."""

    modality_value = "video"

    def represent(self, result: Any) -> tuple[Representation, ...]:
        observation = result.observation
        attributes = observation.attributes
        raw_frames = attributes.get("sampled_frames")
        if not isinstance(raw_frames, (list, tuple)) or not raw_frames:
            raise ValueError("video observation is missing sampled_frames")

        duration = attributes.get("video_duration_seconds")
        frames = validate_timed_records(
            raw_frames,
            start_key="timestamp_seconds",
            duration_seconds=duration,
        )
        outputs: list[Representation] = []
        timeline_frames = []

        for frame in frames:
            embedding = frame.get("embedding")
            vector, norm = validate_vector(
                embedding if isinstance(embedding, (list, tuple)) else ()
            )
            frame_index = frame.get("frame_index")
            if type(frame_index) is not int or frame_index < 0:
                raise ValueError("video frame_index must be a non-negative integer")
            frame_payload: dict[str, Any] = {
                "values": list(vector),
                "dimension": len(vector),
                "l2_norm": norm,
                "vector_semantics": "independent_siglip_frame_embedding",
            }
            candidate_scores = _validate_candidate_scores(frame.get("candidate_scores"))
            if candidate_scores is not None:
                frame_payload["candidate_scores"] = list(candidate_scores)
                frame_payload["candidate_score_semantics"] = (
                    "sigmoid_image_text_matching_scores_not_calibrated_probabilities"
                )
            outputs.append(_new_representation(
                result,
                kind=RepresentationKind.VECTOR,
                payload=frame_payload,
                metadata={
                    "transform": "siglip_video_frame_vector.v1",
                    "frame_index": frame_index,
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "width": frame.get("width"),
                    "height": frame.get("height"),
                    "model_id": attributes.get("model_id"),
                    "model_revision": attributes.get("model_revision"),
                },
            ))
            timeline_frames.append({
                "frame_index": frame_index,
                "timestamp_seconds": frame["timestamp_seconds"],
                "width": frame.get("width"),
                "height": frame.get("height"),
            })

        outputs.append(_new_representation(
            result,
            kind=RepresentationKind.TEMPORAL,
            payload={
                "time_unit": "seconds",
                "time_origin": "video_presentation_timestamp",
                "sampling_semantics": "sparse_independent_frames",
                "frames": timeline_frames,
            },
            metadata={
                "transform": "siglip_sparse_video_timeline.v1",
                "duration_seconds": duration,
                "requested_sampling_interval_seconds": attributes.get(
                    "requested_sampling_interval_seconds"
                ),
                "effective_sampling_interval_seconds": attributes.get(
                    "effective_sampling_interval_seconds"
                ),
                "temporal_reasoning": False,
            },
        ))
        return tuple(outputs)


def default_adapters() -> tuple[RepresentationAdapter, ...]:
    return (
        TextRepresentationAdapter(),
        StructuredRepresentationAdapter(),
        ImageRepresentationAdapter(),
        AudioRepresentationAdapter(),
        VideoRepresentationAdapter(),
    )
