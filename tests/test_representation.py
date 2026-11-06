from importlib import import_module
from datetime import datetime, timezone

import pytest

from ada.representation import (
    MultimodalRepresentation,
    RelationEdge,
    Representation,
    RepresentationAligner,
    RepresentationEngine,
    RepresentationKind,
    RepresentationResult,
    RepresentationResultStatus,
    RepresentationStatus,
    SpatialPoint,
    SpatialRegion,
    tokenize_text,
    validate_vector,
)

perception_contracts = import_module("03_PERCEPTION.contracts")
perception_engine_module = import_module("03_PERCEPTION.engine")
Modality = perception_contracts.Modality
Observation = perception_contracts.Observation
PerceptionInput = perception_contracts.PerceptionInput
PerceptionResult = perception_contracts.PerceptionResult
PerceptionStatus = perception_contracts.PerceptionStatus
PerceptionEngine = perception_engine_module.PerceptionEngine
content_digest = perception_contracts.content_digest


def make_result(
    modality,
    *,
    content=b"source",
    attributes=None,
    extracted_text=None,
    status=None,
    correlation_id=None,
):
    input_id = f"input-{modality.value}"
    digest = content_digest(content)
    observation_status = status or PerceptionStatus.OBSERVED
    observation = Observation(
        modality=modality,
        status=observation_status,
        summary=f"test {modality.value} observation",
        source_input_id=input_id,
        content_sha256=digest,
        extracted_text=extracted_text,
        attributes=attributes or {},
    )
    return PerceptionResult(
        input_id=input_id,
        correlation_id=correlation_id,
        modality=modality,
        status=observation_status,
        observation=observation,
        content_sha256=digest,
    )


def test_text_perception_becomes_lexical_tokens_with_exact_offsets():
    observed = PerceptionEngine().perceive(PerceptionInput(
        modality=Modality.TEXT,
        content="ADA's representation, v1!",
        correlation_id="text-1",
    ))
    result = RepresentationEngine().transform(observed)

    assert result.status is RepresentationResultStatus.COMPLETE
    assert len(result.representations) == 1
    rep = result.representations[0]
    assert rep.kind is RepresentationKind.TOKEN
    assert rep.payload["tokenizer_id"] == "ada.lexical.v1"
    assert rep.payload["tokenizer_semantics"] == "unicode_surface_tokens_not_model_token_ids"
    tokens = rep.payload["tokens"]
    assert tokens[0]["text"] == "ADA's"
    source = "ADA's representation, v1!"
    assert source[tokens[0]["start"]:tokens[0]["end"]] == "ADA's"
    assert rep.source_input_id == observed.input_id
    assert rep.source_content_sha256 == observed.content_sha256


def test_unicode_lexical_offsets_are_code_point_offsets():
    tokens = tokenize_text("naïve café — ADA")
    assert [item.text for item in tokens] == ["naïve", "café", "—", "ADA"]
    source = "naïve café — ADA"
    assert all(source[item.start:item.end] == item.text for item in tokens)


def test_structured_observation_becomes_shape_only_symbolic_representation():
    observed = PerceptionEngine().perceive(PerceptionInput(
        modality=Modality.STRUCTURED,
        content={"device": {"enabled": True}, "samples": [1, 2, 3]},
    ))
    result = RepresentationEngine().transform(observed)

    assert result.status is RepresentationResultStatus.COMPLETE
    rep = result.representations[0]
    assert rep.kind is RepresentationKind.SYMBOLIC
    assert rep.payload["top_level_field_count"] == 2
    assert rep.payload["interpretation"] == "shape_only_no_semantic_interpretation"
    assert rep.payload["structure"]["fields"]["device"]["type"] == "object"


def test_image_embedding_becomes_validated_vector():
    observed = make_result(
        Modality.IMAGE,
        attributes={
            "embedding": [0.6, 0.8],
            "embedding_dimension": 2,
            "embedding_type": "l2_normalized_image_feature",
            "model_id": "google/siglip-base-patch16-224",
            "model_revision": "test-revision",
        },
    )
    result = RepresentationEngine().transform(observed)

    assert result.status is RepresentationResultStatus.COMPLETE
    rep = result.representations[0]
    assert rep.kind is RepresentationKind.VECTOR
    assert rep.payload["values"] == (0.6, 0.8)
    assert rep.payload["dimension"] == 2
    assert rep.payload["l2_norm"] == pytest.approx(1.0)
    assert rep.metadata["model_id"] == "google/siglip-base-patch16-224"


def test_audio_observation_becomes_tokens_and_timestamped_segments():
    observed = make_result(
        Modality.AUDIO,
        attributes={
            "model_id": "openai/whisper-base",
            "model_revision": "test-revision",
            "duration_seconds": 2.0,
            "segments": [
                {"text": "hello", "start_seconds": 0.0, "end_seconds": 0.7},
                {"text": "world", "start_seconds": 0.8, "end_seconds": 1.6},
            ],
        },
        extracted_text="hello world",
    )
    result = RepresentationEngine().transform(observed)

    assert result.status is RepresentationResultStatus.COMPLETE
    assert [rep.kind for rep in result.representations] == [
        RepresentationKind.TOKEN,
        RepresentationKind.TEMPORAL,
    ]
    timeline = result.representations[1].payload
    assert timeline["time_unit"] == "seconds"
    assert timeline["segments"][1]["start_seconds"] == 0.8


def test_video_observation_becomes_frame_vectors_and_sparse_timeline():
    observed = make_result(
        Modality.VIDEO,
        status=PerceptionStatus.PARTIAL,
        attributes={
            "model_id": "google/siglip-base-patch16-224",
            "model_revision": "test-revision",
            "video_duration_seconds": 10.0,
            "requested_sampling_interval_seconds": 5.0,
            "effective_sampling_interval_seconds": 5.0,
            "sampled_frames": [
                {"frame_index": 0, "timestamp_seconds": 0.0, "width": 320, "height": 240, "embedding": [1.0, 0.0]},
                {"frame_index": 10, "timestamp_seconds": 5.0, "width": 320, "height": 240, "embedding": [0.0, 1.0]},
            ],
        },
    )
    result = RepresentationEngine().transform(observed)

    assert result.status is RepresentationResultStatus.PARTIAL
    assert len(result.representations) == 3
    assert [rep.kind for rep in result.representations] == [
        RepresentationKind.VECTOR,
        RepresentationKind.VECTOR,
        RepresentationKind.TEMPORAL,
    ]
    assert all(rep.status is RepresentationStatus.PARTIAL for rep in result.representations)
    timeline = result.representations[-1]
    assert timeline.payload["frames"][1]["timestamp_seconds"] == 5.0
    assert timeline.metadata["temporal_reasoning"] is False


def test_unavailable_timestamps_fail_closed_instead_of_inventing_timeline():
    observed = make_result(
        Modality.AUDIO,
        attributes={"segments": [{"text": "no timestamp"}]},
        extracted_text="transcript",
    )
    result = RepresentationEngine().transform(observed)
    assert result.status is RepresentationResultStatus.FAILED
    assert result.representations == ()
    assert "ValueError" in result.errors[0]


def test_vector_validation_rejects_non_finite_values_and_dimension_mismatch():
    with pytest.raises(ValueError, match="finite"):
        validate_vector([0.0, float("nan")])
    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_vector([1.0, 2.0], expected_dimension=3)
    with pytest.raises(TypeError, match="numeric"):
        validate_vector([1.0, True])


def test_representation_payload_is_deeply_immutable_and_json_round_trips():
    item = Representation(
        kind=RepresentationKind.SYMBOLIC,
        modality="structured",
        source_input_id="input-1",
        source_content_sha256="a" * 64,
        payload={"tree": {"keys": ["a", "b"]}},
        metadata={"adapter": "test"},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(TypeError):
        item.payload["tree"]["new"] = 1

    restored = Representation.from_dict(item.to_dict())
    assert restored.to_dict() == item.to_dict()
    assert restored.representation_digest == item.representation_digest


def test_representation_rejects_unsupported_schema_versions_and_digest_formats():
    with pytest.raises(ValueError, match="unsupported"):
        Representation(
            kind=RepresentationKind.TOKEN,
            modality="text",
            source_input_id="input",
            source_content_sha256="b" * 64,
            schema_version="2.0.0",
            payload={"tokens": []},
        )
    with pytest.raises(ValueError, match="SHA-256"):
        Representation(
            kind=RepresentationKind.TOKEN,
            modality="text",
            source_input_id="input",
            source_content_sha256="bad",
            payload={"tokens": []},
        )


def test_rejected_perception_result_is_not_converted_to_successful_representation():
    digest = content_digest(b"bad")
    observed = PerceptionResult(
        input_id="rejected-1",
        correlation_id=None,
        modality=Modality.IMAGE,
        status=PerceptionStatus.REJECTED,
        observation=None,
        content_sha256=digest,
        errors=("invalid image",),
    )
    result = RepresentationEngine().transform(observed)
    assert result.status is RepresentationResultStatus.REJECTED
    assert result.representations == ()
    assert result.errors == ("invalid image",)


def test_representation_engine_detects_bad_adapter_provenance():
    class BadAdapter:
        modality_value = "text"

        def represent(self, source):
            return (Representation(
                kind=RepresentationKind.TOKEN,
                modality="text",
                source_input_id="some-other-input",
                source_content_sha256=source.content_sha256,
                payload={"tokens": ["fake"]},
            ),)

    observed = PerceptionEngine().perceive(PerceptionInput(
        modality=Modality.TEXT,
        content="source",
    ))
    engine = RepresentationEngine(adapters=(BadAdapter(),))
    result = engine.transform(observed)
    assert result.status is RepresentationResultStatus.FAILED
    assert "provenance" in result.errors[0]


def test_aligner_groups_only_shared_explicit_correlation_ids():
    p_engine = PerceptionEngine()
    r_engine = RepresentationEngine()
    text_result = r_engine.transform(p_engine.perceive(PerceptionInput(
        modality=Modality.TEXT,
        content="hello",
        correlation_id="shared",
    )))
    audio_source = make_result(
        Modality.AUDIO,
        attributes={},
        extracted_text="hello",
        correlation_id="shared",
    )
    audio_result = r_engine.transform(audio_source)
    uncorrelated_result = r_engine.transform(p_engine.perceive(PerceptionInput(
        modality=Modality.TEXT,
        content="unlinked",
    )))

    groups = RepresentationAligner().align(
        [text_result, audio_result, uncorrelated_result]
    )
    assert len(groups) == 1
    group = groups[0]
    assert isinstance(group, MultimodalRepresentation)
    assert group.correlation_id == "shared"
    assert set(group.modalities) == {"text", "audio"}
    assert group.semantic_fusion_performed is False
    assert group.alignment_basis == "explicit_correlation_id"


def test_spatial_and_relational_contracts_validate_provenance_and_frames():
    low = SpatialPoint((0, 0), coordinate_frame="image_pixels", unit="pixel")
    high = SpatialPoint((100, 80), coordinate_frame="image_pixels", unit="pixel")
    region = SpatialRegion(low, high)
    assert region.minimum.coordinates == (0.0, 0.0)

    with pytest.raises(ValueError, match="coordinate frame"):
        SpatialRegion(low, SpatialPoint((1, 1), coordinate_frame="world", unit="pixel"))
    edge = RelationEdge(
        source_representation_id="rep-a",
        target_representation_id="rep-b",
        relation_type="explicitly_related_to",
        evidence_input_ids=("input-a", "input-b"),
    )
    assert edge.direction == "directed"
    with pytest.raises(ValueError, match="provenance"):
        RelationEdge("rep-a", "rep-b", "related", ())


def test_representation_result_checks_source_provenance():
    item = Representation(
        kind=RepresentationKind.TOKEN,
        modality="text",
        source_input_id="different-input",
        source_content_sha256="c" * 64,
        payload={"tokens": []},
    )
    with pytest.raises(ValueError, match="source_input_id"):
        RepresentationResult(
            input_id="expected-input",
            correlation_id=None,
            modality="text",
            source_content_sha256="c" * 64,
            status=RepresentationResultStatus.COMPLETE,
            representations=(item,),
        )
