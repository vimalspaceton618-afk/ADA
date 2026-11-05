from types import SimpleNamespace

import pytest

from ada.perception import (
    Modality,
    Observation,
    PerceptionEngine,
    PerceptionInput,
    PerceptionStatus,
    PerceptionValidator,
    SigLIPVideoAdapter,
)
from importlib import import_module

content_digest = import_module("03_PERCEPTION.contracts").content_digest


video_module = import_module("03_PERCEPTION.video_perception.siglip_video")
preprocessing_module = import_module("03_PERCEPTION.video_perception.preprocessing")


class FakeFrame:
    def __init__(self, timestamp, *, width=320, height=240):
        self.time = timestamp
        self.width = width
        self.height = height


class FakeContainer:
    def __init__(self, frames, *, duration=None, stream=None):
        self._frames = frames
        self.duration = duration
        self.streams = SimpleNamespace(video=[stream or SimpleNamespace(index=0, time_base=1)])
        self.format = SimpleNamespace(name="test-container")
        self.closed = False

    def decode(self, *, video):
        assert video == 0
        yield from self._frames

    def close(self):
        self.closed = True


class FakeVisionAdapter:
    modality = Modality.IMAGE

    def perceive(self, item):
        assert item.modality is Modality.IMAGE
        return Observation(
            modality=Modality.IMAGE,
            status=PerceptionStatus.OBSERVED,
            summary="fake image embedding",
            source_input_id=item.input_id,
            content_sha256=content_digest(item.content),
            attributes={
                "model_id": "google/siglip-base-patch16-224",
                "model_revision": "test-revision",
                "embedding": [0.6, 0.8],
            },
        )


def test_video_adapter_samples_frames_and_preserves_source_and_timestamps(monkeypatch):
    frames = [FakeFrame(value) for value in (0.0, 1.0, 4.9, 5.0, 8.0, 10.0)]
    container = FakeContainer(frames, duration=11_000_000)
    monkeypatch.setattr(
        video_module.SigLIPVideoAdapter,
        "_encode_frame",
        staticmethod(lambda frame: f"jpeg-{frame.time}".encode()),
    )
    adapter = SigLIPVideoAdapter(
        vision_adapter=FakeVisionAdapter(),
        sample_interval_seconds=5.0,
        max_frames=4,
        container_opener=lambda file_obj, mode: container,
    )
    engine = PerceptionEngine()
    engine.register_adapter(adapter)
    source = PerceptionInput(
        modality=Modality.VIDEO,
        content=b"synthetic-video-container",
        content_type="video/mp4",
        correlation_id="video-session-1",
    )

    result = engine.perceive(source)
    assert result.status is PerceptionStatus.PARTIAL
    assert result.observation is not None
    assert result.observation.content_sha256 == content_digest(source.content)
    assert result.observation.attributes["sampled_frame_count"] == 3
    sampled = result.observation.attributes["sampled_frames"]
    assert [frame["timestamp_seconds"] for frame in sampled] == [0.0, 5.0, 10.0]
    assert all(frame["embedding"] == (0.6, 0.8) or frame["embedding"] == [0.6, 0.8] for frame in sampled)
    assert result.observation.attributes["temporal_reasoning"] is False
    assert result.observation.attributes["audio_extracted"] is False
    assert container.closed is True
    assert PerceptionValidator().validate(source, result).valid is True


def test_video_decoder_enforces_max_frames_and_duration(monkeypatch):
    frames = [FakeFrame(value) for value in (0.0, 1.0, 5.0, 10.0, 15.0)]
    container = FakeContainer(frames, duration=16_000_000)
    decoded = preprocessing_module.decode_video(
        b"video",
        "video/mp4",
        sample_interval_seconds=5,
        max_frames=2,
        max_duration_seconds=20,
        container_opener=lambda file_obj, mode: container,
    )
    assert [frame.timestamp_seconds for frame in decoded.frames] == [0.0, 10.0]
    assert container.closed is True

    long_container = FakeContainer([], duration=21_000_000)
    with pytest.raises(preprocessing_module.VideoDecodeError, match="duration exceeds"):
        preprocessing_module.decode_video(
            b"video",
            "video/mp4",
            max_duration_seconds=20,
            container_opener=lambda file_obj, mode: long_container,
        )
    assert long_container.closed is True


def test_video_decoder_rejects_invalid_mime_frame_size_and_missing_video():
    with pytest.raises(preprocessing_module.VideoDecodeError, match="unsupported"):
        preprocessing_module.decode_video(b"video", "application/octet-stream")

    large_frame = FakeFrame(0.0, width=10_000, height=10_000)
    container = FakeContainer([large_frame])
    with pytest.raises(preprocessing_module.VideoDecodeError, match="pixel limit"):
        preprocessing_module.decode_video(
            b"video",
            "video/mp4",
            container_opener=lambda file_obj, mode: container,
        )
    assert container.closed is True

    empty = FakeContainer([])
    empty.streams = SimpleNamespace(video=[])
    with pytest.raises(preprocessing_module.VideoDecodeError, match="video stream"):
        preprocessing_module.decode_video(
            b"video",
            "video/mp4",
            container_opener=lambda file_obj, mode: empty,
        )
    assert empty.closed is True


def test_video_adapter_requires_image_adapter_and_exports_capability_boundaries():
    with pytest.raises(TypeError, match="IMAGE"):
        SigLIPVideoAdapter(vision_adapter=SimpleNamespace(modality=Modality.AUDIO))
    capabilities = SigLIPVideoAdapter.capabilities
    assert capabilities.supports_frame_sampling is True
    assert capabilities.supports_frame_embeddings is True
    assert capabilities.supports_temporal_reasoning is False
    assert capabilities.supports_audio_extraction is False
