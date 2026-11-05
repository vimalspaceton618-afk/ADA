# ADA Perception

03_PERCEPTION/ is the canonical source tree for ADA's perception boundary. The stable Python import surface is ada.perception. This subsystem receives external inputs, validates modality contracts, produces provenance-preserving observations, and exposes adapter interfaces for modality-specific models.

Perception is not cognition: it records and structures evidence. It must not silently infer intent, decide actions, or manufacture unsupported semantic outputs.

## Architecture map

| Component | Source | Responsibility | Current capability |
|---|---|---|---|
| Contracts | contracts.py | Modality, input, observation, result and evidence records | Implemented |
| Input interface | input_interface/normalizer.py | Fail-closed modality checks and size limits | Implemented |
| Text perception | text_perception/processor.py | Exact text preservation and deterministic surface statistics | Baseline implemented |
| Structured perception | structured_perception/processor.py | JSON-compatible structure and shape inspection | Baseline implemented |
| Visual perception | visual_perception/adapter.py | Image/video model adapter contract | Interface only |
| Audio perception | audio_perception/adapter.py | ASR/acoustic model adapter contract | Interface only |
| Sensory fusion | sensory_fusion/fuser.py | Group observations by explicit correlation ID | Deterministic grouping only |
| Pattern detection | pattern_detection/detector.py | Repeated token n-gram detection | Surface baseline |
| Entity recognition | entity_recognition/recognizer.py | Source-grounded entity extraction contract | Interface only |
| Event detection | event_detection/detector.py | Source-grounded temporal event contract | Interface only |
| Validation | perception_validation/validator.py | Input/result/digest/provenance integrity checks | Implemented |
| Dispatch | engine.py | Normalize, select adapter, validate adapter output | Implemented |

## Public API

    from ada.perception import (
        Modality,
        PerceptionInput,
        PerceptionEngine,
        PerceptionValidator,
    )

    engine = PerceptionEngine()
    source = PerceptionInput(
        modality=Modality.TEXT,
        content="ADA receives this text as source evidence.",
        source="user",
        correlation_id="session-001",
    )
    result = engine.perceive(source)
    report = PerceptionValidator().validate(source, result)

The built-in engine supports deterministic text and structured-data observation. Image, audio, and video inputs are rejected when malformed and return UNSUPPORTED when valid but no real modality adapter is registered. Registering an adapter requires a concrete implementation that returns a provenance-matched Observation.

## Core invariants

- Every input has a stable input ID, modality, source and receipt timestamp.
- Correlation IDs are explicit; arrival-time proximity does not imply shared meaning.
- Content digests preserve source provenance.
- Input and observation metadata are deeply immutable JSON-compatible values.
- Binary modalities require raw bytes and an explicit content type.
- Input size limits are enforced before adapter dispatch.
- Adapter outputs must match the source input ID, modality and normalized content digest.
- Adapter exceptions are converted into explicit FAILED results.
- Missing adapters produce UNSUPPORTED, never fabricated captions, transcripts, embeddings or labels.
- Confidence is optional and must be a finite value in [0, 1] when supplied.

## Fusion boundary

SensoryFuser groups successful observations only when they share an explicit correlation_id. It does not perform learned cross-attention, semantic alignment, object identity resolution, or contradiction arbitration. Those require a later trained fusion component and explicit alignment/evaluation data.

## Model integration boundary

Visual and audio adapter protocols are defined, but no image encoder, OCR model, ASR model, video encoder, or pretrained weights are bundled. The subsystem is model-agnostic and does not add mandatory ML dependencies. Future adapters must document model identity, version, preprocessing, licenses, confidence semantics, timestamp conventions, and source offsets.

## Downstream interfaces

PerceptionResult and Observation are intended to be consumed by 04_REPRESENTATION. Memory and cognition must consume normalized representations rather than reaching into raw modality-specific buffers. The runtime can schedule perception tasks through its public task contracts, but perception code does not own scheduling or autonomous execution.

## Operational limits

The current implementation is an engineering foundation, not a complete multimodal intelligence system. It does not claim semantic text understanding, language identification, trained NER, visual recognition, audio transcription, video understanding, learned multimodal fusion, or production benchmark performance.
