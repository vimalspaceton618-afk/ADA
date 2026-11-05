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
| Visual perception | visual_perception/adapter.py, visual_perception/siglip.py | Image adapter contract and pretrained SigLIP feature extraction | SigLIP image inference implemented; OCR/object detection deferred |
| Audio perception | audio_perception/adapter.py, audio_perception/preprocessing.py, audio_perception/whisper.py | WAV/FLAC decoding and Whisper ASR | Optional transcription adapter implemented; diarization/event detection deferred |\n| Video perception | video_perception/adapter.py, video_perception/preprocessing.py, video_perception/siglip_video.py | Bounded PyAV frame sampling and per-frame SigLIP embeddings | Optional sparse frame feature extraction implemented; temporal reasoning/audio extraction deferred |
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
        SigLIPVisionAdapter,
        WhisperAudioAdapter,
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

    # Optional pretrained image feature path (checkpoint must be cached locally).
    visual = SigLIPVisionAdapter.from_pretrained(
        "./models/siglip-base-patch16-224",
        local_files_only=True,
        revision="main",  # Pin a Hub commit SHA for reproducible deployments.
    )
    engine.register_adapter(visual)
    with open("sample.jpg", "rb") as image_file:
        image_bytes = image_file.read()
    image_input = PerceptionInput(
        modality=Modality.IMAGE,
        content=image_bytes,
        content_type="image/jpeg",
        metadata={"candidate_labels": ["a cat", "a dog", "a vehicle"]},
    )
    image_result = engine.perceive(image_input)

The built-in engine supports deterministic text and structured-data observation. SigLIP image, Whisper audio, and SigLIP-backed video adapters are optional and must be explicitly registered before inference. Missing adapters return UNSUPPORTED.

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

## Visual model integration: SigLIP Base

ADA includes an optional adapter for Google SigLIP Base (ViT-B/16, approximately 203.2M parameters). The Hugging Face model repository identifies the checkpoint as Apache-2.0 and documents image-text retrieval and zero-shot image classification. It was pretrained on English image-text pairs from WebLI and uses 224x224 image preprocessing. See the [SigLIP model card](https://huggingface.co/google/siglip-base-patch16-224) and [Transformers SigLIP API](https://huggingface.co/docs/transformers/main/model_doc/siglip).

Install optional dependencies:

    python -m pip install -e ".[perception-vision]"

Download the checkpoint into a local directory (approximately 813 MB):

    hf download google/siglip-base-patch16-224 --local-dir ./models/siglip-base-patch16-224

Load the model from that directory:

    from ada.perception import SigLIPVisionAdapter

    visual = SigLIPVisionAdapter.from_pretrained(
        "./models/siglip-base-patch16-224",
        local_files_only=True,
        revision="main",
    )
    engine.register_adapter(visual)

from_pretrained() defaults to local-only loading, uses safetensors, and sets trust_remote_code=False. Network downloads are disabled unless the caller explicitly sets local_files_only=False. For reproducible deployments, use a specific Hub commit SHA instead of the mutable main revision. The adapter records the resolved model revision in each observation.

## Audio model integration: Whisper Base

ADA includes an optional multilingual ASR adapter for OpenAI Whisper Base. Hugging Face reports approximately 72.6M parameters and Apache-2.0 model-repository licensing. The multilingual checkpoint supports speech recognition in the spoken language and speech translation; this adapter deliberately exposes transcription only. See the [Whisper Base model card](https://huggingface.co/openai/whisper-base) and [Whisper model documentation](https://github.com/openai/whisper).

Install optional dependencies:

    python -m pip install -e ".[perception-audio]"

Download the checkpoint locally (the safetensors weights are approximately 290 MB):

    hf download openai/whisper-base --local-dir ./models/whisper-base

Load and register it:

    from ada.perception import WhisperAudioAdapter

    audio = WhisperAudioAdapter.from_pretrained(
        "./models/whisper-base",
        local_files_only=True,
        revision="main",
    )
    engine.register_adapter(audio)

from_pretrained() defaults to local-only loading, uses safetensors, disables remote custom code, and uses CPU by default when CUDA is unavailable. Set local_files_only=False only to explicitly permit model downloads. Pin a Hub commit SHA for reproducible deployments.

The current decoder accepts WAV and FLAC bytes with a matching MIME content_type. It downmixes channels to mono, resamples to 16 kHz, rejects silent/non-finite audio, and enforces a configurable maximum duration (30 minutes by default). MP3/AAC and video containers are not accepted by this adapter.

Whisper returns a transcript in Observation.extracted_text. When the Transformers pipeline returns chunk boundaries, the adapter preserves segment-level timestamps in seconds relative to the supplied audio clip. An optional metadata.language value conditions the transcription language; it is not an automatically detected language field. Transcripts may contain recognition errors and must be verified for consequential uses.

This adapter does not implement speaker diarization, word-level forced alignment, voice identification, acoustic event classification, or emotion recognition. Audio embedding extraction is also not exposed by this ASR adapter.

The image input must contain JPEG, PNG, or WebP bytes and a matching MIME content_type. EXIF orientation is applied, and the decoder enforces a 40-megapixel default cap before model preprocessing. The SigLIP processor performs model-specific resize/rescale/normalization.

Every successful image observation contains a normalized image embedding, source digest, model ID/revision, and decoded image dimensions. Supplying metadata.candidate_labels requests caller-scoped image/text sigmoid scores. Those scores are not calibrated probabilities, and their ranking depends on the supplied candidate set. The adapter does not produce captions, OCR text, bounding boxes, object detections, or spatial grounding.

No model weights are committed to this repository. Whisper ASR is optional and must be explicitly registered. Video temporal encoding, diarization and acoustic event models are not bundled. Future adapters must document model identity, version, preprocessing, licenses, confidence semantics, timestamp conventions, and source offsets.

## Downstream interfaces

PerceptionResult and Observation are intended to be consumed by 04_REPRESENTATION. Memory and cognition must consume normalized representations rather than reaching into raw modality-specific buffers. The runtime can schedule perception tasks through its public task contracts, but perception code does not own scheduling or autonomous execution.

## Operational limits

The current implementation is an engineering foundation, not a complete multimodal intelligence system. It does not claim semantic text understanding, language identification, trained NER, visual recognition, audio transcription, video understanding, learned multimodal fusion, or production benchmark performance.
