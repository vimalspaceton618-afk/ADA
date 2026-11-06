# ADA Representation

04_REPRESENTATION/ is the versioned intermediate-representation boundary between 03_PERCEPTION/ and downstream memory, cognition, world-model, and reasoning components.

It converts source-grounded perception observations into typed, immutable, serializable projections. It does not rewrite the source observation, infer unsupported semantics, or merge unrelated modalities.

## Public API

    from ada.perception import Modality, PerceptionEngine, PerceptionInput
    from ada.representation import RepresentationEngine

    perception = PerceptionEngine()
    representation = RepresentationEngine()

    source = PerceptionInput(
        modality=Modality.TEXT,
        content="ADA preserves source-grounded text.",
        correlation_id="session-001",
    )
    observed = perception.perceive(source)
    result = representation.transform(observed)

    for item in result.representations:
        print(item.kind.value, item.representation_digest)
        serializable = item.to_dict()

## v1 projections

| Source observation | Representation | Semantics |
|---|---|---|
| TEXT | TOKEN | Unicode lexical tokens with exact source character offsets; not model token IDs |
| STRUCTURED | SYMBOLIC | Perception-provided shape tree and top-level key counts; no semantic interpretation |
| IMAGE | VECTOR | Validated SigLIP image feature vector with model identity/revision |
| AUDIO | TOKEN + TEMPORAL | Whisper transcript tokens and pipeline chunk boundaries in seconds |
| VIDEO | VECTOR + TEMPORAL | Independent SigLIP frame vectors and sparse frame timestamps |

Representation payloads are deeply immutable, JSON-compatible, and carry the source input ID and SHA-256 digest. The schema_version is currently 1.0.0. The JSON Schema document is at schemas/representation.schema.json and declares JSON Schema Draft 2020-12.

## Pipeline boundary

1. PerceptionEngine produces a PerceptionResult.
2. RepresentationEngine.transform() validates the perception/observation identity and digest.
3. A modality adapter produces one or more typed Representation records.
4. The engine verifies that every output preserves the source ID, modality, and digest.
5. Unsupported, rejected, and failed perception results remain explicit; they do not become empty successful representations.

The engine cannot re-hash raw source bytes because PerceptionResult carries the digest rather than the original input. It therefore verifies all provenance links available at this boundary.

## Alignment boundary

RepresentationAligner groups successful representation results only when they share an explicit caller-provided correlation_id. The resulting MultimodalRepresentation records the exact basis for grouping and sets semantic_fusion_performed=False.

This is not learned cross-modal alignment, shared embedding-space projection, semantic equivalence, entity resolution, or contradiction arbitration.

## Contract-only areas

- Spatial: typed coordinate/region contracts exist. No current perception adapter emits spatial grounding.
- Relational: typed provenance-linked relation edges exist. No automatic relation extraction is enabled.
- Multimodal: explicit-correlation bundles exist. No learned fusion or semantic alignment is implemented.

These boundaries are intentionally explicit because the current SigLIP image adapter does not emit boxes or coordinates, and the perception subsystem does not provide trained entity/relation outputs.

## Resource and correctness properties

- No mandatory third-party dependencies.
- No model checkpoint loading or inference occurs in this folder.
- Vectors are checked for dimension, numeric type, and finite values; they are not silently renormalized.
- Token offsets are zero-based, half-open Unicode code-point offsets into the exact extracted text.
- Audio/video timestamps are seconds relative to the source media timeline described by the perception observation.
- Metadata/payloads reject unsupported Python objects and non-finite floats.
- A source result marked PARTIAL remains PARTIAL downstream.
- Adapters are replaceable by modality and can be explicitly registered or removed.

## Verification

The test suite covers perception handoff, fake/controlled observations, immutable payloads, schema serialization round-trips, lexical offsets, symbolic shape conversion, vector validation, temporal timestamps, explicit correlation grouping, and fail-closed output provenance.

These tests do not demonstrate real model quality, calibrated confidence, semantic multimodal fusion, or spatial grounding.
