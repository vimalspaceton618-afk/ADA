"""ADA canonical versioned representation subsystem."""

from .adapters import (
    AudioRepresentationAdapter,
    ImageRepresentationAdapter,
    RepresentationAdapter,
    StructuredRepresentationAdapter,
    TextRepresentationAdapter,
    VideoRepresentationAdapter,
)
from .contracts import (
    SCHEMA_VERSION,
    Representation,
    RepresentationKind,
    RepresentationResult,
    RepresentationResultStatus,
    RepresentationStatus,
)
from .engine import RepresentationEngine
from .multimodal_representation import MultimodalRepresentation
from .relational_representation import RelationEdge
from .representation_alignment import RepresentationAligner
from .spatial_representation import SpatialPoint, SpatialRegion
from .symbolic_representation import shape_payload, validate_shape_tree
from .temporal_representation import validate_timed_records
from .token_representation import LexicalToken, tokenize_text, tokens_to_payload
from .vector_representation import validate_vector

__all__ = [
    "SCHEMA_VERSION",
    "Representation",
    "RepresentationKind",
    "RepresentationResult",
    "RepresentationResultStatus",
    "RepresentationStatus",
    "RepresentationEngine",
    "RepresentationAdapter",
    "TextRepresentationAdapter",
    "StructuredRepresentationAdapter",
    "ImageRepresentationAdapter",
    "AudioRepresentationAdapter",
    "VideoRepresentationAdapter",
    "LexicalToken",
    "tokenize_text",
    "tokens_to_payload",
    "validate_vector",
    "shape_payload",
    "validate_shape_tree",
    "validate_timed_records",
    "SpatialPoint",
    "SpatialRegion",
    "RelationEdge",
    "MultimodalRepresentation",
    "RepresentationAligner",
]
