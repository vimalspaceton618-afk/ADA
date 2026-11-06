"""Grouped multi-modal representation records; grouping is not semantic fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid

from ..contracts import Representation


@dataclass(frozen=True, slots=True)
class MultimodalRepresentation:
    correlation_id: str
    representations: tuple[Representation, ...]
    input_ids: tuple[str, ...]
    modalities: tuple[str, ...]
    alignment_basis: str = "explicit_correlation_id"
    semantic_fusion_performed: bool = False
    bundle_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not isinstance(self.correlation_id, str) or not self.correlation_id.strip():
            raise ValueError("correlation_id must be non-empty")
        representations = tuple(self.representations)
        if not representations or any(
            not isinstance(item, Representation) for item in representations
        ):
            raise ValueError("bundle requires Representation values")
        if self.alignment_basis != "explicit_correlation_id":
            raise ValueError("only explicit correlation-ID grouping is supported")
        if self.semantic_fusion_performed is not False:
            raise ValueError("semantic fusion is not implemented by this bundle")
        input_ids = tuple(self.input_ids)
        modalities = tuple(sorted(set(self.modalities)))
        if not input_ids or any(not value for value in input_ids):
            raise ValueError("bundle input_ids must be non-empty")
        if not modalities:
            raise ValueError("bundle modalities must be non-empty")
        if any(item.source_input_id not in input_ids for item in representations):
            raise ValueError("every representation must reference a bundle input")
        if any(item.modality not in modalities for item in representations):
            raise ValueError("every representation modality must be declared in the bundle")
        object.__setattr__(self, "representations", representations)
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "modalities", modalities)
