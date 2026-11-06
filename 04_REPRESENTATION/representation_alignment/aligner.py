"""Alignment by explicit caller correlation only; no learned alignment is performed."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ..contracts import RepresentationResult, RepresentationResultStatus
from ..multimodal_representation.bundle import MultimodalRepresentation


class RepresentationAligner:
    """Group representations that share an explicit correlation identifier."""

    def align(
        self,
        results: Iterable[RepresentationResult],
    ) -> tuple[MultimodalRepresentation, ...]:
        groups: dict[str, list[RepresentationResult]] = defaultdict(list)
        for result in results:
            if not isinstance(result, RepresentationResult):
                raise TypeError("results must contain RepresentationResult values")
            if (
                result.correlation_id
                and result.status in {
                    RepresentationResultStatus.COMPLETE,
                    RepresentationResultStatus.PARTIAL,
                }
                and result.representations
            ):
                groups[result.correlation_id].append(result)

        bundles = []
        for correlation_id, items in sorted(groups.items()):
            representations = tuple(
                representation
                for item in items
                for representation in item.representations
            )
            input_ids = tuple(dict.fromkeys(item.input_id for item in items))
            modalities = tuple(sorted({item.modality for item in items}))
            bundles.append(MultimodalRepresentation(
                correlation_id=correlation_id,
                representations=representations,
                input_ids=input_ids,
                modalities=modalities,
            ))
        return tuple(bundles)
