"""Group observations by explicit correlation without inventing semantics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from ..contracts import Observation, PerceptionResult


@dataclass(frozen=True, slots=True)
class FusionBundle:
    correlation_id: str
    observations: tuple[Observation, ...]
    input_ids: tuple[str, ...]
    modalities: tuple[str, ...]
    completeness: str
    notes: tuple[str, ...] = ()


class SensoryFuser:
    """Collects observations sharing a caller-supplied correlation identifier.

    This is an alignment/grouping primitive, not a learned multimodal fusion
    network. It does not infer that two inputs describe the same real-world
    object or event merely because they arrived close together.
    """

    def fuse(self, results: Iterable[PerceptionResult]) -> tuple[FusionBundle, ...]:
        groups: dict[str, list[PerceptionResult]] = defaultdict(list)
        for result in results:
            if not isinstance(result, PerceptionResult):
                raise TypeError("results must contain PerceptionResult instances")
            if result.correlation_id and result.observation is not None:
                groups[result.correlation_id].append(result)

        bundles = []
        for correlation_id, items in sorted(groups.items()):
            observations = tuple(item.observation for item in items if item.observation is not None)
            modalities = tuple(sorted({item.modality.value for item in items}))
            notes = []
            if len(modalities) < 2:
                notes.append("Only one modality is represented; no cross-modal fusion performed")
            if any(item.status.value != "observed" for item in items):
                notes.append("One or more inputs did not produce a fully observed result")
            bundles.append(FusionBundle(
                correlation_id=correlation_id,
                observations=observations,
                input_ids=tuple(item.input_id for item in items),
                modalities=modalities,
                completeness="multi_modal_group" if len(modalities) > 1 else "single_modal_group",
                notes=tuple(notes),
            ))
        return tuple(bundles)
