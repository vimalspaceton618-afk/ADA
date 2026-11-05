"""Deterministic structural inspection for JSON-compatible mapping inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import Modality, Observation, PerceptionInput, PerceptionStatus, content_digest


def _shape(value: Any, depth: int = 0, max_depth: int = 8) -> Any:
    if depth >= max_depth:
        return "max_depth"
    if isinstance(value, Mapping):
        return {
            "type": "object",
            "keys": len(value),
            "fields": {str(key): _shape(item, depth + 1, max_depth) for key, item in value.items()},
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "array",
            "length": len(value),
            "element_types": sorted({type(item).__name__ for item in value}),
        }
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "unsupported"


class StructuredPerceptionAdapter:
    modality = Modality.STRUCTURED

    def perceive(self, item: PerceptionInput) -> Observation:
        if not isinstance(item.content, Mapping):
            raise TypeError("StructuredPerceptionAdapter requires a mapping")
        shape = _shape(item.content)
        return Observation(
            modality=Modality.STRUCTURED,
            status=PerceptionStatus.OBSERVED,
            summary=f"Structured input received with {len(item.content)} top-level fields",
            source_input_id=item.input_id,
            content_sha256=content_digest(item.content),
            attributes={
                "top_level_keys": tuple(item.content.keys()),
                "top_level_field_count": len(item.content),
                "shape": shape,
                "semantic_interpretation": "not_performed",
            },
        )
