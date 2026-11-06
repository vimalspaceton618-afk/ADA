"""Validated dispatch from PerceptionResult into canonical representations."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from .adapters import RepresentationAdapter, default_adapters
from .contracts import (
    Representation,
    RepresentationResult,
    RepresentationResultStatus,
)


class RepresentationEngine:
    """Convert perception observations into versioned, typed projections.

    The engine preserves the source digest and IDs, rejects mismatched adapter
    outputs, and does not manufacture representations for unsupported inputs.
    """

    def __init__(self, *, adapters: Iterable[RepresentationAdapter] | None = None) -> None:
        self._lock = threading.RLock()
        self._adapters: dict[str, RepresentationAdapter] = {}
        for adapter in (tuple(adapters) if adapters is not None else default_adapters()):
            self.register_adapter(adapter)

    def register_adapter(
        self,
        adapter: RepresentationAdapter,
        *,
        replace: bool = False,
    ) -> None:
        modality = getattr(adapter, "modality_value", None)
        if not isinstance(modality, str) or not modality.strip():
            raise TypeError("adapter.modality_value must be a non-empty modality string")
        if not callable(getattr(adapter, "represent", None)):
            raise TypeError("adapter must provide represent(PerceptionResult)")
        with self._lock:
            if modality in self._adapters and not replace:
                raise ValueError(f"representation adapter already registered: {modality}")
            self._adapters[modality] = adapter

    def unregister_adapter(self, modality: str) -> bool:
        if not isinstance(modality, str) or not modality.strip():
            raise ValueError("modality must be a non-empty string")
        with self._lock:
            return self._adapters.pop(modality, None) is not None

    @staticmethod
    def _empty_result(result, status, *, errors=(), warnings=()):
        return RepresentationResult(
            input_id=result.input_id,
            correlation_id=result.correlation_id,
            modality=result.modality.value,
            source_content_sha256=result.content_sha256,
            status=status,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def transform(self, result) -> RepresentationResult:
        """Transform one validated perception result.

        This method cannot re-check raw input bytes because PerceptionResult
        intentionally carries only the source digest. It verifies every
        available result/observation/representation provenance link.
        """
        from importlib import import_module

        perception = import_module("03_PERCEPTION.contracts")
        if not isinstance(result, perception.PerceptionResult):
            raise TypeError("transform requires a 03_PERCEPTION.PerceptionResult")

        source_status = getattr(result.status, "value", None)
        if source_status not in {"observed", "partial"}:
            mapping = {
                "unsupported": RepresentationResultStatus.UNSUPPORTED,
                "rejected": RepresentationResultStatus.REJECTED,
                "failed": RepresentationResultStatus.FAILED,
            }
            status = mapping.get(source_status, RepresentationResultStatus.FAILED)
            return self._empty_result(
                result,
                status,
                errors=result.errors,
                warnings=result.warnings,
            )

        observation = result.observation
        if observation is None:
            return self._empty_result(
                result,
                RepresentationResultStatus.FAILED,
                errors=("perception result has no observation",),
                warnings=result.warnings,
            )

        if observation.source_input_id != result.input_id:
            return self._empty_result(
                result,
                RepresentationResultStatus.FAILED,
                errors=("observation source_input_id does not match perception result",),
            )
        if observation.modality is not result.modality:
            return self._empty_result(
                result,
                RepresentationResultStatus.FAILED,
                errors=("observation modality does not match perception result",),
            )
        if getattr(observation.status, "value", None) != source_status:
            return self._empty_result(
                result,
                RepresentationResultStatus.FAILED,
                errors=("observation status does not match perception result",),
            )
        if observation.content_sha256 != result.content_sha256:
            return self._empty_result(
                result,
                RepresentationResultStatus.FAILED,
                errors=("observation digest does not match perception result",),
            )

        with self._lock:
            adapter = self._adapters.get(result.modality.value)
        if adapter is None:
            return self._empty_result(
                result,
                RepresentationResultStatus.UNSUPPORTED,
                warnings=(f"No representation adapter registered for {result.modality.value}",),
            )

        try:
            representations = tuple(adapter.represent(result))
            if not representations:
                raise ValueError("representation adapter returned no representations")
            if any(not isinstance(item, Representation) for item in representations):
                raise TypeError("representation adapter must return Representation values")
            seen_ids: set[str] = set()
            for item in representations:
                if item.source_input_id != result.input_id:
                    raise ValueError("representation provenance does not match perception input")
                if item.source_content_sha256 != result.content_sha256:
                    raise ValueError("representation source digest does not match perception result")
                if item.modality != result.modality.value:
                    raise ValueError("representation modality does not match perception result")
                if item.representation_id in seen_ids:
                    raise ValueError("representation adapter returned duplicate representation IDs")
                seen_ids.add(item.representation_id)

            status = (
                RepresentationResultStatus.PARTIAL
                if source_status == "partial"
                else RepresentationResultStatus.COMPLETE
            )
            return RepresentationResult(
                input_id=result.input_id,
                correlation_id=result.correlation_id,
                modality=result.modality.value,
                source_content_sha256=result.content_sha256,
                status=status,
                representations=representations,
                warnings=tuple(result.warnings) + tuple(observation.warnings),
            )
        except Exception as exc:
            return self._empty_result(
                result,
                RepresentationResultStatus.FAILED,
                errors=(f"{type(exc).__name__}: {exc}",),
                warnings=tuple(result.warnings) + tuple(observation.warnings),
            )

    def represent(self, result) -> RepresentationResult:
        """Alias for transform() to support pipeline-style call sites."""
        return self.transform(result)
