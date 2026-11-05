"""Modality-aware perception dispatch with provenance and fail-closed behavior."""

from __future__ import annotations

import threading
from collections.abc import Protocol

from .contracts import (
    Modality, Observation, PerceptionInput, PerceptionResult, PerceptionStatus,
    content_digest,
)
from .input_interface.normalizer import InputNormalizer
from .structured_perception.processor import StructuredPerceptionAdapter
from .text_perception.processor import TextPerceptionAdapter


class PerceptionAdapter(Protocol):
    modality: Modality

    def perceive(self, item: PerceptionInput) -> Observation: ...


class PerceptionEngine:
    """Dispatches normalized inputs to explicitly registered modality adapters.

    Text and JSON-compatible structured inspection are built in. SigLIP image,
    Whisper audio, and SigLIP-backed sparse video adapters are optional and must
    be explicitly registered. No synthetic captions, transcripts, embeddings,
    temporal reasoning, or classifications are generated.
    """

    def __init__(self, *, normalizer: InputNormalizer | None = None) -> None:
        self.normalizer = normalizer or InputNormalizer()
        self._lock = threading.RLock()
        self._adapters: dict[Modality, PerceptionAdapter] = {}
        self.register_adapter(TextPerceptionAdapter())
        self.register_adapter(StructuredPerceptionAdapter())

    def register_adapter(self, adapter: PerceptionAdapter, *, replace: bool = False) -> None:
        modality = getattr(adapter, "modality", None)
        if not isinstance(modality, Modality):
            raise TypeError("adapter.modality must be a Modality")
        if not callable(getattr(adapter, "perceive", None)):
            raise TypeError("adapter must provide a callable perceive(input) method")
        with self._lock:
            if modality in self._adapters and not replace:
                raise ValueError(f"adapter already registered for modality: {modality.value}")
            self._adapters[modality] = adapter

    def unregister_adapter(self, modality: Modality) -> bool:
        if not isinstance(modality, Modality):
            raise TypeError("modality must be a Modality")
        with self._lock:
            return self._adapters.pop(modality, None) is not None

    def perceive(self, item: PerceptionInput) -> PerceptionResult:
        if not isinstance(item, PerceptionInput):
            raise TypeError("item must be PerceptionInput")
        digest = content_digest(item.content)
        try:
            normalized = self.normalizer.normalize(item)
        except Exception as exc:
            return PerceptionResult(
                input_id=item.input_id,
                correlation_id=item.correlation_id,
                modality=item.modality,
                status=PerceptionStatus.REJECTED,
                observation=None,
                content_sha256=digest,
                errors=(f"{type(exc).__name__}: {exc}",),
            )

        with self._lock:
            adapter = self._adapters.get(normalized.modality)
        if adapter is None:
            return PerceptionResult(
                input_id=item.input_id,
                correlation_id=item.correlation_id,
                modality=item.modality,
                status=PerceptionStatus.UNSUPPORTED,
                observation=None,
                content_sha256=digest,
                warnings=(f"No perception adapter registered for {item.modality.value}",),
            )

        try:
            observation = adapter.perceive(normalized)
            if not isinstance(observation, Observation):
                raise TypeError("perception adapter must return Observation")
            if observation.modality is not item.modality:
                raise ValueError("adapter returned an observation for the wrong modality")
            if observation.source_input_id != item.input_id:
                raise ValueError("adapter observation provenance does not match input")
            if observation.content_sha256 != content_digest(normalized.content):
                raise ValueError("adapter observation digest does not match normalized content")
            return PerceptionResult(
                input_id=item.input_id,
                correlation_id=item.correlation_id,
                modality=item.modality,
                status=observation.status,
                observation=observation,
                content_sha256=digest,
                warnings=observation.warnings,
            )
        except Exception as exc:
            return PerceptionResult(
                input_id=item.input_id,
                correlation_id=item.correlation_id,
                modality=item.modality,
                status=PerceptionStatus.FAILED,
                observation=None,
                content_sha256=digest,
                errors=(f"{type(exc).__name__}: {exc}",),
            )
