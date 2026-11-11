"""Validated model registry and bounded inference adapter."""
from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("artifact_digest must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ModelManifest:
    model_id: str
    version: str
    artifact_digest: str
    tokenizer_version: str
    modalities: frozenset[str]
    max_context_tokens: int
    max_output_tokens: int
    precision: str
    cache_format: str

    def __post_init__(self):
        for name in ("model_id", "version", "tokenizer_version", "precision", "cache_format"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        _sha(self.artifact_digest)
        if not self.modalities or any(not x.strip() for x in self.modalities):
            raise ValueError("modalities must be non-empty")
        if type(self.max_context_tokens) is not int or self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive integer")
        if type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= self.max_context_tokens:
            raise ValueError("max_output_tokens must be positive and within context limit")


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    model_id: str
    input: Any
    modality: str = "text"
    max_output_tokens: int = 256
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if not self.model_id.strip() or not self.modality.strip():
            raise ValueError("model_id and modality are required")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive integer")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    request_id: str
    model_id: str
    output: Any
    latency_seconds: float
    status: str = "succeeded"


class ModelRegistry:
    def __init__(self):
        self._models = {}
        self._lock = threading.RLock()

    def register(self, manifest: ModelManifest, runner: Callable[[Any, int], Any]):
        if not isinstance(manifest, ModelManifest) or not callable(runner):
            raise TypeError("manifest and callable runner are required")
        key = (manifest.model_id, manifest.version)
        with self._lock:
            prior = self._models.get(key)
            if prior is not None and prior[0] != manifest:
                raise ValueError("model version is immutable")
            self._models[key] = (manifest, runner)

    def resolve(self, model_id: str, version: str | None = None):
        with self._lock:
            candidates = [(k, v) for k, v in self._models.items()
                          if k[0] == model_id and (version is None or k[1] == version)]
            if not candidates:
                raise KeyError(model_id)
            candidates.sort(key=lambda pair: pair[0][1])
            return candidates[-1][1]


class InferenceEngine:
    """Calls only explicitly registered model runners; no implicit model fallback."""

    def __init__(self, registry: ModelRegistry):
        if not isinstance(registry, ModelRegistry):
            raise TypeError("registry must be ModelRegistry")
        self.registry = registry

    def generate(self, request: InferenceRequest, *, version: str | None = None):
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be InferenceRequest")
        manifest, runner = self.registry.resolve(request.model_id, version)
        if request.modality not in manifest.modalities:
            raise ValueError("requested modality is not declared by model manifest")
        if request.max_output_tokens > manifest.max_output_tokens:
            raise ValueError("requested output exceeds model limit")
        if isinstance(request.input, Mapping) and "token_count" in request.input:
            count = request.input["token_count"]
            if type(count) is not int or count < 0 or count > manifest.max_context_tokens:
                raise ValueError("input token_count exceeds model context contract")
        started = time.monotonic()
        output = runner(request.input, request.max_output_tokens)
        latency = time.monotonic() - started
        if not math.isfinite(latency):
            raise RuntimeError("invalid inference latency")
        return InferenceResult(request.request_id, manifest.model_id, output, latency)


__all__ = [
    "ModelManifest", "InferenceRequest", "InferenceResult",
    "ModelRegistry", "InferenceEngine",
]
