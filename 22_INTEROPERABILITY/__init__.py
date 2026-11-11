"""Versioned plugin manifests and strict JSON contract serialization."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PluginManifest:
    plugin_id: str
    version: str
    api_version: str
    capabilities: frozenset[str]
    input_schema: Mapping[str, str]
    output_schema: Mapping[str, str]

    def __post_init__(self):
        for name in ("plugin_id", "version", "api_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        if not self.capabilities:
            raise ValueError("plugin must declare capabilities")
        supported = {"str", "int", "float", "bool", "object", "array"}
        for schema in (self.input_schema, self.output_schema):
            if any(not isinstance(k, str) or v not in supported for k, v in schema.items()):
                raise ValueError("unsupported schema field or type")


class PluginRegistry:
    """Only explicitly allowlisted plugins can be registered."""

    def __init__(self, allowlist: frozenset[str], *, api_version: str = "1"):
        self.allowlist = frozenset(allowlist)
        self.api_version = api_version
        self._items = {}
        self._lock = threading.RLock()

    def register(self, manifest: PluginManifest):
        if not isinstance(manifest, PluginManifest):
            raise TypeError("manifest must be PluginManifest")
        if manifest.plugin_id not in self.allowlist:
            raise PermissionError("plugin is not in the explicit allowlist")
        if manifest.api_version != self.api_version:
            raise ValueError("plugin API version is incompatible")
        key = (manifest.plugin_id, manifest.version)
        with self._lock:
            previous = self._items.get(key)
            if previous is not None and previous != manifest:
                raise ValueError("plugin version is immutable")
            self._items[key] = manifest
        return manifest

    def resolve(self, plugin_id: str, version: str):
        with self._lock:
            return self._items[(plugin_id, version)]


class ContractSerializer:
    """Canonical JSON codec with exact field and primitive-type validation."""

    _types = {
        "str": str, "int": int, "float": (int, float),
        "bool": bool, "object": Mapping, "array": (list, tuple),
    }

    @classmethod
    def validate(cls, payload: Mapping[str, Any], schema: Mapping[str, str]):
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        missing = set(schema) - set(payload)
        unknown = set(payload) - set(schema)
        if missing or unknown:
            raise ValueError(f"schema mismatch missing={sorted(missing)} unknown={sorted(unknown)}")
        for field, kind in schema.items():
            expected = cls._types[kind]
            value = payload[field]
            if kind == "int" and isinstance(value, bool):
                raise TypeError(f"{field} must be int, not bool")
            if not isinstance(value, expected):
                raise TypeError(f"{field} must match schema type {kind}")
        return True

    @classmethod
    def encode(cls, payload: Mapping[str, Any], schema: Mapping[str, str],
               *, max_bytes: int = 1_000_000) -> bytes:
        cls.validate(payload, schema)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(raw) > max_bytes:
            raise ValueError("serialized payload exceeds size limit")
        return raw

    @classmethod
    def decode(cls, raw: bytes, schema: Mapping[str, str],
               *, max_bytes: int = 1_000_000):
        if not isinstance(raw, bytes) or len(raw) > max_bytes:
            raise ValueError("invalid or oversized payload")
        payload = json.loads(raw.decode("utf-8"))
        cls.validate(payload, schema)
        return payload


__all__ = ["PluginManifest", "PluginRegistry", "ContractSerializer"]
