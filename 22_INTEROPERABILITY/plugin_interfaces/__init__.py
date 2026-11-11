"""Plugin callable interface enforced by an allowlisted registry."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from .. import PluginManifest, PluginRegistry

class Plugin(Protocol):
    def invoke(self, operation: str, arguments: Mapping[str, Any]) -> Any: ...

@dataclass(frozen=True, slots=True)
class PluginBinding:
    manifest: PluginManifest
    implementation: Plugin

class PluginHost:
    def __init__(self, registry: PluginRegistry):
        if not isinstance(registry, PluginRegistry): raise TypeError("registry must be PluginRegistry")
        self.registry, self._bindings = registry, {}
    def bind(self, manifest: PluginManifest, implementation: Plugin):
        self.registry.resolve(manifest.plugin_id, manifest.version)
        if not callable(getattr(implementation, "invoke", None)): raise TypeError("plugin must implement invoke")
        self._bindings[(manifest.plugin_id, manifest.version)] = PluginBinding(manifest, implementation)
    def invoke(self, plugin_id: str, version: str, operation: str, arguments: Mapping[str, Any]):
        binding = self._bindings[(plugin_id, version)]
        if operation not in binding.manifest.capabilities: raise PermissionError("plugin capability not declared")
        return binding.implementation.invoke(operation, arguments)

__all__ = ["Plugin", "PluginBinding", "PluginHost"]
