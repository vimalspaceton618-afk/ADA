"""Versioned module capability contracts and compatibility validation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True, slots=True)
class ModuleContract:
    module_id: str
    version: str
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]
    capabilities: frozenset[str]
    def __post_init__(self):
        if not self.module_id.strip() or not self.version.strip(): raise ValueError("module ID/version required")
        if not self.inputs or not self.outputs: raise ValueError("input and output schemas must be declared")
        if any(not isinstance(k, str) or not isinstance(v, str) for schema in (self.inputs, self.outputs) for k, v in schema.items()):
            raise TypeError("schema keys and values must be strings")

def compatible(producer: ModuleContract, consumer: ModuleContract):
    if not isinstance(producer, ModuleContract) or not isinstance(consumer, ModuleContract):
        raise TypeError("ModuleContract values required")
    return all(name in producer.outputs and producer.outputs[name] == kind
               for name, kind in consumer.inputs.items())

__all__ = ["ModuleContract", "compatible"]
