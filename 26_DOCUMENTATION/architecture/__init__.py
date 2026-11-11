"""Machine-readable architecture component catalogue."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ArchitectureComponent:
    component_id: str
    responsibility: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    maturity: str = "reference"
    def __post_init__(self):
        if any(not x.strip() for x in (self.component_id, self.responsibility)): raise ValueError("component identity required")
        if self.maturity not in {"planned", "reference", "experimental", "production"}: raise ValueError("invalid maturity")

class ArchitectureCatalog:
    def __init__(self): self._components = {}
    def add(self, component: ArchitectureComponent):
        if not isinstance(component, ArchitectureComponent): raise TypeError("component must be ArchitectureComponent")
        if component.component_id in self._components: raise ValueError("component already exists")
        self._components[component.component_id] = component
    def dependency_order(self):
        result, visiting, visited = [], set(), set()
        def visit(name):
            if name in visiting: raise ValueError("architecture dependency cycle")
            if name in visited: return
            if name not in self._components: raise KeyError(name)
            visiting.add(name)
            for dep in self._components[name].dependencies: visit(dep)
            visiting.remove(name); visited.add(name); result.append(name)
        for key in sorted(self._components): visit(key)
        return tuple(result)

__all__ = ["ArchitectureComponent", "ArchitectureCatalog"]
