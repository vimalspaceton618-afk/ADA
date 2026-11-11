"""Search-space manifests for the ADA four-block cycle."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product

@dataclass(frozen=True, slots=True)
class ArchitectureCandidate:
    mamba_layers_per_cycle: int
    mod_capacity: float
    sparse_keep: float
    memory_position: str
    def __post_init__(self):
        if not 1 <= self.mamba_layers_per_cycle <= 7: raise ValueError("invalid Mamba cycle depth")
        if not 0 < self.mod_capacity <= 1 or not 0 < self.sparse_keep <= 1: raise ValueError("ratios must be in (0,1]")
        if self.memory_position not in {"before_attention", "after_attention"}: raise ValueError("invalid memory position")

def candidates():
    for m, mod, keep, pos in product((1,2,3,4,5,6,7), (0.25,0.375,0.5,0.625,0.75),
                                      (0.5,0.625,0.75,0.875), ("before_attention","after_attention")):
        yield ArchitectureCandidate(m, mod, keep, pos)

__all__ = ["ArchitectureCandidate", "candidates"]
