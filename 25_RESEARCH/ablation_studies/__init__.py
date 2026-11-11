"""Factorial ablation plan manifests for ADA v4.5 components."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product

COMPONENTS = ("mod", "dsa_gqa", "fg_moe", "external_memory", "mtp")

@dataclass(frozen=True, slots=True)
class AblationConfig:
    experiment_id: str
    enabled: frozenset[str]
    seed: int
    def __post_init__(self):
        if not self.experiment_id.strip() or not self.enabled.issubset(COMPONENTS):
            raise ValueError("invalid ablation configuration")
        if type(self.seed) is not int: raise TypeError("seed must be int")

def full_factorial(seed: int):
    return tuple(AblationConfig(f"factorial-{i:02d}", frozenset(c for c, on in zip(COMPONENTS, flags) if on), seed)
                 for i, flags in enumerate(product((False, True), repeat=len(COMPONENTS))))

__all__ = ["COMPONENTS", "AblationConfig", "full_factorial"]
