"""Stable, lazy exports for ADA's optional PyTorch neural components."""

from importlib import import_module

_MEMORY_EXPORTS = {"EnergyMemory", "EnergyMemoryOutput"}
_EXPERT_EXPORTS = {"ExactGLUExpert", "MoEEnergyMemoryFFN"}
_DECODER_EXPORTS = {"ADADecoderBlock", "Mamba2SSD", "RMSNorm", "TensorNetAttention"}


def __getattr__(name: str):
    if name in _MEMORY_EXPORTS:
        module = import_module("05_MEMORY.neural.energy_memory")
        value = getattr(module, name)
    elif name in _EXPERT_EXPORTS:
        module = import_module(
            "09_NEURAL_SUBSTRATE.expert_systems.moe_energy_memory"
        )
        value = getattr(module, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _MEMORY_EXPORTS | _EXPERT_EXPORTS | _DECODER_EXPORTS)


__all__ = sorted(_MEMORY_EXPORTS | _EXPERT_EXPORTS | _DECODER_EXPORTS)
