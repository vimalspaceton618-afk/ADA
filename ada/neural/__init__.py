"""Stable, lazy exports for ADA's optional PyTorch neural components."""

from importlib import import_module

_SSD_EXPORTS = {"segment_sum", "ssd_minimal_discrete"}

_MEMORY_EXPORTS = {"EnergyMemory", "EnergyMemoryOutput"}
_EXPERT_EXPORTS = {"ExactGLUExpert", "MoEEnergyMemoryFFN"}
_DECODER_EXPORTS = {"ADADecoderBlock", "Mamba2SSD", "RMSNorm", "TensorNetAttention"}
_MODEL_EXPORTS = {"ADAConfig", "ADAForCausalLM", "ADAOutput"}


def __getattr__(name: str):
    if name in _MEMORY_EXPORTS:
        module = import_module("05_MEMORY.neural.energy_memory")
    elif name in _EXPERT_EXPORTS:
        module = import_module("09_NEURAL_SUBSTRATE.expert_systems.moe_energy_memory")
    elif name in _DECODER_EXPORTS:
        module = import_module("09_NEURAL_SUBSTRATE.neural_core.decoder_block")
    elif name in _MODEL_EXPORTS:
        module = import_module("09_NEURAL_SUBSTRATE.neural_core.modeling_ada")
    elif name in _SSD_EXPORTS:
        module = import_module("09_NEURAL_SUBSTRATE.neural_core.ssd_reference")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | _MEMORY_EXPORTS
        | _EXPERT_EXPORTS
        | _DECODER_EXPORTS
        | _MODEL_EXPORTS
        | _SSD_EXPORTS
    )


__all__ = sorted(
    _MEMORY_EXPORTS | _EXPERT_EXPORTS | _DECODER_EXPORTS | _MODEL_EXPORTS | _SSD_EXPORTS
)
