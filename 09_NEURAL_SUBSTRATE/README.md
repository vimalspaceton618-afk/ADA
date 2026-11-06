# ADA Neural Substrate

This directory contains executable neural components. The first implemented expert-system block is the PDF-aligned Energy Memory + sparse MoE FFN.

## Integrated execution block

Canonical module: `expert_systems/moe_energy_memory.py`

Public import:

```python
import torch
from ada.neural import MoEEnergyMemoryFFN

block = MoEEnergyMemoryFFN(
    d_model=768,
    num_experts=8,
    num_slots=512,
    expansion_factor=4,
    temperature=0.1,
)
hidden_states, aux = block(torch.randn(2, 128, 768))
```

## Forward path

1. Flatten token representations from `[B,L,D]` to `[B*L,D]`.
2. Produce router logits and optionally add learned, training-only noise.
3. Compute softmax routing probabilities and choose one expert per token.
4. Dispatch tokens sparsely to bias-free SwiGLU experts.
5. Use a straight-through hard selector so the forward path is top-1 while router gradients flow through soft probabilities.
6. Compute `E * sum(f_i * P_i)` as the auxiliary load-balancing loss during training.
7. Retrieve from the 512-slot Hopfield Energy Memory bank in parallel.
8. Learn a per-token scalar blend between expert output and memory output.
9. Add the residual and return hidden states with router/memory diagnostics.

The output already includes its residual input. A parent layer must not add the same residual again. The training loop must add `aux["load_balancing_loss"]` to its task loss with a chosen coefficient; the block does not silently modify the caller's objective.

## Memory initialization

The integrated block uses the detailed Hopfield source configuration: memory vectors initialized with standard deviation `1/sqrt(d_model)`, temperature `0.1` by default, and a bias-free feature gate initialized to zero (initial sigmoid gate = 0.5). `EnergyMemory.energy(x)` exposes the exact scalar energy whose input gradient is `x - attention @ memory_bank`.

## Scope boundary

This implements the model-side FFN/MoE execution component, not a complete ADA language model. The repository does not yet have an executable Mamba/TensorNet sequence mixer, decoder stack, token embedding/language-model head, or end-to-end training runtime. These must be integrated as separate neural-substrate milestones; no placeholder implementation is claimed here.
