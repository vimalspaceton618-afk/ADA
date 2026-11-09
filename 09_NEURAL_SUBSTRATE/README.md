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


## Hybrid decoder block

Canonical module: `neural_core/decoder_block.py`

`ADADecoderBlock` connects the verified MoE/Energy Memory FFN to a pre-norm sequence mixer. Even-numbered layers use the PDF's gated depthwise-convolution `Mamba2SSD` code path; odd-numbered layers use causal low-rank multi-head attention.

```python
from ada.neural import ADADecoderBlock

block = ADADecoderBlock(
    dim=768, layer_idx=0, n_heads=2,
    num_experts=8, num_slots=512,
)
hidden_states, aux = block(hidden_states)
loss = task_loss + 0.01 * aux["load_balancing_loss"]
```

The FFN supports an explicit residual stream so the block can normalize its FFN input while preserving the unnormalized residual. The block owns both residual additions; callers must not add another residual around it.

### Source fidelity and limitations

- The PDF's displayed `Mamba2SSD` implementation is a gated depthwise convolution. It does not implement the selective state-space recurrence, discretization, or associative scan. The repository preserves that source path but does not claim a production Mamba-2 SSD implementation.
- The PDF's displayed `TensorNetAttention.forward` is an output-projection stub. ADA implements causal low-rank Q/K multi-head attention to make the alternating attention branch executable. This is an explicit completion of an incomplete source sketch, not a verbatim reproduction.
- This milestone implements a decoder **block**, not yet a full token embedding/decoder stack/language-model head or training runtime.


## Full causal language-model stack

Canonical module: `neural_core/modeling_ada.py`

`ADAForCausalLM` assembles token and learned position embeddings, alternating decoder blocks, final RMSNorm, and a tied-by-default or untied vocabulary projection.

```python
from ada.neural import ADAConfig, ADAForCausalLM

config = ADAConfig(
    vocab_size=32000, dim=768, n_layers=12, n_heads=12,
    attention_rank=64, max_seq_len=2048,
)
model = ADAForCausalLM(config)
result = model(input_ids, labels=input_ids)
loss = result.loss + 0.01 * result.auxiliary_loss
loss.backward()
```

- Labels use next-token shifting internally; label position zero is not included in the loss.
- The returned `auxiliary_loss` is the mean layer load-balancing loss. Add it to the task loss explicitly.
- `attention_mask` is a boolean [batch, sequence] mask. If omitted and `pad_token_id` is configured, it is inferred from input IDs.
- `save_pretrained` / `from_pretrained` save the JSON config and PyTorch state dict.
- This is a runnable research model, not a production training/inference runtime. It currently uses learned absolute position embeddings and the decoder mixers described above; no claim of RoPE, full Mamba selective scan, fused kernels, KV cache, distributed training, or JAX/TPU execution is made.


## Chunked SSD reference core

Canonical module: `neural_core/ssd_reference.py`

The public `ada.neural.ssd_minimal_discrete` function implements the readable
four-stage discrete SSD factorization: intra-chunk output, chunk-state
summaries, inter-chunk state propagation, and state-to-output conversion.
It supports grouped B/C inputs when the head count is divisible by the group
count, optional initial states, and returns both sequence outputs and the
final recurrent state.

The function accepts **already discretized** `X` and `A` tensors. It does
not perform Mamba-2's input projection, causal convolution, timestep
parameterization, gated normalization, or output projection. It is a
correctness/reference core, not a fused scan or complete Mamba-2 mixer.
Its structure follows the official reference Listing 1:
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py

The reference tests compare chunked outputs and final states against a
token-by-token recurrence, check future-token causality, and verify gradients.
They run in the CPU-PyTorch CI job.
