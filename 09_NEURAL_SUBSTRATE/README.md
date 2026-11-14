# ADA Neural Substrate

This directory contains executable neural components. The first implemented expert-system block is the PDF-aligned Energy Memory + sparse MoE FFN.

## Official Colab training entry point

The single supported Colab notebook is
`notebooks/colab/ADA_Pretraining.ipynb`. It runs the real-data pretraining
script `scripts/ada_pretrain_fineweb_edu.py`, streams FineWeb-Edu records,
and saves checkpoints and metrics to Google Drive.

The older synthetic GPU validation runner remains available as a developer
diagnostic script, but it is not a training workflow and has no separate
Colab notebook. Do not use it to start pretraining.

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

This implements the model-side FFN/MoE execution component. The repository also contains an executable reference sequence mixer and causal language-model stack described below. These are research components; runnable code does not imply trained weights or a production training runtime.


## Hybrid decoder block

Canonical module: `neural_core/decoder_block.py`

`ADADecoderBlock` connects the verified MoE/Energy Memory FFN to a pre-norm sequence mixer. Odd-numbered layers use causal low-rank multi-head attention. Even-numbered layers use the PDF-faithful gated depthwise-convolution path by default; the full PyTorch Mamba-2 reference path can be selected explicitly with `mixer_backend="mamba2_reference"`.

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

- The PDF's displayed `Mamba2SSD` implementation remains a gated depthwise convolution and is preserved as the default `pdf_conv` backend.
- `Mamba2ReferenceMixer` is a separate, opt-in implementation that assembles input-dependent timestep parameters, causal depthwise convolution, grouped B/C, the chunked SSD recurrence, D skip, gated RMSNorm, and output projection. It is a correctness-oriented PyTorch reference, not a fused or production-optimized Mamba-2 kernel.
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
- This is a runnable research model, not a production training/inference runtime. It currently uses learned absolute position embeddings and the decoder mixers described above. The optional Mamba-2 backend uses a reference SSD scan; no claim of RoPE, fused Triton/CUDA kernels, incremental KV/SSM cache, distributed training, or JAX/TPU execution is made.


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

 
## Full PyTorch Mamba-2 reference mixer

Canonical module: `neural_core/mamba2_reference.py`

The opt-in `Mamba2ReferenceMixer` assembles the stages that were previously
separate in this repository:

1. Project each token into the gate, SSM input, grouped B/C, and per-head
   timestep branches.
2. Apply a causal depthwise convolution and SiLU to the SSM input and B/C
   branches.
3. Convert timestep logits with softplus, combine them with the stable
   negative transition parameter `A = -exp(A_log)`, and discretize the
   sequence inputs/transitions.
4. Run the grouped four-stage SSD reference scan, including non-multiple
   sequence lengths through zero padding.
5. Add the per-head D skip path, apply SiLU-gated RMSNorm, and project back to
   the model dimension.

Select it on `ADAConfig` when the expanded model width is divisible by
`mamba2_headdim`, and the resulting SSM head count is divisible by
`mamba2_ngroups`:

```python
config = ADAConfig(
    vocab_size=32000,
    dim=768,
    n_layers=12,
    n_heads=12,
    mixer_backend="mamba2_reference",
    mamba2_d_state=64,
    mamba2_headdim=64,
    mamba2_ngroups=1,
    mamba2_chunk_size=64,
)
```

This backend is deliberately opt-in. Existing configurations and checkpoints
retain the PDF-faithful `pdf_conv` default. The reference backend accumulates
its SSD scan in FP32 (FP64 for FP64 inputs), supports grouped B/C and optional
learnable initial states, and can return the final recurrent state. It does
not yet provide a one-token decode/update API, fused kernels, or distributed
execution.

Its structure follows the public Mamba-2 reference implementation:
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py
and the SSD factorization:
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py
