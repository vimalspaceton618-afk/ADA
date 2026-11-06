"""Sparse top-1 MoE FFN fused with ADA's Hopfield Energy Memory.

This is the first executable expert-system block in 09_NEURAL_SUBSTRATE.
The implementation follows the detailed MoEEnergyMemoryFFN formulation in
Ada-src.pdf (SwiGLU experts, noisy top-1 routing, straight-through router
gradient, load-balance auxiliary loss, Energy Memory branch and learned blend).

The returned hidden state includes the residual connection, matching the PDF's
forward contract. Callers must NOT add a second residual around this module.
"""

from __future__ import annotations

import math
from importlib import import_module

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    EnergyMemory = import_module("05_MEMORY.neural.energy_memory").EnergyMemory
except ImportError as exc:  # pragma: no cover - optional neural dependency
    raise ImportError(
        "MoEEnergyMemoryFFN requires PyTorch; install ADA with the neural extra."
    ) from exc


class ExactGLUExpert(nn.Module):
    """Bias-free SwiGLU expert: down(SiLU(gate(x)) * up(x))."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoEEnergyMemoryFFN(nn.Module):
    """Noisy top-1 sparse MoE with a parallel Energy Memory branch.

    Input:
        x: [batch, sequence, d_model]

    Returns:
        (hidden_states, aux)
        hidden_states is [B, L, D] and includes the residual x.
        aux exposes router probabilities/indices, expert load statistics,
        load-balancing loss, blend coefficients, and memory diagnostics.

    Only the selected expert is evaluated per token. The straight-through
    selector has a hard one-hot forward value and a soft router gradient.
    The caller's training objective should add aux["load_balancing_loss"]
    with an appropriate coefficient.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_experts: int = 8,
        num_slots: int = 512,
        expansion_factor: int = 4,
        temperature: float = 0.1,
        router_noise: bool = True,
    ) -> None:
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if type(num_experts) is not int or num_experts <= 0:
            raise ValueError("num_experts must be a positive integer")
        if type(num_slots) is not int or num_slots <= 0:
            raise ValueError("num_slots must be a positive integer")
        if type(expansion_factor) is not int or expansion_factor <= 0:
            raise ValueError("expansion_factor must be a positive integer")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a real number")
        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        if type(router_noise) is not bool:
            raise TypeError("router_noise must be a bool")

        self.d_model = d_model
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.d_ff = d_model * expansion_factor
        self.router_noise = router_noise

        self.W_r = nn.Linear(d_model, num_experts, bias=False)
        self.W_noise = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExactGLUExpert(d_model, self.d_ff) for _ in range(num_experts)]
        )

        # Match the detailed PDF's stable Hopfield initialization:
        # M ~ N(0, 1/sqrt(D)); gate starts at sigmoid(0) = 0.5.
        self.energy_memory = EnergyMemory(
            dim=d_model,
            slots=num_slots,
            temperature=float(temperature),
            init_std=1.0 / math.sqrt(d_model),
            bias=False,
        )
        nn.init.zeros_(self.energy_memory.W_mem_gate.weight)

        # Per-token scalar blend between MoE and memory branches.
        self.W_blend = nn.Linear(d_model, 1, bias=False)

        # Detached snapshots only: never retain a previous autograd graph.
        self.last_aux: dict[str, Tensor] | None = None
        self.current_load_balancing_loss: Tensor | None = None

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, sequence, d_model]")
        batch, sequence, width = x.shape
        if batch <= 0 or sequence <= 0:
            raise ValueError("batch and sequence dimensions must be non-empty")
        if width != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {width}")
        if not x.is_floating_point():
            raise TypeError("x must use a floating-point dtype")
        if x.device != self.W_r.weight.device:
            raise ValueError("x and MoEEnergyMemoryFFN parameters must share a device")

        flat = x.reshape(-1, width)
        logits_det = self.W_r(flat)

        if self.training and self.router_noise:
            noise_scale = F.softplus(self.W_noise(flat))
            logits = logits_det + torch.randn_like(logits_det) * noise_scale
        else:
            logits = logits_det

        probabilities = F.softmax(logits, dim=-1)
        expert_indices = torch.argmax(logits, dim=-1)
        hard_onehot = F.one_hot(
            expert_indices, num_classes=self.num_experts
        ).to(dtype=flat.dtype)

        # Switch-style load-balancing regularizer:
        # E * sum_i(f_i * P_i), where f is hard dispatch fraction and P
        # is mean soft router probability. It encourages, but cannot
        # guarantee, balanced routing.
        expert_fraction = hard_onehot.detach().float().mean(dim=0)
        mean_router_probability = probabilities.float().mean(dim=0)
        if self.training:
            load_balancing_loss = self.num_experts * torch.sum(
                expert_fraction * mean_router_probability
            )
        else:
            load_balancing_loss = probabilities.new_zeros(())

        # Hard forward routing; gradients flow through probabilities.
        straight_through = hard_onehot + probabilities - probabilities.detach()
        moe_flat = torch.zeros_like(flat)

        for expert_id, expert in enumerate(self.experts):
            token_indices = torch.nonzero(
                expert_indices.eq(expert_id), as_tuple=True
            )[0]
            if token_indices.numel() == 0:
                continue

            token_inputs = flat.index_select(0, token_indices)
            expert_output = expert(token_inputs)
            selector = straight_through.index_select(
                0, token_indices
            )[:, expert_id].unsqueeze(-1)
            dispatched = selector * expert_output
            moe_flat = moe_flat.index_copy(0, token_indices, dispatched)

        moe_output = moe_flat.reshape(batch, sequence, width)

        memory_result = self.energy_memory(x, return_aux=True)
        alpha = torch.sigmoid(self.W_blend(x))
        blended = alpha * moe_output + (1.0 - alpha) * memory_result.output

        # The PDF's FFN contract includes its own residual. Do not add one
        # again in a parent block.
        hidden_states = x + blended

        aux: dict[str, Tensor] = {
            "router_probs": probabilities.reshape(batch, sequence, self.num_experts),
            "expert_indices": expert_indices.reshape(batch, sequence),
            "expert_fraction": expert_fraction,
            "mean_router_probability": mean_router_probability,
            "load_balancing_loss": load_balancing_loss,
            "blend_alpha": alpha,
            "moe_output": moe_output,
            "memory_output": memory_result.output,
            "memory_attention": memory_result.attention,
            "memory_raw": memory_result.raw,
            "memory_gate": memory_result.gate,
        }

        self.current_load_balancing_loss = load_balancing_loss.detach()
        self.last_aux = {key: value.detach() for key, value in aux.items()}
        return hidden_states, aux

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_experts={self.num_experts}, "
            f"num_slots={self.num_slots}, d_ff={self.d_ff}, "
            f"router_noise={self.router_noise}"
        )
