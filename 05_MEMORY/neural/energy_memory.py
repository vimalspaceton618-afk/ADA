"""ADA Energy-Based Global Memory, matching the associative read path in Ada-src.pdf."""
from __future__ import annotations

from typing import NamedTuple

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError("EnergyMemory requires PyTorch; install ADA with the neural extra.") from exc


class EnergyMemoryOutput(NamedTuple):
    output: Tensor
    attention: Tensor
    raw: Tensor
    gate: Tensor


class EnergyMemory(nn.Module):
    """Trainable K-slot associative memory for [batch, sequence, dim] activations.

    Source-aligned equations:
      scores = X @ M.T
      A = softmax(scores / temperature)
      Y_raw = A @ M
      Y_mem = sigmoid(W_mem_gate(X)) * Y_raw

    The bank is trained through gradients. This module does not invent online
    write/erase behaviour absent from the supplied ADA source.
    """

    def __init__(self, dim: int, *, slots: int = 512,
                 temperature: float = 1.0, init_std: float = 0.02,
                 bias: bool = True) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if type(slots) is not int or slots <= 0:
            raise ValueError("slots must be a positive integer")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a real number")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if isinstance(init_std, bool) or not isinstance(init_std, (int, float)):
            raise TypeError("init_std must be a real number")
        if init_std <= 0:
            raise ValueError("init_std must be greater than zero")
        self.dim = dim
        self.slots = slots
        self.temperature = float(temperature)
        self.energy_mem = nn.Parameter(torch.empty(slots, dim))
        nn.init.normal_(self.energy_mem, mean=0.0, std=float(init_std))
        self.W_mem_gate = nn.Linear(dim, dim, bias=bias)

    @property
    def memory_bank(self) -> nn.Parameter:
        return self.energy_mem

    @property
    def gate_projection(self) -> nn.Linear:
        return self.W_mem_gate

    def forward(self, x: Tensor, *, return_aux: bool = False):
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, sequence, dim]")
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}, got {x.shape[-1]}")
        if not x.is_floating_point():
            raise TypeError("x must use a floating-point dtype")
        if x.device != self.energy_mem.device:
            raise ValueError("x and EnergyMemory parameters must be on the same device")
        scores = torch.matmul(x, self.energy_mem.transpose(0, 1))
        attention = F.softmax(scores / self.temperature, dim=-1)
        raw = torch.matmul(attention, self.energy_mem)
        gate = torch.sigmoid(self.W_mem_gate(x))
        output = gate * raw
        return EnergyMemoryOutput(output, attention, raw, gate) if return_aux else output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, slots={self.slots}, temperature={self.temperature:g}"
