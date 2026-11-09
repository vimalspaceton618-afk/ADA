"""Pre-norm ADA decoder block with alternating sequence mixers.

The block connects the verified Energy Memory + sparse MoE FFN to a causal
sequence-mixing path. Mamba2SSD follows the exact convolution/gating source
snippet in Ada-src.pdf; despite its source name, that snippet does not include
the selective state-space recurrence/scan, so this implementation does not
claim a complete Mamba-2 SSD kernel.

TensorNetAttention implements the source's low-rank Q/K projection intent as
causal multi-head attention. The PDF's displayed TensorNet forward method is
only an output projection stub, so this executable attention is an explicit
completion of that incomplete source sketch, not a verbatim reproduction.
"""

from __future__ import annotations

import math

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from ada.neural import MoEEnergyMemoryFFN
    from .mamba2_reference import Mamba2ReferenceMixer
except ImportError as exc:  # pragma: no cover - optional neural dependency
    raise ImportError(
        "ADA decoder blocks require PyTorch; install ADA with the neural extra."
    ) from exc


class RMSNorm(nn.Module):
    """RMS normalization with a learned scale and no mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError("eps must be a positive real number")
        if not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError("eps must be finite and greater than zero")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight


class Mamba2SSD(nn.Module):
    """PDF-faithful gated depthwise-convolution mixer (not a full SSD scan).

    The supplied PDF projects into a content and residual branch, applies a
    causal depthwise Conv1d, gates with SiLU, and projects back to model width.
    """

    def __init__(
        self,
        dim: int,
        *,
        expand: int = 2,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if type(expand) is not int or expand <= 0:
            raise ValueError("expand must be a positive integer")
        if type(kernel_size) is not int or kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")

        self.dim = dim
        self.d_inner = dim * expand
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=kernel_size,
            groups=self.d_inner,
            padding=0,
        )
        self.out_proj = nn.Linear(self.d_inner, dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [batch, sequence, {self.dim}]")
        if x.shape[1] == 0:
            raise ValueError("sequence dimension must be non-empty")

        length = x.shape[1]
        content, residual = self.in_proj(x).chunk(2, dim=-1)
        content = F.pad(
            content.transpose(1, 2),
            (self.kernel_size - 1, 0),
        )
        content = self.conv1d(content).transpose(1, 2)
        return self.out_proj(F.silu(content) * F.silu(residual))


class TensorNetAttention(nn.Module):
    """Causal multi-head attention with low-rank query/key projections."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        rank: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if type(n_heads) is not int or n_heads <= 0:
            raise ValueError("n_heads must be a positive integer")
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a real number")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.dim = dim
        self.n_heads = n_heads
        self.rank = rank
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, n_heads * rank, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * rank, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: Tensor, *, attention_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [batch, sequence, {self.dim}]")
        batch, length, _ = x.shape
        if length == 0:
            raise ValueError("sequence dimension must be non-empty")

        q = self.q_proj(x).view(batch, length, self.n_heads, self.rank)
        k = self.k_proj(x).view(batch, length, self.n_heads, self.rank)
        v = self.v_proj(x).view(batch, length, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.rank)
        causal_mask = torch.ones(
            length, length, dtype=torch.bool, device=x.device
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must be boolean")
            key_mask = ~attention_mask[:, None, None, :]
            scores = scores.masked_fill(key_mask, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.dim)
        output = self.o_proj(attended)

        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).to(output.dtype)
        return output


class ADADecoderBlock(nn.Module):
    """Pre-norm mixer + integrated MoE/Energy Memory FFN decoder block.

    Even layer indices use the PDF's Mamba2SSD convolution/gating mixer;
    odd indices use causal low-rank TensorNetAttention. Both residual paths
    are owned by this block: the FFN already returns its own residual output.
    """

    def __init__(
        self,
        dim: int,
        layer_idx: int,
        *,
        n_heads: int = 2,
        attention_rank: int = 64,
        mixer_expand: int = 2,
        mixer_kernel_size: int = 4,
        num_experts: int = 8,
        num_slots: int = 512,
        expansion_factor: int = 4,
        temperature: float = 0.1,
        dropout: float = 0.0,
        mixer_backend: str = "pdf_conv",
        mamba2_d_state: int = 64,
        mamba2_headdim: int = 64,
        mamba2_ngroups: int = 1,
        mamba2_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if type(layer_idx) is not int or layer_idx < 0:
            raise ValueError("layer_idx must be a non-negative integer")
        if mixer_backend not in {"pdf_conv", "mamba2_reference"}:
            raise ValueError("mixer_backend must be 'pdf_conv' or 'mamba2_reference'")

        self.dim = dim
        self.layer_idx = layer_idx
        if layer_idx % 2 == 0 and mixer_backend == "mamba2_reference":
            self.mixer_type = "mamba2_ssd_reference"
            self.mixer = Mamba2ReferenceMixer(
                dim, d_state=mamba2_d_state, d_conv=mixer_kernel_size,
                expand=mixer_expand, headdim=mamba2_headdim,
                ngroups=mamba2_ngroups, chunk_size=mamba2_chunk_size,
            )
        elif layer_idx % 2 == 0:
            self.mixer_type = "mamba2_conv"
            self.mixer = Mamba2SSD(
                dim, expand=mixer_expand, kernel_size=mixer_kernel_size
            )
        else:
            self.mixer_type = "tensornet_causal_attention"
            self.mixer = TensorNetAttention(
                dim, n_heads, rank=attention_rank, dropout=dropout
            )
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = MoEEnergyMemoryFFN(
            d_model=dim,
            num_experts=num_experts,
            num_slots=num_slots,
            expansion_factor=expansion_factor,
            temperature=temperature,
        )

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [batch, sequence, {self.dim}]")
        if x.shape[1] == 0:
            raise ValueError("sequence dimension must be non-empty")

        if isinstance(self.mixer, TensorNetAttention):
            mixed = self.mixer(self.norm1(x), attention_mask=attention_mask)
        else:
            mixed = self.mixer(self.norm1(x))
            if attention_mask is not None:
                if attention_mask.shape != x.shape[:2] or attention_mask.dtype != torch.bool:
                    raise ValueError("attention_mask must be boolean [batch, sequence]")
                mixed = mixed * attention_mask.unsqueeze(-1).to(mixed.dtype)

        after_mixer = x + mixed
        hidden_states, aux = self.ffn(
            self.norm2(after_mixer), residual=after_mixer
        )
        return hidden_states, aux
