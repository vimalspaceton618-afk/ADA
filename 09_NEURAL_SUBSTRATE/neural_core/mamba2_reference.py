"""Correctness-oriented PyTorch Mamba-2 mixer assembled around ADA's SSD core."""
from __future__ import annotations
import math

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Mamba2ReferenceMixer requires PyTorch; install ADA with the neural extra."
    ) from exc

from .ssd_reference import ssd_minimal_discrete


class _GatedRMSNorm(nn.Module):
    """RMSNorm applied after SiLU gating (norm_before_gate=False)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError("eps must be a positive real number")
        if not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError("eps must be finite and greater than zero")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        if x.shape != gate.shape:
            raise ValueError("x and gate must have identical shapes")
        gated = x * F.silu(gate)
        variance = gated.float().square().mean(dim=-1, keepdim=True)
        normalized = gated * torch.rsqrt(variance + self.eps).to(gated.dtype)
        return normalized * self.weight.to(dtype=normalized.dtype)


class Mamba2ReferenceMixer(nn.Module):
    """Full PyTorch Mamba-2-style mixer backed by ADA's chunked SSD reference.

    Input/output layout is [batch, sequence, d_model]. The projection emits
    [z, x, B, C, dt]. The x/B/C branches pass through causal depthwise
    convolution; dt is softplus-parameterized per head. The scan uses
    X = dt*x and A = dt*(-exp(A_log)), followed by the D skip path, gated
    RMSNorm, and output projection.

    Non-multiple sequence lengths are zero-padded to full SSD chunks and
    cropped after scanning. Final state uses FP32 for FP16/BF16/FP32 inputs
    and FP64 for FP64 inputs.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: tuple[float, float] = (0.0, float("inf")),
        A_init_range: tuple[float, float] = (1.0, 16.0),
        learnable_init_states: bool = False,
        bias: bool = False,
        conv_bias: bool = True,
        conv_init: float | None = None,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        positive_ints = {
            "d_model": d_model, "d_state": d_state, "d_conv": d_conv,
            "expand": expand, "headdim": headdim, "ngroups": ngroups,
            "chunk_size": chunk_size,
        }
        for name, value in positive_ints.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(learnable_init_states) is not bool:
            raise TypeError("learnable_init_states must be a bool")
        for name, value in (
            ("dt_min", dt_min), ("dt_max", dt_max),
            ("dt_init_floor", dt_init_floor),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if dt_max < dt_min:
            raise ValueError("dt_max must be greater than or equal to dt_min")
        if (
            not isinstance(dt_limit, tuple) or len(dt_limit) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in dt_limit)
        ):
            raise TypeError("dt_limit must be a pair of real numbers")
        if (
            not math.isfinite(float(dt_limit[0])) or dt_limit[0] < 0
            or dt_limit[1] < dt_limit[0] or math.isnan(float(dt_limit[1]))
        ):
            raise ValueError("dt_limit must satisfy 0 <= lower <= upper")
        if (
            not isinstance(A_init_range, tuple) or len(A_init_range) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in A_init_range)
        ):
            raise TypeError("A_init_range must be a pair of real numbers")
        if (
            not math.isfinite(float(A_init_range[0]))
            or not math.isfinite(float(A_init_range[1]))
            or A_init_range[0] <= 0 or A_init_range[1] < A_init_range[0]
        ):
            raise ValueError("A_init_range must satisfy 0 < lower <= upper")
        if conv_init is not None and (
            isinstance(conv_init, bool) or not isinstance(conv_init, (int, float))
            or not math.isfinite(float(conv_init)) or conv_init <= 0
        ):
            raise ValueError("conv_init must be a positive finite number or None")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.n_heads = self.d_inner // headdim if self.d_inner % headdim == 0 else 0
        self.chunk_size = chunk_size
        self.dt_limit = (float(dt_limit[0]), float(dt_limit[1]))
        if self.n_heads == 0:
            raise ValueError("d_model * expand must be divisible by headdim")
        if self.n_heads % ngroups:
            raise ValueError("ngroups must divide the number of SSM heads")

        projection_width = 2 * self.d_inner + 2 * ngroups * d_state + self.n_heads
        self.in_proj = nn.Linear(d_model, projection_width, bias=bias)
        conv_width = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            conv_width, conv_width, kernel_size=d_conv, groups=conv_width,
            padding=d_conv - 1, bias=conv_bias,
        )
        if conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -float(conv_init), float(conv_init))

        dt = torch.exp(
            torch.rand(self.n_heads)
            * (math.log(float(dt_max)) - math.log(float(dt_min)))
            + math.log(float(dt_min))
        ).clamp(min=float(dt_init_floor))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        initial_A = torch.empty(self.n_heads, dtype=torch.float32).uniform_(
            float(A_init_range[0]), float(A_init_range[1])
        )
        self.A_log = nn.Parameter(torch.log(initial_A))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.norm = _GatedRMSNorm(self.d_inner, eps=norm_eps)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dt_bias._no_weight_decay = True
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        if learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.n_heads, headdim, d_state)
            )
            self.init_states._no_weight_decay = True
        else:
            self.register_parameter("init_states", None)

    def _discrete_scan(
        self, x: Tensor, dt: Tensor, B: Tensor, C: Tensor, *,
        initial_states: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch, length, _ = x.shape
        scan_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x_heads = x.reshape(batch, length, self.n_heads, self.headdim).to(scan_dtype)
        dt_scan = dt.to(scan_dtype)
        X = x_heads * dt_scan.unsqueeze(-1)
        A = dt_scan * (-torch.exp(self.A_log.to(scan_dtype))).view(1, 1, self.n_heads)
        B_scan, C_scan = B.to(scan_dtype), C.to(scan_dtype)

        if initial_states is None and self.init_states is not None:
            initial_states = self.init_states.unsqueeze(0).expand(batch, -1, -1, -1)
        elif initial_states is not None:
            if not isinstance(initial_states, Tensor):
                raise TypeError("initial_states must be a torch tensor or None")
            expected = (batch, self.n_heads, self.headdim, self.d_state)
            if tuple(initial_states.shape) != expected:
                raise ValueError(
                    "initial_states must have shape [batch, heads, headdim, d_state]"
                )
            if initial_states.device != x.device:
                raise ValueError("initial_states must be on the same device as the input")
            initial_states = initial_states.to(dtype=scan_dtype)

        # Padded entries have identity transitions and zero inputs.
        padding = (-length) % self.chunk_size
        if padding:
            X = torch.cat(
                [X, X.new_zeros(batch, padding, self.n_heads, self.headdim)], dim=1
            )
            A = torch.cat([A, A.new_zeros(batch, padding, self.n_heads)], dim=1)
            B_scan = torch.cat(
                [B_scan, B_scan.new_zeros(batch, padding, self.ngroups, self.d_state)],
                dim=1,
            )
            C_scan = torch.cat(
                [C_scan, C_scan.new_zeros(batch, padding, self.ngroups, self.d_state)],
                dim=1,
            )
        outputs, final_state = ssd_minimal_discrete(
            X, A, B_scan, C_scan, self.chunk_size,
            initial_states=initial_states,
        )
        return outputs[:, :length], final_state

    def forward(
        self, u: Tensor, *, initial_states: Tensor | None = None,
        return_final_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if not isinstance(u, Tensor) or u.ndim != 3 or u.shape[-1] != self.d_model:
            raise ValueError(f"u must have shape [batch, sequence, {self.d_model}]")
        if u.shape[0] <= 0 or u.shape[1] <= 0:
            raise ValueError("batch and sequence dimensions must be non-empty")
        if not u.is_floating_point():
            raise TypeError("u must use a floating-point dtype")
        if u.device != self.in_proj.weight.device:
            raise ValueError("u and mixer parameters must be on the same device")
        if type(return_final_state) is not bool:
            raise TypeError("return_final_state must be a bool")

        batch, length, _ = u.shape
        z, xBC, dt_raw = torch.split(
            self.in_proj(u),
            [
                self.d_inner,
                self.d_inner + 2 * self.ngroups * self.d_state,
                self.n_heads,
            ],
            dim=-1,
        )
        xBC = self.conv1d(xBC.transpose(1, 2))[..., :length].transpose(1, 2)
        xBC = F.silu(xBC)
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        B = B.reshape(batch, length, self.ngroups, self.d_state)
        C = C.reshape(batch, length, self.ngroups, self.d_state)
        dt = F.softplus(dt_raw + self.dt_bias.to(dtype=dt_raw.dtype))
        if self.dt_limit != (0.0, float("inf")):
            dt = dt.clamp(min=self.dt_limit[0], max=self.dt_limit[1])

        y, final_state = self._discrete_scan(
            x, dt, B, C, initial_states=initial_states
        )
        y = y.to(dtype=x.dtype)
        x_heads = x.reshape(batch, length, self.n_heads, self.headdim)
        y = y + self.D.to(dtype=x.dtype).view(1, 1, self.n_heads, 1) * x_heads
        y = y.reshape(batch, length, self.d_inner)
        output = self.out_proj(self.norm(y, z))
        if return_final_state:
            return output, final_state
        return output


__all__ = ["Mamba2ReferenceMixer"]
