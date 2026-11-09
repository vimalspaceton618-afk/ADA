"""Readable PyTorch reference for the chunked Mamba-2 SSD core.

This module implements the discrete SSD factorization (the reference algorithm
from the Mamba-2 paper) for grouped B/C parameters. It is intentionally a
correctness-oriented PyTorch implementation, not a fused Triton/CUDA kernel.

Inputs are already discretized:
  X: [batch, length, heads, head_dim] (the input multiplied by delta)
  A: [batch, length, heads] (the continuous A multiplied by delta)
  B,C: [batch, length, groups, state_dim]
The number of heads must be divisible by groups. B/C groups are broadcast
across their corresponding contiguous head groups.
"""
from __future__ import annotations

try:
    import torch
    from torch import Tensor
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "SSD reference requires PyTorch; install ADA with the neural extra."
    ) from exc


def _validate_inputs(
    X: Tensor, A: Tensor, B: Tensor, C: Tensor, block_len: int
) -> tuple[int, int, int, int, int, int]:
    if not all(isinstance(value, Tensor) for value in (X, A, B, C)):
        raise TypeError("X, A, B, and C must be torch tensors")
    if X.ndim != 4:
        raise ValueError("X must have shape [batch, length, heads, head_dim]")
    if A.ndim != 3:
        raise ValueError("A must have shape [batch, length, heads]")
    if B.ndim != 4 or C.ndim != 4:
        raise ValueError("B and C must have shape [batch, length, groups, state_dim]")
    batch, length, heads, head_dim = X.shape
    if batch <= 0 or length <= 0 or heads <= 0 or head_dim <= 0:
        raise ValueError("X dimensions must be non-empty")
    if A.shape != (batch, length, heads):
        raise ValueError("A shape must match X batch, length, and heads")
    if B.shape[:2] != (batch, length) or C.shape != B.shape:
        raise ValueError("B and C must share [batch, length, groups, state_dim]")
    groups, state_dim = B.shape[2:]
    if groups <= 0 or state_dim <= 0 or heads % groups:
        raise ValueError("groups must be positive and divide the number of heads")
    if any(value.device != X.device for value in (A, B, C)):
        raise ValueError("X, A, B, and C must be on the same device")
    if type(block_len) is not int or block_len <= 0:
        raise ValueError("block_len must be a positive integer")
    if length % block_len:
        raise ValueError("sequence length must be divisible by block_len")
    if any(value.dtype != X.dtype for value in (A, B, C)):
        raise TypeError("X, A, B, and C must have the same dtype")
    if not all(value.is_floating_point() for value in (X, A, B, C)):
        raise TypeError("X, A, B, and C must use floating-point dtypes")
    if not all(torch.isfinite(value).all().item() for value in (X, A, B, C)):
        raise ValueError("X, A, B, and C must contain only finite values")
    return batch, length, heads, head_dim, groups, state_dim


def segment_sum(x: Tensor) -> Tensor:
    """Stable lower-triangular segment sums over the final sequence axis."""
    if not isinstance(x, Tensor) or x.ndim < 1:
        raise ValueError("x must be a tensor with at least one dimension")
    length = x.shape[-1]
    if length <= 0:
        raise ValueError("segment dimension must be non-empty")
    expanded = x.unsqueeze(-1).expand(*x.shape, length)
    strict_lower = torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=x.device),
        diagonal=-1,
    )
    expanded = expanded.masked_fill(~strict_lower, 0)
    sums = torch.cumsum(expanded, dim=-2)
    lower = torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=x.device),
        diagonal=0,
    )
    return sums.masked_fill(~lower, -torch.inf)


def ssd_minimal_discrete(
    X: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    block_len: int,
    *,
    initial_states: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute the chunked SSD output and final state.

    This is the four-stage SSD factorization:
      1. intra-chunk (diagonal) output,
      2. intra-chunk state summaries,
      3. inter-chunk state recurrence,
      4. state-to-output (off-diagonal) contribution.

    Returns:
      Y: [batch, length, heads, head_dim]
      final_state: [batch, heads, head_dim, state_dim]
    """
    batch, length, heads, head_dim, groups, state_dim = _validate_inputs(
        X, A, B, C, block_len
    )
    chunks = length // block_len
    group_repeats = heads // groups

    if initial_states is None:
        initial_states = X.new_zeros(batch, heads, head_dim, state_dim)
    elif not isinstance(initial_states, Tensor):
        raise TypeError("initial_states must be a tensor or None")
    elif initial_states.shape != (batch, heads, head_dim, state_dim):
        raise ValueError(
            "initial_states must have shape [batch, heads, head_dim, state_dim]"
        )
    elif initial_states.device != X.device or initial_states.dtype != X.dtype:
        raise ValueError("initial_states must match X device and dtype")
    elif not torch.isfinite(initial_states).all().item():
        raise ValueError("initial_states must contain only finite values")

    # Split time into fixed-size chunks. Use float32 for exponentials and
    # cumulative sums, then cast contraction operands back to input dtype.
    Xc = X.reshape(batch, chunks, block_len, heads, head_dim)
    Ac = A.float().reshape(batch, chunks, block_len, heads).permute(0, 3, 1, 2)
    Bc = B.repeat_interleave(group_repeats, dim=2).reshape(
        batch, chunks, block_len, heads, state_dim
    )
    Cc = C.repeat_interleave(group_repeats, dim=2).reshape(
        batch, chunks, block_len, heads, state_dim
    )

    # 1. Intra-chunk output: lower-triangular SSD matrix.
    decay = torch.exp(segment_sum(Ac))
    y_diagonal = torch.einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp",
        Cc,
        Bc,
        decay,
        Xc,
    )

    # 2. Summarize each chunk's input contribution at its final position.
    cumulative_A = torch.cumsum(Ac, dim=-1)
    decay_to_boundary = torch.exp(
        cumulative_A[..., -1:].float() - cumulative_A.float()
    ).permute(0, 2, 3, 1)
    chunk_states = torch.einsum(
        "bclhn,bclhp,bclh->bchpn",
        Bc,
        Xc,
        decay_to_boundary.to(dtype=X.dtype),
    )

    # 3. Propagate initial and per-chunk states across chunk boundaries.
    boundary_A = cumulative_A[..., -1]
    padded_boundary_A = F.pad(boundary_A, (1, 0))
    chunk_decay = torch.exp(segment_sum(padded_boundary_A))
    states_with_initial = torch.cat(
        [initial_states.unsqueeze(1), chunk_states], dim=1
    )
    all_states = torch.einsum(
        "bhzc,bchpn->bzhpn", chunk_decay, states_with_initial
    )
    states_at_chunk_start = all_states[:, :-1]
    final_state = all_states[:, -1]

    # 4. Convert carried states back to outputs within each chunk.
    decay_from_chunk_start = torch.exp(cumulative_A.float()).permute(
        0, 2, 3, 1
    )
    y_off_diagonal = torch.einsum(
        "bclhn,bchpn,bhcl->bclhp",
        Cc,
        states_at_chunk_start,
        decay_from_chunk_start.to(dtype=X.dtype),
    )
    output = (y_diagonal + y_off_diagonal).reshape(
        batch, length, heads, head_dim
    )
    return output, final_state
