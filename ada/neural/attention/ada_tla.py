"""ADA-TLA: Adaptive Dual-state Linear Attention.

This module is a clean, typed PyTorch reference implementation derived from
ADA's research notebook (Ada-src.pdf, pp. 95-104 and 118-123).

It deliberately prioritizes mathematical transparency and autograd
correctness over performance. The parallel scan is a work-efficient
Blelloch scan expressed with Python lists of tensor nodes; it is not a fused
GPU kernel. Do not infer production throughput from this implementation.

Tensor conventions:
    q, k: [batch, heads, length, key_dim]
    v:    [batch, heads, length, value_dim]
    g:    [batch, heads, length]

The two recurrent states are:
    S_t = sigmoid(g_t) * S_(t-1) + phi(k_t)^T v_t
    z_t = sigmoid(g_t) * z_(t-1) + phi(k_t)

and output:
    y_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + eps)
"""
from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor
import torch.nn.functional as F

AffineState: TypeAlias = tuple[Tensor, Tensor]


def _validate_inputs(q: Tensor, k: Tensor, v: Tensor, g: Tensor, eps: float) -> None:
    if not all(isinstance(x, Tensor) for x in (q, k, v, g)):
        raise TypeError("q, k, v, and g must all be torch.Tensor instances")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 3:
        raise ValueError("expected q/k/v [B,H,L,D] and g [B,H,L]")
    if q.shape != k.shape:
        raise ValueError(f"q and k shapes must match; got {q.shape} and {k.shape}")
    if q.shape[:3] != v.shape[:3] or q.shape[:3] != g.shape:
        raise ValueError("batch, head, and sequence dimensions must match")
    if q.device != k.device or q.device != v.device or q.device != g.device:
        raise ValueError("all inputs must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype != g.dtype:
        raise ValueError("all inputs must have the same dtype")
    if not q.is_floating_point():
        raise TypeError("q, k, v, and g must use a floating-point dtype")
    if eps <= 0:
        raise ValueError("eps must be strictly positive")


def positive_feature_map(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Strictly positive feature map phi(x) = softplus(x) + eps."""
    if eps <= 0:
        raise ValueError("eps must be strictly positive")
    return F.softplus(x) + eps


def compose_affine(
    a2: Tensor, u2: Tensor, a1: Tensor, u1: Tensor
) -> AffineState:
    """Compose T2 after T1 for T(S) = a*S + U.

    The decay tensors have shape [..., 1], and update tensors have shape
    [..., key_dim, value_dim]. The result represents T2(T1(S)).
    """
    if a1.shape != a2.shape:
        raise ValueError("affine decay tensors must have identical shapes")
    if u1.shape != u2.shape:
        raise ValueError("affine update tensors must have identical shapes")
    if a1.shape != u1.shape[:-2] + (1,):
        raise ValueError("decay shape must match update leading dimensions + (1,)")
    return a2 * a1, a2.unsqueeze(-1) * u1 + u2


def blelloch_exclusive_scan(a: Tensor, u: Tensor) -> AffineState:
    """Return exclusive affine prefixes for each sequence position.

    Args:
        a: [B,H,L,1] multiplicative decay coefficients.
        u: [B,H,L,Dk,Dv] additive state updates.

    Returns:
        (prefix_a, prefix_u), where each prefix contains transformations
        strictly before that token. The identity transformation is (1, 0).

    This is an O(L)-work Blelloch tree scan. Tensor nodes are kept in Python
    lists to avoid unsafe in-place tensor writes and preserve autograd.
    """
    if a.ndim != 4 or u.ndim != 5:
        raise ValueError("expected a [B,H,L,1] and u [B,H,L,Dk,Dv]")
    b, h, length, one = a.shape
    if one != 1 or u.shape[:3] != (b, h, length):
        raise ValueError("incompatible affine scan shapes")
    if a.device != u.device or a.dtype != u.dtype:
        raise ValueError("a and u must share device and dtype")
    if length == 0:
        return a.clone(), u.clone()

    padded_length = 1 << (length - 1).bit_length()
    decays = [a[:, :, i, :] for i in range(length)]
    updates = [u[:, :, i, :, :] for i in range(length)]
    identity_a = torch.ones_like(decays[0])
    identity_u = torch.zeros_like(updates[0])
    decays.extend(identity_a for _ in range(padded_length - length))
    updates.extend(identity_u for _ in range(padded_length - length))

    # Up-sweep: each right node becomes the composition of its subtree.
    stride = 2
    while stride <= padded_length:
        half = stride // 2
        for right in range(stride - 1, padded_length, stride):
            left = right - half
            decays[right], updates[right] = compose_affine(
                decays[right], updates[right], decays[left], updates[left]
            )
        stride *= 2

    # Root becomes the identity, converting the reduction tree to prefixes.
    decays[-1] = identity_a
    updates[-1] = identity_u

    # Down-sweep: propagate the exclusive prefix to both children.
    stride = padded_length
    while stride >= 2:
        half = stride // 2
        for right in range(stride - 1, padded_length, stride):
            left = right - half
            parent_a, parent_u = decays[right], updates[right]
            left_a, left_u = decays[left], updates[left]
            decays[left], updates[left] = parent_a, parent_u
            decays[right], updates[right] = compose_affine(
                left_a, left_u, parent_a, parent_u
            )
        stride //= 2

    return (
        torch.stack(decays[:length], dim=2),
        torch.stack(updates[:length], dim=2),
    )


def _read_state(q_phi: Tensor, state: Tensor, normalizer: Tensor, eps: float) -> Tensor:
    numerator = torch.matmul(q_phi.unsqueeze(-2), state).squeeze(-2)
    denominator = torch.matmul(q_phi.unsqueeze(-2), normalizer).squeeze(-2)
    return numerator / (denominator + eps)


def ada_tla_sequential(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, eps: float = 1e-6
) -> Tensor:
    """Straightforward recurrent ADA-TLA reference implementation."""
    _validate_inputs(q, k, v, g, eps)
    batch, heads, length, key_dim = q.shape
    value_dim = v.shape[-1]
    q_phi = positive_feature_map(q, eps)
    k_phi = positive_feature_map(k, eps)
    gamma = torch.sigmoid(g)

    state = q.new_zeros((batch, heads, key_dim, value_dim))
    normalizer = q.new_zeros((batch, heads, key_dim, 1))
    outputs: list[Tensor] = []
    for index in range(length):
        decay = gamma[:, :, index, None, None]
        key = k_phi[:, :, index]
        value = v[:, :, index]
        state = decay * state + key.unsqueeze(-1) @ value.unsqueeze(-2)
        normalizer = decay * normalizer + key.unsqueeze(-1)
        outputs.append(_read_state(q_phi[:, :, index], state, normalizer, eps))
    if not outputs:
        return q.new_empty((batch, heads, 0, value_dim))
    return torch.stack(outputs, dim=2)


def ada_tla_parallel(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, eps: float = 1e-6
) -> Tensor:
    """Parallel-prefix ADA-TLA implementation, mathematically matching recurrence."""
    _validate_inputs(q, k, v, g, eps)
    q_phi = positive_feature_map(q, eps)
    k_phi = positive_feature_map(k, eps)
    gamma = torch.sigmoid(g)

    state_updates = k_phi.unsqueeze(-1) @ v.unsqueeze(-2)
    normalizer_updates = k_phi.unsqueeze(-1)
    decay = gamma.unsqueeze(-1)

    prefix_decay, prefix_state = blelloch_exclusive_scan(decay, state_updates)
    _, prefix_normalizer = blelloch_exclusive_scan(decay, normalizer_updates)

    inclusive_state = decay.unsqueeze(-1) * prefix_state + state_updates
    inclusive_normalizer = decay.unsqueeze(-1) * prefix_normalizer + normalizer_updates
    return _read_state(q_phi, inclusive_state, inclusive_normalizer, eps)
