"""Correctness tests for the chunked SSD reference implementation."""
from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from ada.neural import ssd_minimal_discrete


def sequential_reference(X, A, B, C, initial):
    batch, length, heads, head_dim = X.shape
    groups, state_dim = B.shape[2:]
    repeats = heads // groups
    expanded_B = B.repeat_interleave(repeats, dim=2)
    expanded_C = C.repeat_interleave(repeats, dim=2)
    state = initial
    outputs = []
    for index in range(length):
        state = state + torch.einsum(
            "bhp,bhn->bhpn", X[:, index], expanded_B[:, index]
        )
        outputs.append(
            torch.einsum("bhpn,bhn->bhp", state, expanded_C[:, index])
        )
        state = state * torch.exp(A[:, index]).unsqueeze(-1).unsqueeze(-1)
    return torch.stack(outputs, dim=1), state


@pytest.mark.parametrize("block_len", [1, 2, 4, 8])
def test_chunked_ssd_matches_sequential_recurrence(block_len):
    torch.manual_seed(17)
    batch, length, heads, head_dim, groups, state_dim = 2, 8, 4, 3, 2, 5
    X = torch.randn(batch, length, heads, head_dim, dtype=torch.float64)
    A = -torch.rand(batch, length, heads, dtype=torch.float64)
    B = torch.randn(batch, length, groups, state_dim, dtype=torch.float64)
    C = torch.randn(batch, length, groups, state_dim, dtype=torch.float64)
    initial = torch.randn(batch, heads, head_dim, state_dim, dtype=torch.float64)

    actual, actual_final = ssd_minimal_discrete(
        X, A, B, C, block_len, initial_states=initial
    )
    expected, expected_final = sequential_reference(X, A, B, C, initial)

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(actual_final, expected_final, rtol=1e-9, atol=1e-9)


def test_ssd_is_causal_with_respect_to_future_inputs():
    torch.manual_seed(23)
    X = torch.randn(1, 8, 2, 4)
    A = -torch.rand(1, 8, 2)
    B = torch.randn(1, 8, 1, 3)
    C = torch.randn(1, 8, 1, 3)
    changed = 5

    baseline, _ = ssd_minimal_discrete(X, A, B, C, 4)
    X2, A2, B2, C2 = X.clone(), A.clone(), B.clone(), C.clone()
    X2[:, changed:] += 3.0
    A2[:, changed:] -= 0.5
    B2[:, changed:] *= 2.0
    C2[:, changed:] -= 1.0
    altered, _ = ssd_minimal_discrete(X2, A2, B2, C2, 4)

    torch.testing.assert_close(
        baseline[:, :changed], altered[:, :changed], rtol=0, atol=0
    )


def test_ssd_backward_produces_finite_gradients():
    torch.manual_seed(31)
    X = torch.randn(1, 8, 2, 3, requires_grad=True)
    A = (-torch.rand(1, 8, 2)).requires_grad_()
    B = torch.randn(1, 8, 1, 4, requires_grad=True)
    C = torch.randn(1, 8, 1, 4, requires_grad=True)

    output, final_state = ssd_minimal_discrete(X, A, B, C, 4)
    (output.square().mean() + final_state.square().mean()).backward()

    for value in (X, A, B, C):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_ssd_rejects_invalid_shapes_and_chunk_lengths():
    X = torch.randn(1, 8, 2, 4)
    A = -torch.rand(1, 8, 2)
    B = torch.randn(1, 8, 1, 3)
    C = torch.randn(1, 8, 1, 3)

    with pytest.raises(ValueError, match="divisible"):
        ssd_minimal_discrete(X, A, B, C, 3)
    with pytest.raises(ValueError, match="initial_states"):
        ssd_minimal_discrete(
            X, A, B, C, 4, initial_states=torch.zeros(1, 2, 4, 3)
        )
    with pytest.raises(ValueError, match="groups"):
        ssd_minimal_discrete(X, A, B.repeat(1, 1, 3, 1), C.repeat(1, 1, 3, 1), 4)
