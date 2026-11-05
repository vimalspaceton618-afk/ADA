"""Numerical and autograd tests for ADA-TLA."""
import pytest
torch = pytest.importorskip("torch")

from ada.neural.attention.ada_tla import (
    ada_tla_parallel,
    ada_tla_sequential,
    blelloch_exclusive_scan,
    compose_affine,
)


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 7, 8, 13, 16, 31, 32, 33, 63, 64, 65])
def test_parallel_matches_sequential_for_arbitrary_lengths(length):
    torch.manual_seed(100 + length)
    q = torch.randn(2, 2, length, 4, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(2, 2, length, 5, dtype=torch.float64)
    g = torch.randn(2, 2, length, dtype=torch.float64)

    expected = ada_tla_sequential(q, k, v, g)
    actual = ada_tla_parallel(q, k, v, g)

    assert actual.shape == (2, 2, length, 5)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)


def test_parallel_and_sequential_gradients_match():
    torch.manual_seed(123)
    values = [
        torch.randn(1, 2, 13, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 2, 13, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 2, 13, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 2, 13, dtype=torch.float64, requires_grad=True),
    ]
    reference_inputs = [x.detach().clone().requires_grad_(True) for x in values]

    reference = ada_tla_sequential(*reference_inputs)
    parallel = ada_tla_parallel(*values)
    (reference.square().mean()).backward()
    (parallel.square().mean()).backward()

    for reference_input, parallel_input in zip(reference_inputs, values):
        torch.testing.assert_close(
            parallel_input.grad, reference_input.grad, rtol=1e-8, atol=1e-10
        )


def test_affine_composition_matches_nested_application():
    torch.manual_seed(9)
    a1, a2 = torch.rand(2, 3, 1), torch.rand(2, 3, 1)
    u1, u2 = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)
    state = torch.randn(2, 3, 4, 5)
    a, u = compose_affine(a2, u2, a1, u1)
    nested = a2.unsqueeze(-1) * (a1.unsqueeze(-1) * state + u1) + u2
    torch.testing.assert_close(a.unsqueeze(-1) * state + u, nested)


def test_scan_returns_exclusive_prefixes():
    a = torch.tensor([[[[0.5], [0.25], [0.75]]]], dtype=torch.float64)
    u = torch.tensor([[[[[1.0]], [[2.0]], [[3.0]]]]], dtype=torch.float64)
    prefix_a, prefix_u = blelloch_exclusive_scan(a, u)

    expected_a = torch.tensor([[[[1.0], [0.5], [0.125]]]], dtype=torch.float64)
    expected_u = torch.tensor([[[[[0.0]], [[1.0]], [[2.25]]]]], dtype=torch.float64)
    torch.testing.assert_close(prefix_a, expected_a)
    torch.testing.assert_close(prefix_u, expected_u)


@pytest.mark.parametrize(
    "mutator, error",
    [
        (lambda q, k, v, g: (q.squeeze(0), k, v, g), ValueError),
        (lambda q, k, v, g: (q, k, v[:, :, :-1], g), ValueError),
        (lambda q, k, v, g: (q, k, v, g.double()), ValueError),
    ],
)
def test_invalid_shapes_and_dtypes_fail_closed(mutator, error):
    q = torch.randn(1, 1, 3, 2)
    k = torch.randn_like(q)
    v = torch.randn(1, 1, 3, 2)
    g = torch.randn(1, 1, 3)
    with pytest.raises(error):
        ada_tla_parallel(*mutator(q, k, v, g))
