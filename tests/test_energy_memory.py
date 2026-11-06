from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")
from ada.memory import EnergyMemory, EnergyMemoryOutput


def test_pdf_energy_memory_shapes_probabilities_and_gate():
    module = EnergyMemory(dim=32)
    result = module(torch.randn(2, 7, 32), return_aux=True)
    assert isinstance(result, EnergyMemoryOutput)
    assert result.output.shape == (2, 7, 32)
    assert result.attention.shape == (2, 7, 512)
    assert result.raw.shape == result.gate.shape == result.output.shape
    torch.testing.assert_close(result.attention.sum(-1), torch.ones(2, 7))
    assert torch.all((result.gate >= 0) & (result.gate <= 1))


def test_energy_memory_gradients_reach_input_bank_and_gate():
    module = EnergyMemory(dim=16, slots=8, temperature=0.7)
    x = torch.randn(2, 3, 16, requires_grad=True)
    module(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.energy_mem.grad is not None
    assert module.W_mem_gate.weight.grad is not None


def test_energy_memory_validation():
    with pytest.raises(ValueError, match="dim"):
        EnergyMemory(0)
    with pytest.raises(ValueError, match="slots"):
        EnergyMemory(8, slots=0)
    with pytest.raises(ValueError, match="temperature"):
        EnergyMemory(8, temperature=0)
    module = EnergyMemory(8)
    with pytest.raises(ValueError, match="shape"):
        module(torch.randn(2, 8))
    with pytest.raises(ValueError, match="final dimension"):
        module(torch.randn(2, 4, 9))


def test_energy_memory_state_dict_and_deterministic_eval():
    module = EnergyMemory(12, slots=6).eval()
    x = torch.randn(1, 4, 12)
    expected = module(x)
    torch.testing.assert_close(module(x), expected)
    restored = EnergyMemory(12, slots=6)
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored(x), expected)


def test_hopfield_energy_gradient_matches_x_minus_associative_retrieval():
    module = EnergyMemory(
        dim=5, slots=7, temperature=0.7, init_std=0.2, bias=False
    ).double()
    x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)

    value = module.energy(x).sum()
    gradient, = torch.autograd.grad(value, x)

    scores = x.detach() @ module.energy_mem.detach().transpose(0, 1)
    attention = torch.softmax(scores / module.temperature, dim=-1)
    expected = x.detach() - attention @ module.energy_mem.detach()
    torch.testing.assert_close(gradient, expected, rtol=1e-9, atol=1e-9)
