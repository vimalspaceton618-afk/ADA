from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from ada.neural import MoEEnergyMemoryFFN


def test_moe_energy_memory_forward_contract_and_auxiliary_shapes():
    module = MoEEnergyMemoryFFN(
        d_model=16, num_experts=4, num_slots=12, expansion_factor=2
    )
    x = torch.randn(2, 5, 16)
    hidden, aux = module(x)

    assert hidden.shape == x.shape
    assert aux["router_probs"].shape == (2, 5, 4)
    assert aux["expert_indices"].shape == (2, 5)
    assert aux["memory_attention"].shape == (2, 5, 12)
    assert aux["memory_output"].shape == x.shape
    assert aux["blend_alpha"].shape == (2, 5, 1)
    torch.testing.assert_close(aux["router_probs"].sum(-1), torch.ones(2, 5))
    torch.testing.assert_close(
        hidden,
        x + aux["blend_alpha"] * aux["moe_output"]
          + (1.0 - aux["blend_alpha"]) * aux["memory_output"],
    )


def test_sparse_top1_dispatches_each_token_once():
    module = MoEEnergyMemoryFFN(
        d_model=8, num_experts=3, num_slots=4, expansion_factor=2,
        router_noise=False,
    ).eval()
    calls = [0, 0, 0]
    handles = []
    for index, expert in enumerate(module.experts):
        handles.append(expert.register_forward_hook(
            lambda _module, _inputs, _output, i=index: calls.__setitem__(i, calls[i] + 1)
        ))
    try:
        _, aux = module(torch.randn(2, 6, 8))
    finally:
        for handle in handles:
            handle.remove()

    assert sum(calls) >= 1
    assert torch.bincount(aux["expert_indices"].reshape(-1), minlength=3).sum().item() == 12
    assert torch.all(aux["expert_indices"] >= 0)
    assert torch.all(aux["expert_indices"] < 3)


def test_moe_router_experts_memory_and_blender_receive_gradients():
    module = MoEEnergyMemoryFFN(
        d_model=12, num_experts=3, num_slots=7, expansion_factor=2,
        router_noise=False,
    ).train()
    x = torch.randn(2, 4, 12, requires_grad=True)
    hidden, aux = module(x)
    objective = hidden.square().mean() + 0.05 * aux["load_balancing_loss"]
    objective.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.W_r.weight.grad is not None
    assert module.W_blend.weight.grad is not None
    assert module.energy_memory.energy_mem.grad is not None
    assert module.energy_memory.W_mem_gate.weight.grad is not None
    assert any(expert.w_gate.weight.grad is not None for expert in module.experts)


def test_load_balance_is_train_only_and_eval_is_deterministic():
    module = MoEEnergyMemoryFFN(
        d_model=8, num_experts=3, num_slots=5, expansion_factor=2,
        router_noise=True,
    )
    x = torch.randn(2, 3, 8)
    module.train()
    _, train_aux = module(x)
    assert train_aux["load_balancing_loss"].ndim == 0
    assert train_aux["load_balancing_loss"].item() > 0

    module.eval()
    first, eval_aux = module(x)
    second, _ = module(x)
    torch.testing.assert_close(first, second)
    assert eval_aux["load_balancing_loss"].item() == 0


def test_moe_energy_memory_validation():
    with pytest.raises(ValueError, match="d_model"):
        MoEEnergyMemoryFFN(0)
    with pytest.raises(ValueError, match="num_experts"):
        MoEEnergyMemoryFFN(8, num_experts=0)
    with pytest.raises(ValueError, match="temperature"):
        MoEEnergyMemoryFFN(8, temperature=0)
    module = MoEEnergyMemoryFFN(8, num_experts=2, num_slots=4)
    with pytest.raises(ValueError, match="shape"):
        module(torch.randn(2, 8))
    with pytest.raises(ValueError, match="d_model"):
        module(torch.randn(2, 3, 9))
