"""PyTorch CPU tests for ADA v4.5 routing and training objectives."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from ada import systems


def test_position_sorted_mod_router_preserves_tokens_and_backpropagates():
    router = systems.inference.adaptive_computation.MoDTokenRouter(8, capacity=0.5)
    x = torch.randn(2, 8, 8, requires_grad=True)
    seen = []
    def causal_block(selected):
        seen.append(selected.detach().clone())
        return selected * 0.25
    output, routing = router.forward(x, causal_block)
    assert output.shape == x.shape
    assert routing.indices.shape == (2, 4)
    assert torch.all(routing.indices[:, 1:] >= routing.indices[:, :-1])
    output.square().mean().backward()
    assert router.module.weight.grad is not None
    assert x.grad is not None
    assert len(seen) == 1


def test_bias_corrected_topk_moe_uses_unbiased_weights_and_sparse_dispatch():
    module = systems.inference.dynamic_routing
    router = module.BiasCorrectedTopKRouter(dim=8, n_experts=8, top_k=2, bias_lr=0.001)
    tokens = torch.randn(12, 8, requires_grad=True)
    routing = router.forward(tokens, training=True)
    assert routing.indices.shape == (12, 2)
    assert routing.weights.shape == (12, 2)
    assert torch.allclose(routing.weights.sum(-1), torch.ones(12), atol=1e-6)
    assert not router.expert_bias.requires_grad
    experts = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(8)])
    output = module.BiasCorrectedTopKRouter.dispatch(experts, tokens, routing)
    assert output.shape == tokens.shape
    output.square().mean().backward()
    assert router.module.weight.grad is not None
    assert tokens.grad is not None


def test_ada_v45_four_head_mtp_is_exact_two_term_objective():
    loss_fn = systems.training.post_training.multi_token_prediction_loss
    batch, length, vocab = 2, 8, 32
    ids = torch.randint(0, vocab, (batch, length))
    main = torch.randn(batch, length, vocab, requires_grad=True)
    mtp = [torch.randn(batch, length, vocab, requires_grad=True) for _ in range(4)]
    total, parts = loss_fn(main, mtp, ids, coefficient=0.3)
    expected = parts["main"] + 0.3 * parts["mtp"]
    assert torch.allclose(total, expected)
    assert len(parts["heads"]) == 4
    total.backward()
    assert main.grad is not None and all(x.grad is not None for x in mtp)


def test_ada_causal_training_step_respects_packed_sequence_loss_mask():
    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(24, 12)
            self.head = torch.nn.Linear(12, 24)
        def forward(self, ids): return self.head(self.embedding(ids))
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 24, (2, 8))
    mask = torch.tensor([[True] * 6 + [False] * 2, [True] * 8])
    result = systems.training.neural_training.train_causal_batch(
        model, optimizer, ids, loss_mask=mask
    )
    assert result["loss"] > 0 and result["grad_norm"] >= 0


def test_verified_checkpoint_loader_rejects_digest_mismatch(tmp_path):
    import hashlib
    torch_model = torch.nn.Linear(4, 2)
    path = tmp_path / "state.pt"
    torch.save(torch_model.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = torch.nn.Linear(4, 2)
    systems.inference.model_loader.load_state_dict_verified(loaded, path, digest)
    with pytest.raises(ValueError, match="digest mismatch"):
        systems.inference.model_loader.load_state_dict_verified(loaded, path, "0" * 64)
