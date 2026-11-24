"""The generic training adapter must work with ADA's native model output."""
from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")

from ada.neural import ADAConfig, ADAForCausalLM

train_causal_batch = importlib.import_module(
    "17_TRAINING.neural_training"
).train_causal_batch


def test_native_ada_model_can_train_via_generic_adapter():
    torch.manual_seed(123)
    model = ADAForCausalLM(ADAConfig(
        vocab_size=23, dim=8, n_layers=1, n_heads=2, attention_rank=4,
        max_seq_len=8, num_experts=2, num_slots=4,
        expansion_factor=2, mixer_expand=2, mixer_kernel_size=2,
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    tokens = torch.randint(0, 23, (2, 5), dtype=torch.long)
    original = model.token_embeddings.weight.detach().clone()
    result = train_causal_batch(model, optimizer, tokens)
    assert result["loss"] > 0
    assert result["grad_norm"] > 0
    assert torch.isfinite(torch.tensor(list(result.values()))).all()
    assert not torch.equal(original, model.token_embeddings.weight.detach())


def test_adapter_rejects_a_structured_output_with_invalid_logits():
    class Broken(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(()))

        def forward(self, input_ids):
            from types import SimpleNamespace
            return SimpleNamespace(logits="not a tensor")

    model = Broken()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="tensor logits"):
        train_causal_batch(model, optimizer, torch.ones(1, 3, dtype=torch.long))
