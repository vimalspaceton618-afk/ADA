from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from ada.neural import ADAConfig, ADAForCausalLM


def tiny_config(**kwargs):
    values = dict(
        vocab_size=41, dim=12, n_layers=2, n_heads=3, attention_rank=4,
        max_seq_len=16, num_experts=3, num_slots=7, expansion_factor=2,
        mixer_expand=2, mixer_kernel_size=3, dropout=0.0,
    )
    values.update(kwargs)
    return ADAConfig(**values)


def test_full_causal_lm_forward_and_shifted_loss():
    model = ADAForCausalLM(tiny_config()).train()
    ids = torch.randint(0, 41, (2, 7))
    result = model(ids, labels=ids)
    assert result.logits.shape == (2, 7, 41)
    assert result.hidden_states.shape == (2, 7, 12)
    assert result.loss is not None and torch.isfinite(result.loss)
    assert result.auxiliary_loss.ndim == 0
    assert len(result.layer_aux) == 2
    (result.loss + 0.01 * result.auxiliary_loss).backward()
    assert model.token_embeddings.weight.grad is not None
    assert model.layers[0].ffn.W_r.weight.grad is not None
    assert model.lm_head.weight is model.token_embeddings.weight


def test_embedding_initialization_is_scaled_for_tied_lm_head():
    torch.manual_seed(17)
    model = ADAForCausalLM(tiny_config(pad_token_id=0))
    token_std = float(model.token_embeddings.weight.detach().std())
    position_std = float(model.position_embeddings.weight.detach().std())
    assert 0.015 < token_std < 0.025
    assert 0.015 < position_std < 0.025
    assert torch.count_nonzero(model.token_embeddings.weight[0]) == 0

    ids = torch.randint(1, 41, (2, 7))
    result = model(ids, labels=ids)
    assert result.loss is not None
    assert float(result.loss.detach()) < 4.5  # near log(vocab_size), not O(10^2)


def test_untied_head_and_parameter_count():
    model = ADAForCausalLM(tiny_config(tie_embeddings=False))
    assert model.lm_head.weight is not model.token_embeddings.weight
    assert model.parameter_count() >= model.parameter_count(trainable_only=True) > 0


def test_padding_mask_is_inferred_from_config():
    model = ADAForCausalLM(tiny_config(pad_token_id=0)).eval()
    ids = torch.tensor([[4, 5, 0, 0]])
    result = model(ids)
    assert torch.isfinite(result.logits).all()


def test_sequence_and_input_validation():
    model = ADAForCausalLM(tiny_config())
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.ones(1, 17, dtype=torch.long))
    with pytest.raises(ValueError, match="vocabulary"):
        model(torch.tensor([[41, 2]]))
    with pytest.raises(ValueError, match="labels"):
        model(torch.ones(1, 4, dtype=torch.long), labels=torch.ones(1, 3, dtype=torch.long))


def test_save_and_load_round_trip(tmp_path):
    model = ADAForCausalLM(tiny_config()).eval()
    ids = torch.randint(0, 41, (1, 5))
    expected = model(ids).logits
    model.save_pretrained(tmp_path)
    restored = ADAForCausalLM.from_pretrained(tmp_path).eval()
    actual = restored(ids).logits
    torch.testing.assert_close(actual, expected)
    assert restored.config == model.config


@pytest.mark.parametrize("kwargs", [
    {"vocab_size": 0}, {"dim": 0}, {"n_layers": 0},
    {"dim": 10, "n_heads": 3}, {"dropout": 1.0},
])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        tiny_config(**kwargs)
