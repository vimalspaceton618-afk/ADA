from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from ada.neural import ADADecoderBlock, Mamba2SSD, RMSNorm, TensorNetAttention


def test_rms_norm_preserves_shape_and_is_finite():
    norm = RMSNorm(12)
    x = torch.randn(2, 5, 12)
    y = norm(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mamba2_pdf_gated_convolution_is_causal():
    torch.manual_seed(7)
    mixer = Mamba2SSD(8, expand=2, kernel_size=4).eval()
    x = torch.randn(1, 6, 8)
    changed = x.clone()
    changed[:, 4:] += 10.0
    y1 = mixer(x)
    y2 = mixer(changed)
    torch.testing.assert_close(y1[:, :4], y2[:, :4])


def test_tensornet_attention_is_causal():
    torch.manual_seed(11)
    mixer = TensorNetAttention(12, 3, rank=4).eval()
    x = torch.randn(2, 6, 12)
    changed = x.clone()
    changed[:, 4:] += 50.0
    y1 = mixer(x)
    y2 = mixer(changed)
    torch.testing.assert_close(y1[:, :4], y2[:, :4])


@pytest.mark.parametrize("layer_idx,expected", [
    (0, "mamba2_conv"),
    (1, "tensornet_causal_attention"),
    (2, "mamba2_conv"),
    (3, "tensornet_causal_attention"),
])
def test_decoder_alternates_mixers(layer_idx, expected):
    block = ADADecoderBlock(
        dim=12, layer_idx=layer_idx, n_heads=3, attention_rank=4,
        num_experts=2, num_slots=5, expansion_factor=2,
    )
    assert block.mixer_type == expected


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_decoder_block_forward_aux_and_backward(layer_idx):
    block = ADADecoderBlock(
        dim=12, layer_idx=layer_idx, n_heads=3, attention_rank=4,
        num_experts=3, num_slots=7, expansion_factor=2,
        dropout=0.0,
    ).train()
    x = torch.randn(2, 5, 12, requires_grad=True)
    output, aux = block(x)

    assert output.shape == x.shape
    assert aux["router_probs"].shape == (2, 5, 3)
    assert aux["memory_attention"].shape == (2, 5, 7)
    loss = output.square().mean() + 0.01 * aux["load_balancing_loss"]
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert block.mixer.parameters().__next__().grad is not None


def test_attention_mask_zeros_padded_queries_and_blocks_padded_keys():
    mixer = TensorNetAttention(8, 2, rank=4).eval()
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    output = mixer(x, attention_mask=mask)
    assert torch.count_nonzero(output[:, 2:]) == 0
    assert torch.isfinite(output).all()


def test_decoder_rejects_bad_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        ADADecoderBlock(dim=10, layer_idx=1, n_heads=3)
    with pytest.raises(ValueError, match="layer_idx"):
        ADADecoderBlock(dim=12, layer_idx=-1)
