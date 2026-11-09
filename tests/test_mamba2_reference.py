from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")
from torch.nn import functional as F

from ada.neural import Mamba2ReferenceMixer, ADAConfig, ADAForCausalLM


def test_mamba2_reference_mixer_shapes_and_final_state_with_partial_chunk():
    mixer = Mamba2ReferenceMixer(
        12, d_state=5, d_conv=3, expand=2, headdim=4,
        ngroups=2, chunk_size=4,
    )
    x = torch.randn(2, 7, 12)
    output, final_state = mixer(x, return_final_state=True)
    assert output.shape == x.shape
    assert final_state.shape == (2, 6, 4, 5)
    assert final_state.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()


def test_mamba2_reference_mixer_is_causal():
    torch.manual_seed(31)
    mixer = Mamba2ReferenceMixer(
        8, d_state=4, d_conv=3, expand=2, headdim=4,
        ngroups=1, chunk_size=4,
    ).eval()
    x = torch.randn(1, 9, 8)
    changed = x.clone()
    changed[:, 5:] += 25.0
    torch.testing.assert_close(mixer(x)[:, :5], mixer(changed)[:, :5])


def test_mamba2_reference_mixer_backpropagates_through_core_parameters():
    torch.manual_seed(37)
    mixer = Mamba2ReferenceMixer(
        8, d_state=4, d_conv=3, expand=2, headdim=4,
        ngroups=2, chunk_size=4, learnable_init_states=True,
    )
    x = torch.randn(2, 6, 8, requires_grad=True)
    mixer(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in (
        mixer.in_proj.weight, mixer.conv1d.weight, mixer.dt_bias,
        mixer.A_log, mixer.D, mixer.norm.weight, mixer.out_proj.weight,
        mixer.init_states,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_mamba2_reference_mixer_matches_explicit_token_recurrence():
    torch.manual_seed(41)
    mixer = Mamba2ReferenceMixer(
        8, d_state=3, d_conv=2, expand=2, headdim=4,
        ngroups=2, chunk_size=2,
    ).eval()
    x = torch.randn(1, 5, 8)
    projected = mixer.in_proj(x)
    z, xBC, dt_raw = torch.split(
        projected,
        [mixer.d_inner, mixer.d_inner + 2 * mixer.ngroups * mixer.d_state, mixer.n_heads],
        dim=-1,
    )
    length = x.shape[1]
    xBC = F.silu(mixer.conv1d(xBC.transpose(1, 2))[..., :length].transpose(1, 2))
    scan_x, B, C = torch.split(
        xBC,
        [mixer.d_inner, mixer.ngroups * mixer.d_state, mixer.ngroups * mixer.d_state],
        dim=-1,
    )
    scan_x = scan_x.view(1, length, mixer.n_heads, mixer.headdim)
    B = B.view(1, length, mixer.ngroups, mixer.d_state)
    C = C.view(1, length, mixer.ngroups, mixer.d_state)
    dt = F.softplus(dt_raw + mixer.dt_bias)
    A = -torch.exp(mixer.A_log)
    repeated_B = B.repeat_interleave(mixer.n_heads // mixer.ngroups, dim=2)
    repeated_C = C.repeat_interleave(mixer.n_heads // mixer.ngroups, dim=2)
    state = torch.zeros(1, mixer.n_heads, mixer.headdim, mixer.d_state)
    serial_outputs = []
    for index in range(length):
        dt_t, x_t = dt[:, index], scan_x[:, index]
        state = (
            state * torch.exp(dt_t * A).view(1, mixer.n_heads, 1, 1)
            + x_t.unsqueeze(-1) * dt_t.view(1, mixer.n_heads, 1, 1)
            * repeated_B[:, index].unsqueeze(2)
        )
        y_t = torch.einsum("bhpn,bhn->bhp", state, repeated_C[:, index])
        y_t = y_t + mixer.D.view(1, mixer.n_heads, 1) * x_t
        serial_outputs.append(y_t)
    serial = torch.stack(serial_outputs, dim=1).reshape(1, length, mixer.d_inner)
    expected = mixer.out_proj(mixer.norm(serial, z))
    torch.testing.assert_close(mixer(x), expected, rtol=2e-5, atol=2e-5)


def test_mamba2_reference_backend_can_be_selected_for_full_ada_model():
    config = ADAConfig(
        vocab_size=29, dim=12, n_layers=2, n_heads=3, attention_rank=4,
        max_seq_len=12, num_experts=2, num_slots=5, expansion_factor=2,
        mixer_expand=2, mixer_kernel_size=3, dropout=0.0,
        mixer_backend="mamba2_reference", mamba2_d_state=4,
        mamba2_headdim=4, mamba2_ngroups=2, mamba2_chunk_size=4,
    )
    model = ADAForCausalLM(config).train()
    ids = torch.randint(0, 29, (2, 7))
    result = model(ids, labels=ids)
    assert result.logits.shape == (2, 7, 29)
    assert result.loss is not None and torch.isfinite(result.loss)
    (result.loss + 0.01 * result.auxiliary_loss).backward()
    assert model.layers[0].mixer.A_log.grad is not None
    assert model.layers[0].mixer.A_log.grad.isfinite().all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 0},
        {"d_model": 8, "headdim": 3},
        {"d_model": 8, "expand": 2, "headdim": 4, "ngroups": 3},
        {"d_model": 8, "chunk_size": 0},
        {"d_model": 8, "dt_min": 0.0},
    ],
)
def test_mamba2_reference_mixer_rejects_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        Mamba2ReferenceMixer(**kwargs)
