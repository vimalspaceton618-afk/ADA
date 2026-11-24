"""Native ADA causal-LM interrupted/resumed training equivalence on CPU.

This uses deterministic synthetic token IDs, NOT a FineWeb network download.
It exercises real ADA weights, optimizer state, stochastic MoE routing, the
published checkpoint bundle, restricted deserialization and replay offset.
"""
from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")

from ada.atomic_checkpoint import commit_checkpoint, resolve_committed_checkpoint
from ada.neural import ADAConfig, ADAForCausalLM
from ada.pretraining_resume import replay_completed_batches


def _config():
    return ADAConfig(
        vocab_size=19, dim=8, n_layers=2, n_heads=2,
        attention_rank=4, max_seq_len=6, num_experts=2,
        num_slots=4, expansion_factor=1, mixer_expand=2,
        mixer_kernel_size=2, dropout=0.0,
        mixer_backend="mamba2_reference", mamba2_d_state=3,
        mamba2_headdim=4, mamba2_ngroups=1, mamba2_chunk_size=2,
    )


def _step(model, optimizer, ids):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(ids, labels=ids)
    total = output.loss + 0.01 * output.auxiliary_loss
    assert torch.isfinite(total).item()
    total.backward()
    optimizer.step()
    return float(total.detach())


def _new_training_pair(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    model = ADAForCausalLM(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007)
    return model, optimizer


def test_real_ada_optimizer_resume_matches_uninterrupted_training(tmp_path):
    tokens = [
        torch.randint(0, 19, (1, 5), generator=torch.Generator().manual_seed(90 + i))
        for i in range(3)
    ]
    continuous_model, continuous_optimizer = _new_training_pair(1234)
    continuous_loss = [
        _step(continuous_model, continuous_optimizer, ids) for ids in tokens
    ]
    expected_weights = {
        name: weight.detach().clone()
        for name, weight in continuous_model.state_dict().items()
    }

    interrupted_model, interrupted_optimizer = _new_training_pair(1234)
    first_loss = _step(interrupted_model, interrupted_optimizer, tokens[0])
    assert first_loss == pytest.approx(continuous_loss[0], rel=1e-7)
    saved_torch_rng = torch.get_rng_state().clone()
    saved_python_rng = random.getstate()

    def save_bundle(stage):
        interrupted_model.save_pretrained(stage / "model")
        torch.save(
            {
                "step": 1,
                "tokens_seen": 4,
                "optimizer": interrupted_optimizer.state_dict(),
                "torch_rng_state": saved_torch_rng,
                "python_rng_state": saved_python_rng,
                "seed": 1234,
            },
            stage / "training_state.pt",
        )
        (stage / "tokenizer.json").write_text('{"dummy_tokenizer": true}')
        (stage / "run_manifest.json").write_text(
            json.dumps({"batch_size": 1, "sequence_length": 4, "seed": 1234})
        )

    commit_checkpoint(
        tmp_path, step=1, tokens_seen=4, writer=save_bundle
    )

    # A newly constructed model consumes global RNG during initialization.
    # Restore the saved RNG state after all model/optimizer construction.
    committed = resolve_committed_checkpoint(tmp_path)
    resumed_model = ADAForCausalLM.from_pretrained(committed / "model")
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.0007)
    state = torch.load(
        committed / "training_state.pt", map_location="cpu", weights_only=True
    )
    resumed_optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng_state"])
    random.setstate(state["python_rng_state"])
    assert torch.equal(torch.get_rng_state(), saved_torch_rng)
    assert random.getstate() == saved_python_rng

    remaining = replay_completed_batches(iter(tokens), state["step"])
    resumed_loss = [
        _step(resumed_model, resumed_optimizer, ids) for ids in remaining
    ]
    assert resumed_loss == pytest.approx(continuous_loss[1:], rel=1e-6, abs=1e-7)
    for name, expected in expected_weights.items():
        torch.testing.assert_close(
            resumed_model.state_dict()[name], expected, rtol=1e-6, atol=1e-7,
        )
