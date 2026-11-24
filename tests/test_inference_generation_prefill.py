"""Incremental generation must preserve prompt context and per-row EOS."""
from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
generation = importlib.import_module("18_INFERENCE.inference_engine")


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def step(self, tokens, state):
        assert not torch.is_grad_enabled()
        self.calls.append(tokens.clone())
        logits = torch.zeros(tokens.shape[0], 1, 4)
        logits[:, :, 2] = 12.0
        return logits, {"calls": len(self.calls)}


def test_first_call_prefills_all_prompt_tokens_and_later_calls_decode_one():
    runner = RecordingRunner()
    prompt = torch.tensor([[3, 1, 2]], dtype=torch.long)
    generated, state = generation.AutoregressiveGenerator(runner).generate(
        prompt, config=generation.GenerationConfig(max_new_tokens=2, top_k=1)
    )
    assert runner.calls[0].tolist() == prompt.tolist()
    assert runner.calls[1].shape == (1, 1)
    assert generated.tolist() == [[3, 1, 2, 2, 2]]
    assert state["calls"] == 2


def test_prefilled_state_uses_one_token_increment():
    runner = RecordingRunner()
    generation.AutoregressiveGenerator(runner).generate(
        torch.tensor([[1, 2, 3]]),
        config=generation.GenerationConfig(max_new_tokens=1, top_k=1),
        state={"already_prefilled": True},
    )
    assert runner.calls[0].tolist() == [[3]]


def test_finished_rows_remain_eos_while_other_rows_continue():
    class MixedEOSRunner:
        def __init__(self):
            self.calls = 0

        def step(self, tokens, state):
            self.calls += 1
            logits = torch.zeros(2, 1, 3)
            logits[0, :, 0] = 20.0   # row 0 ends on the first token
            logits[1, :, 1 if self.calls == 1 else 0] = 20.0
            return logits, state

    runner = MixedEOSRunner()
    tokens, _ = generation.AutoregressiveGenerator(runner).generate(
        torch.tensor([[1, 2], [2, 1]]),
        config=generation.GenerationConfig(
            max_new_tokens=4, top_k=1, eos_token_id=0
        ),
    )
    assert runner.calls == 2
    assert tokens.tolist() == [[1, 2, 0, 0], [2, 1, 1, 0]]


def test_malformed_runner_logits_are_rejected():
    class BadRunner:
        def step(self, tokens, state):
            return torch.zeros(1, 3), state

    with pytest.raises(ValueError, match="logits shaped"):
        generation.AutoregressiveGenerator(BadRunner()).generate(
            torch.tensor([[1, 2]]),
            config=generation.GenerationConfig(max_new_tokens=1),
        )
