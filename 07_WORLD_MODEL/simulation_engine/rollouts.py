"""Isolated latent rollouts over the learned world-model interface."""
from __future__ import annotations
from dataclasses import dataclass
try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("Latent rollouts require PyTorch; install ADA with [neural].") from exc


@dataclass(frozen=True)
class LatentRollout:
    latent_features: Tensor
    predicted_rewards: Tensor
    continuation_probabilities: Tensor
    discounted_return: Tensor
    actions: Tensor


def _expand_state(state, count: int):
    from ..environment_model.rssm import RSSMState
    if not isinstance(state, RSSMState):
        raise TypeError("initial_state must be an RSSMState")
    if state.deterministic.shape[0] != 1:
        raise ValueError("rollout branching expects one initial state")
    return RSSMState(*(value.expand(count, *value.shape[1:]).clone() for value in state))


def imagine_rollout(world_model, initial_state, actions: Tensor, *, discount: float = 0.99,
                    sample: bool = False) -> LatentRollout:
    """Run one candidate sequence without modifying the supplied state."""
    if not isinstance(actions, Tensor) or actions.ndim != 3 or actions.shape[0] != 1:
        raise ValueError("actions must be [1,horizon,action_dim]")
    if isinstance(discount, bool) or not isinstance(discount, (int, float)) or not 0 < discount <= 1:
        raise ValueError("discount must be in (0,1]")
    with torch.no_grad():
        output = world_model.imagine(initial_state, actions, sample=sample)
        rewards = output["reward_predictions"]
        continuation = torch.sigmoid(output["continuation_logits"])
        weights = torch.pow(
            torch.tensor(float(discount), device=rewards.device, dtype=rewards.dtype),
            torch.arange(rewards.shape[1], device=rewards.device, dtype=rewards.dtype),
        ).unsqueeze(0)
        survival = torch.cumprod(
            torch.cat((torch.ones_like(continuation[:, :1]), continuation[:, :-1]), dim=1),
            dim=1,
        )
        returns = (weights * survival * rewards).sum(dim=1)
        return LatentRollout(
            output["latent_features"], rewards, continuation, returns, actions.clone()
        )


def compare_rollouts(world_model, initial_state, action_sequences: Tensor, *,
                     discount: float = 0.99, sample: bool = False) -> tuple[LatentRollout, ...]:
    """Branch multiple action sequences from the same immutable latent state."""
    if not isinstance(action_sequences, Tensor) or action_sequences.ndim != 3:
        raise ValueError("action_sequences must be [candidates,horizon,action_dim]")
    if action_sequences.shape[0] < 1:
        raise ValueError("at least one candidate sequence is required")
    state = _expand_state(initial_state, action_sequences.shape[0])
    with torch.no_grad():
        output = world_model.imagine(state, action_sequences, sample=sample)
        rewards = output["reward_predictions"]
        continuation = torch.sigmoid(output["continuation_logits"])
        powers = torch.pow(
            torch.tensor(float(discount), device=rewards.device, dtype=rewards.dtype),
            torch.arange(rewards.shape[1], device=rewards.device, dtype=rewards.dtype),
        ).unsqueeze(0)
        survival = torch.cumprod(
            torch.cat((torch.ones_like(continuation[:, :1]), continuation[:, :-1]), dim=1),
            dim=1,
        )
        scores = (powers * survival * rewards).sum(dim=1)
        return tuple(
            LatentRollout(
                output["latent_features"][i:i+1], rewards[i:i+1], continuation[i:i+1],
                scores[i:i+1], action_sequences[i:i+1].clone()
            ) for i in range(action_sequences.shape[0])
        )
