"""Optional PyTorch recurrent state-space dynamics for ADA World Model."""
from __future__ import annotations
from typing import NamedTuple
try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("RSSM requires PyTorch. Install ADA with the [neural] extra.") from exc


class RSSMState(NamedTuple):
    """One recurrent latent state; stochastic is [..., groups, classes]."""
    deterministic: Tensor
    stochastic: Tensor
    logits: Tensor

    @property
    def features(self) -> Tensor:
        return torch.cat((self.deterministic, self.stochastic.flatten(start_dim=-2)), dim=-1)


class RSSMSequence(NamedTuple):
    """Posterior and prior state sequences, batch-first [B,T,...]."""
    posterior: RSSMState
    prior: RSSMState


class RecurrentStateSpaceModel(nn.Module):
    """Compact action-conditioned RSSM with grouped categorical stochastic latents.

    Inputs are dense observation embeddings rather than raw pixels. A visual
    encoder can be connected upstream without coupling this module to a model.
    """
    def __init__(self, observation_dim: int, action_dim: int, *,
                 deterministic_dim: int = 256, stochastic_groups: int = 16,
                 classes: int = 16, hidden_dim: int = 256, unimix: float = 0.01):
        super().__init__()
        for name, value in (
            ("observation_dim", observation_dim), ("action_dim", action_dim),
            ("deterministic_dim", deterministic_dim), ("stochastic_groups", stochastic_groups),
            ("classes", classes), ("hidden_dim", hidden_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(unimix, bool) or not isinstance(unimix, (int, float)):
            raise TypeError("unimix must be numeric")
        if not 0.0 <= float(unimix) < 1.0:
            raise ValueError("unimix must be in [0, 1)")
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.deterministic_dim, self.stochastic_groups = deterministic_dim, stochastic_groups
        self.classes, self.hidden_dim, self.unimix = classes, hidden_dim, float(unimix)
        stochastic_dim = stochastic_groups * classes
        self.stochastic_projection = nn.Linear(stochastic_dim, hidden_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.recurrent = nn.GRUCell(hidden_dim, deterministic_dim)
        self.prior_head = nn.Sequential(nn.Linear(deterministic_dim, hidden_dim), nn.SiLU(),
                                        nn.Linear(hidden_dim, stochastic_dim))
        self.posterior_head = nn.Sequential(
            nn.Linear(deterministic_dim + observation_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, stochastic_dim))

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> RSSMState:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        parameter = next(self.parameters())
        device, dtype = device or parameter.device, dtype or parameter.dtype
        deter = torch.zeros(batch_size, self.deterministic_dim, device=device, dtype=dtype)
        stoch = torch.full((batch_size, self.stochastic_groups, self.classes),
                           1.0 / self.classes, device=device, dtype=dtype)
        return RSSMState(deter, stoch, torch.zeros_like(stoch))

    def _validate_state(self, state: RSSMState) -> None:
        if not isinstance(state, RSSMState):
            raise TypeError("state must be an RSSMState")
        batch = state.deterministic.shape[0] if state.deterministic.ndim == 2 else -1
        expected = ((batch, self.deterministic_dim),
                    (batch, self.stochastic_groups, self.classes),
                    (batch, self.stochastic_groups, self.classes))
        for value, shape, name in zip(state, expected, ("deterministic", "stochastic", "logits")):
            if value.ndim != len(shape) or tuple(value.shape) != shape:
                raise ValueError(f"state.{name} must have shape {shape}")
            if not torch.is_floating_point(value) or not torch.isfinite(value).all():
                raise ValueError(f"state.{name} must contain finite floating-point values")

    def _sample(self, logits: Tensor, *, sample: bool) -> Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        probs = (1.0 - self.unimix) * probs + self.unimix / self.classes
        if sample:
            value = F.gumbel_softmax(probs.clamp_min(1e-8).log(), tau=1.0, hard=True, dim=-1)
        else:
            value = F.one_hot(probs.argmax(dim=-1), num_classes=self.classes).to(probs.dtype)
        return value.to(dtype=logits.dtype)

    def _dynamics(self, state: RSSMState, action: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_state(state)
        if not isinstance(action, Tensor) or action.ndim != 2:
            raise ValueError("action must be a [batch, action_dim] tensor")
        if action.shape != (state.deterministic.shape[0], self.action_dim):
            raise ValueError("action shape does not match batch and action_dim")
        if action.device != state.deterministic.device:
            raise ValueError("action and state must share a device")
        if not torch.isfinite(action).all():
            raise ValueError("action must contain finite values")
        x = self.stochastic_projection(state.stochastic.flatten(start_dim=1)) + self.action_projection(action)
        deter = self.recurrent(F.silu(self.input_norm(x)), state.deterministic)
        logits = self.prior_head(deter).view(-1, self.stochastic_groups, self.classes)
        return deter, logits

    def observe_step(self, state: RSSMState, action: Tensor, observation_embedding: Tensor,
                     *, sample: bool = True) -> tuple[RSSMState, RSSMState]:
        """Assimilate one embedding and return (posterior_state, prior_state)."""
        deter, prior_logits = self._dynamics(state, action)
        if not isinstance(observation_embedding, Tensor) or observation_embedding.ndim != 2:
            raise ValueError("observation_embedding must be [batch, observation_dim]")
        if observation_embedding.shape != (deter.shape[0], self.observation_dim):
            raise ValueError("observation_embedding shape does not match RSSM configuration")
        if observation_embedding.device != deter.device or not torch.isfinite(observation_embedding).all():
            raise ValueError("observation embedding must be finite and share the state device")
        prior = RSSMState(deter, self._sample(prior_logits, sample=sample), prior_logits)
        posterior_logits = self.posterior_head(torch.cat((deter, observation_embedding), -1))
        posterior_logits = posterior_logits.view(-1, self.stochastic_groups, self.classes)
        posterior = RSSMState(deter, self._sample(posterior_logits, sample=sample), posterior_logits)
        return posterior, prior

    def imagine_step(self, state: RSSMState, action: Tensor, *, sample: bool = True) -> RSSMState:
        """Advance from a latent state using only the learned prior."""
        deter, logits = self._dynamics(state, action)
        return RSSMState(deter, self._sample(logits, sample=sample), logits)

    @staticmethod
    def _stack(states: list[RSSMState]) -> RSSMState:
        if not states:
            raise ValueError("cannot stack an empty RSSM sequence")
        return RSSMState(*(torch.stack([state[i] for state in states], dim=1) for i in range(3)))

    def observe_sequence(self, observation_embeddings: Tensor, actions: Tensor, *,
                         is_first: Tensor | None = None, sample: bool = True) -> RSSMSequence:
        """Filter [B,T,E] observations with [B,T-1,A] preceding actions."""
        if not isinstance(observation_embeddings, Tensor) or observation_embeddings.ndim != 3:
            raise ValueError("observation_embeddings must be [batch,time,observation_dim]")
        batch, length, width = observation_embeddings.shape
        if batch < 1 or length < 1 or width != self.observation_dim:
            raise ValueError("observation_embeddings dimensions do not match configuration")
        if not isinstance(actions, Tensor) or actions.shape != (batch, length - 1, self.action_dim):
            raise ValueError("actions must have shape [batch,time-1,action_dim]")
        if actions.device != observation_embeddings.device:
            raise ValueError("actions and observations must share a device")
        if not torch.isfinite(observation_embeddings).all() or not torch.isfinite(actions).all():
            raise ValueError("sequence inputs must be finite")
        if is_first is not None:
            if not isinstance(is_first, Tensor) or is_first.shape != (batch, length) or is_first.dtype != torch.bool:
                raise ValueError("is_first must be a boolean [batch,time] tensor")
            if is_first.device != observation_embeddings.device:
                raise ValueError("is_first and observations must share a device")

        state = self.initial_state(batch, device=observation_embeddings.device,
                                   dtype=observation_embeddings.dtype)
        zero_action = torch.zeros(batch, self.action_dim, device=actions.device, dtype=actions.dtype)
        posterior_states, prior_states = [], []
        for t in range(length):
            if is_first is not None:
                reset = is_first[:, t].unsqueeze(-1)
                fresh = self.initial_state(batch, device=state.deterministic.device,
                                           dtype=state.deterministic.dtype)
                state = RSSMState(*(torch.where(reset, a, b) for a, b in zip(fresh, state)))
            action = zero_action if t == 0 else actions[:, t - 1]
            state, prior = self.observe_step(state, action, observation_embeddings[:, t], sample=sample)
            posterior_states.append(state)
            prior_states.append(prior)
        return RSSMSequence(self._stack(posterior_states), self._stack(prior_states))

    def imagine(self, initial_state: RSSMState, actions: Tensor, *, sample: bool = True) -> RSSMState:
        """Free-run a sequence of actions without conditioning on observations."""
        self._validate_state(initial_state)
        if not isinstance(actions, Tensor) or actions.ndim != 3:
            raise ValueError("actions must be [batch,horizon,action_dim]")
        if actions.shape[0] != initial_state.deterministic.shape[0] or actions.shape[2] != self.action_dim:
            raise ValueError("actions do not match initial state or action_dim")
        if actions.shape[1] < 1 or not torch.isfinite(actions).all():
            raise ValueError("actions must have a non-empty finite horizon")
        current, states = initial_state, []
        for t in range(actions.shape[1]):
            current = self.imagine_step(current, actions[:, t], sample=sample)
            states.append(current)
        return self._stack(states)
