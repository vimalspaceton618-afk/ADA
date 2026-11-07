"""Compact trainable RSSM world model over dense observation features."""
from __future__ import annotations
try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("RSSMWorldModel requires PyTorch. Install ADA with the [neural] extra.") from exc
from .rssm import RSSMState, RecurrentStateSpaceModel


class RSSMWorldModel(nn.Module):
    """Encode feature observations, infer latent state, and predict transitions.

    This model consumes fixed-width vectors (for example pooled visual features
    or structured numeric states), not raw images or videos.
    """
    def __init__(self, observation_dim: int, action_dim: int, *,
                 embedding_dim: int = 128, deterministic_dim: int = 256,
                 stochastic_groups: int = 16, classes: int = 16, hidden_dim: int = 256,
                 kl_balance: float = 0.8, free_nats: float = 1.0,
                 kl_scale: float = 1.0, observation_scale: float = 1.0,
                 reward_scale: float = 1.0, continuation_scale: float = 1.0):
        super().__init__()
        for name, value in (("observation_dim", observation_dim), ("action_dim", action_dim),
                            ("embedding_dim", embedding_dim), ("deterministic_dim", deterministic_dim),
                            ("stochastic_groups", stochastic_groups), ("classes", classes),
                            ("hidden_dim", hidden_dim)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        scales = (kl_balance, free_nats, kl_scale, observation_scale, reward_scale, continuation_scale)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not torch.isfinite(torch.tensor(float(x))) for x in scales):
            raise ValueError("loss parameters must be finite numeric values")
        if not 0 <= kl_balance <= 1 or free_nats < 0 or min(kl_scale, observation_scale, reward_scale, continuation_scale) < 0:
            raise ValueError("invalid KL balance, free nats or loss scale")
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.kl_balance, self.free_nats = float(kl_balance), float(free_nats)
        self.kl_scale, self.observation_scale = float(kl_scale), float(observation_scale)
        self.reward_scale, self.continuation_scale = float(reward_scale), float(continuation_scale)
        self.encoder = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                      nn.SiLU(), nn.Linear(hidden_dim, embedding_dim))
        self.rssm = RecurrentStateSpaceModel(
            embedding_dim, action_dim, deterministic_dim=deterministic_dim,
            stochastic_groups=stochastic_groups, classes=classes, hidden_dim=hidden_dim)
        feature_dim = deterministic_dim + stochastic_groups * classes
        self.observation_decoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
                                                  nn.Linear(hidden_dim, observation_dim))
        transition_dim = feature_dim + action_dim
        self.reward_head = nn.Sequential(nn.Linear(transition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.continuation_head = nn.Sequential(nn.Linear(transition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, observations: Tensor, actions: Tensor, *,
                is_first: Tensor | None = None, sample: bool = True) -> dict:
        """Predict next observations/rewards from [B,T,O] and [B,T-1,A]."""
        if not isinstance(observations, Tensor) or observations.ndim != 3:
            raise ValueError("observations must be [batch,time,observation_dim]")
        batch, length, dim = observations.shape
        if batch < 1 or length < 2 or dim != self.observation_dim:
            raise ValueError("observations must have shape [B,T>=2,observation_dim]")
        if not isinstance(actions, Tensor) or actions.shape != (batch, length - 1, self.action_dim):
            raise ValueError("actions must have shape [batch,time-1,action_dim]")
        if observations.device != actions.device:
            raise ValueError("observations and actions must share a device")
        if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
            raise ValueError("observations and actions must be finite")
        embeddings = self.encoder(observations)
        sequence = self.rssm.observe_sequence(embeddings, actions, is_first=is_first, sample=sample)
        post_features, prior_features = sequence.posterior.features, sequence.prior.features
        transition_features = torch.cat((post_features[:, :-1], actions), dim=-1)
        return {
            "posterior": sequence.posterior,
            "prior": sequence.prior,
            "observation_predictions": self.observation_decoder(prior_features[:, 1:]),
            "reward_predictions": self.reward_head(transition_features).squeeze(-1),
            "continuation_logits": self.continuation_head(transition_features).squeeze(-1),
            "observation_targets": observations[:, 1:],
            "posterior_features": post_features,
            "prior_features": prior_features,
        }

    def loss(self, output: dict, *, rewards: Tensor | None = None,
             terminated: Tensor | None = None, transition_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Balanced KL + prior feature prediction + optional reward/continuation losses."""
        posterior, prior = output.get("posterior"), output.get("prior")
        if not isinstance(posterior, RSSMState) or not isinstance(prior, RSSMState):
            raise TypeError("output must be produced by RSSMWorldModel.forward")
        q_logits, p_logits = posterior.logits[:, 1:].float(), prior.logits[:, 1:].float()
        q_log, p_log = F.log_softmax(q_logits, -1), F.log_softmax(p_logits, -1)
        q_prob = q_log.exp()
        dynamics_kl = (q_prob.detach() * (q_log.detach() - p_log)).sum(-1).sum(-1).clamp_min(self.free_nats)
        representation_kl = (q_prob * (q_log - p_log.detach())).sum(-1).sum(-1).clamp_min(self.free_nats)
        balanced_kl = self.kl_balance * dynamics_kl + (1.0 - self.kl_balance) * representation_kl
        prediction, target = output["observation_predictions"], output["observation_targets"]
        observation_error = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
        valid = torch.ones_like(observation_error, dtype=torch.bool)
        if transition_mask is not None:
            if not isinstance(transition_mask, Tensor) or transition_mask.shape != valid.shape or transition_mask.dtype != torch.bool:
                raise ValueError("transition_mask must be boolean [batch,time-1]")
            valid = valid & transition_mask
        if not valid.any():
            raise ValueError("transition_mask excludes every transition")
        mean = lambda x: x[valid].mean()
        kl_loss, observation_loss = mean(balanced_kl), mean(observation_error)
        reward_loss = prediction.new_zeros(())
        continuation_loss = prediction.new_zeros(())
        if rewards is not None:
            if rewards.shape != valid.shape or not torch.isfinite(rewards).all():
                raise ValueError("rewards must be finite with shape [batch,time-1]")
            reward_loss = mean(F.smooth_l1_loss(output["reward_predictions"], rewards, reduction="none"))
        if terminated is not None:
            if terminated.shape != valid.shape or terminated.dtype != torch.bool:
                raise ValueError("terminated must be boolean [batch,time-1]")
            continuation_loss = mean(F.binary_cross_entropy_with_logits(
                output["continuation_logits"], (~terminated).to(output["continuation_logits"].dtype),
                reduction="none"))
        total = (self.kl_scale * kl_loss + self.observation_scale * observation_loss
                 + self.reward_scale * reward_loss + self.continuation_scale * continuation_loss)
        return {
            "loss": total, "kl_loss": kl_loss,
            "dynamics_kl": mean(dynamics_kl), "representation_kl": mean(representation_kl),
            "observation_loss": observation_loss, "reward_loss": reward_loss,
            "continuation_loss": continuation_loss,
        }

    def imagine(self, initial_state: RSSMState, actions: Tensor, *, sample: bool = False) -> dict[str, Tensor]:
        """Free-run latent dynamics and scalar heads without modifying input state."""
        states = self.rssm.imagine(initial_state, actions, sample=sample)
        transition = torch.cat((states.features, actions), dim=-1)
        return {
            "latent_features": states.features,
            "reward_predictions": self.reward_head(transition).squeeze(-1),
            "continuation_logits": self.continuation_head(transition).squeeze(-1),
        }
