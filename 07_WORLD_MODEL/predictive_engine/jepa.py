"""Action-conditioned latent predictor inspired by V-JEPA 2-AC.

This vector/temporal predictor is a small research adapter, not a reproduction
of Meta's patch-level causal Vision Transformer or training data pipeline.
"""
from __future__ import annotations
import copy
try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("ActionConditionedJEPA requires PyTorch; install ADA with [neural].") from exc


@torch.no_grad()
def update_ema(target: nn.Module, online: nn.Module, momentum: float = 0.996) -> None:
    """Update a target encoder using exponential moving average parameters."""
    if not isinstance(target, nn.Module) or not isinstance(online, nn.Module):
        raise TypeError("target and online must be torch modules")
    if isinstance(momentum, bool) or not isinstance(momentum, (int, float)) or not 0 <= momentum < 1:
        raise ValueError("momentum must be in [0,1)")
    target_params = dict(target.named_parameters())
    online_params = dict(online.named_parameters())
    if target_params.keys() != online_params.keys():
        raise ValueError("target and online parameter names must match")
    for name, target_param in target_params.items():
        source = online_params[name]
        if target_param.shape != source.shape:
            raise ValueError(f"parameter shape mismatch for {name}")
        target_param.lerp_(source.detach(), 1.0 - float(momentum))
    target_buffers = dict(target.named_buffers())
    online_buffers = dict(online.named_buffers())
    if target_buffers.keys() != online_buffers.keys():
        raise ValueError("target and online buffer names must match")
    for name, target_buffer in target_buffers.items():
        source = online_buffers[name]
        if target_buffer.shape != source.shape:
            raise ValueError(f"buffer shape mismatch for {name}")
        if torch.is_floating_point(target_buffer):
            target_buffer.lerp_(source.detach(), 1.0 - float(momentum))
        else:
            target_buffer.copy_(source)


class EMATargetEncoder(nn.Module):
    """Frozen deep-copied target encoder with an explicit EMA update method."""
    def __init__(self, online_encoder: nn.Module, *, momentum: float = 0.996):
        super().__init__()
        if not isinstance(online_encoder, nn.Module):
            raise TypeError("online_encoder must be a torch module")
        self.encoder = copy.deepcopy(online_encoder)
        self.momentum = float(momentum)
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0,1)")
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        update_ema(self.encoder, online_encoder, self.momentum)
        self.encoder.eval()

    @torch.no_grad()
    def forward(self, observations: Tensor) -> Tensor:
        self.encoder.eval()
        return self.encoder(observations)


class ActionConditionedJEPA(nn.Module):
    """Predict the next frame/state embedding from context and aligned actions.

    context_latents: [B,T,D], actions: [B,T,A], optional state: [B,T,S].
    The causal transformer ensures prediction at t cannot read later context.
    The training target is expected to be a detached or EMA target embedding.
    """
    def __init__(self, latent_dim: int, action_dim: int, *,
                 state_dim: int = 0, hidden_dim: int = 256,
                 num_layers: int = 4, num_heads: int = 4,
                 max_sequence_length: int = 256, dropout: float = 0.0):
        super().__init__()
        for name, value in (("latent_dim", latent_dim), ("action_dim", action_dim),
                            ("hidden_dim", hidden_dim), ("num_layers", num_layers),
                            ("num_heads", num_heads), ("max_sequence_length", max_sequence_length)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(state_dim) is not int or state_dim < 0:
            raise ValueError("state_dim must be a non-negative integer")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.latent_dim, self.action_dim = latent_dim, action_dim
        self.state_dim, self.max_sequence_length = state_dim, max_sequence_length
        input_dim = latent_dim + action_dim + state_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=4 * hidden_dim,
            dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                       enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.predictor = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context_latents: Tensor, actions: Tensor,
                state: Tensor | None = None) -> Tensor:
        if not isinstance(context_latents, Tensor) or context_latents.ndim != 3:
            raise ValueError("context_latents must be [batch,time,latent_dim]")
        batch, length, width = context_latents.shape
        if batch < 1 or length < 1 or length > self.max_sequence_length or width != self.latent_dim:
            raise ValueError("context latent dimensions exceed the configured contract")
        if not isinstance(actions, Tensor) or actions.shape != (batch, length, self.action_dim):
            raise ValueError("actions must be [batch,time,action_dim]")
        parts = [context_latents, actions]
        if self.state_dim:
            if not isinstance(state, Tensor) or state.shape != (batch, length, self.state_dim):
                raise ValueError("state must be [batch,time,state_dim] when state_dim is configured")
            parts.append(state)
        elif state is not None:
            raise ValueError("state was supplied but state_dim is zero")
        if context_latents.device != actions.device or any(x.device != context_latents.device for x in parts):
            raise ValueError("all predictor inputs must share a device")
        if any(not torch.isfinite(x).all() for x in parts):
            raise ValueError("predictor inputs must be finite")
        hidden = self.input_projection(torch.cat(parts, dim=-1))
        hidden = hidden + self.position_embedding[:, :length]
        causal_mask = torch.triu(
            torch.ones(length, length, device=hidden.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.temporal_encoder(hidden, mask=causal_mask)
        return self.predictor(self.output_norm(hidden))

    def loss(self, predictions: Tensor, target_latents: Tensor, *,
             mask: Tensor | None = None, variance_floor: float = 0.0) -> dict[str, Tensor]:
        """JEPA latent regression; target is detached to avoid target-side gradients."""
        if predictions.shape != target_latents.shape or predictions.ndim != 3:
            raise ValueError("predictions and target_latents must share [B,T,D] shape")
        if not torch.isfinite(predictions).all() or not torch.isfinite(target_latents).all():
            raise ValueError("JEPA latents must be finite")
        target = target_latents.detach()
        per_token = 1.0 - F.cosine_similarity(predictions, target, dim=-1)
        regression = F.smooth_l1_loss(predictions, target, reduction="none").mean(-1)
        valid = torch.ones_like(per_token, dtype=torch.bool)
        if mask is not None:
            if mask.shape != valid.shape or mask.dtype != torch.bool:
                raise ValueError("mask must be boolean [batch,time]")
            valid = valid & mask
        if not valid.any():
            raise ValueError("mask excludes every latent prediction")
        prediction_loss = (per_token + regression)[valid].mean()
        variance_loss = predictions.new_zeros(())
        if variance_floor:
            if variance_floor <= 0:
                raise ValueError("variance_floor must be positive")
            samples = target[valid]
            if samples.shape[0] > 1:
                std = torch.sqrt(samples.var(dim=0, unbiased=False) + 1e-4)
                variance_loss = F.relu(float(variance_floor) - std).mean()
        return {
            "loss": prediction_loss + variance_loss,
            "prediction_loss": prediction_loss,
            "variance_floor_loss": variance_loss,
        }
