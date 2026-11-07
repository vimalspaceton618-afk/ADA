"""Cross-Entropy Method planner over caller-supplied latent dynamics."""
from __future__ import annotations
from dataclasses import dataclass
try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("CEMPlanner requires PyTorch; install ADA with [neural].") from exc


@dataclass(frozen=True)
class CEMPlan:
    """A proposed sequence only; it is never sent to an actuator."""
    first_action: Tensor | None
    action_sequence: Tensor
    score: float
    estimated_cost: float
    feasible: bool
    iterations: int


class CEMPlanner:
    """Sample bounded action sequences and refine a Gaussian elite distribution.

    The transition/reward/cost callables are supplied by the model owner. The
    planner is an optimizer, not a world model and not a safety certificate.
    """
    def __init__(self, *, horizon: int = 12, samples: int = 256,
                 elite_count: int = 32, iterations: int = 5,
                 discount: float = 0.99, momentum: float = 0.1,
                 min_std: float = 0.03):
        for name, value in (("horizon", horizon), ("samples", samples),
                            ("elite_count", elite_count), ("iterations", iterations)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if elite_count > samples:
            raise ValueError("elite_count cannot exceed samples")
        for name, value in (("discount", discount), ("momentum", momentum), ("min_std", min_std)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be finite")
        if not 0 < discount <= 1 or not 0 <= momentum < 1 or min_std <= 0:
            raise ValueError("discount, momentum or min_std is outside its valid range")
        self.horizon, self.samples, self.elite_count = horizon, samples, elite_count
        self.iterations, self.discount = iterations, float(discount)
        self.momentum, self.min_std = float(momentum), float(min_std)

    @staticmethod
    def _vector(value, name, device, dtype):
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim != 1 or tensor.numel() < 1 or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be a finite non-empty vector")
        return tensor

    def plan(self, initial_state: Tensor, transition_fn, reward_fn, *,
             action_low, action_high, cost_fn=None, max_cost: float = float("inf"),
             generator: torch.Generator | None = None) -> CEMPlan:
        if not isinstance(initial_state, Tensor) or initial_state.ndim < 1:
            raise ValueError("initial_state must be a tensor with a feature dimension")
        if initial_state.shape[0] != 1:
            raise ValueError("CEM planning expects a single initial state with batch dimension 1")
        if not torch.isfinite(initial_state).all():
            raise ValueError("initial_state must be finite")
        device, dtype = initial_state.device, initial_state.dtype
        low = self._vector(action_low, "action_low", device, dtype)
        high = self._vector(action_high, "action_high", device, dtype)
        if low.shape != high.shape or torch.any(low >= high):
            raise ValueError("action bounds must have matching shapes and low < high")
        if not callable(transition_fn) or not callable(reward_fn):
            raise TypeError("transition_fn and reward_fn must be callable")
        if cost_fn is not None and not callable(cost_fn):
            raise TypeError("cost_fn must be callable or None")
        if isinstance(max_cost, bool) or not isinstance(max_cost, (int, float)) or max_cost < 0:
            raise ValueError("max_cost must be non-negative")
        action_dim = low.numel()
        mean = (low + high).view(1, 1, action_dim).expand(self.horizon, 1, action_dim).squeeze(1).clone()
        std = ((high - low) * 0.5).view(1, action_dim).expand(self.horizon, action_dim).clone()
        best_sequence, best_score, best_cost = None, float("-inf"), float("inf")
        any_feasible = False

        with torch.no_grad():
            for _ in range(self.iterations):
                noise = torch.randn(
                    self.samples, self.horizon, action_dim, device=device,
                    dtype=dtype, generator=generator,
                )
                candidates = mean.unsqueeze(0) + std.unsqueeze(0) * noise
                candidates = torch.maximum(torch.minimum(candidates, high), low)
                state = initial_state.expand(self.samples, *initial_state.shape[1:]).clone()
                scores = torch.zeros(self.samples, device=device, dtype=dtype)
                costs = torch.zeros_like(scores)
                discount = 1.0
                for step in range(self.horizon):
                    action = candidates[:, step]
                    state = transition_fn(state, action)
                    reward = reward_fn(state, action).reshape(self.samples)
                    if not torch.isfinite(reward).all() or not torch.isfinite(state).all():
                        raise ValueError("transition/reward model returned non-finite values")
                    scores += discount * reward
                    if cost_fn is not None:
                        cost = cost_fn(state, action).reshape(self.samples)
                        if not torch.isfinite(cost).all() or torch.any(cost < 0):
                            raise ValueError("cost_fn must return finite non-negative costs")
                        costs += discount * cost
                    discount *= self.discount
                feasible = costs <= float(max_cost)
                if feasible.any():
                    any_feasible = True
                    feasible_indices = torch.where(feasible)[0]
                    count = min(self.elite_count, feasible_indices.numel())
                    elite_indices = feasible_indices[torch.topk(scores[feasible], count).indices]
                    elite_actions = candidates[elite_indices]
                    elite_mean = elite_actions.mean(0)
                    elite_std = elite_actions.std(0, unbiased=False).clamp_min(self.min_std)
                    mean = self.momentum * mean + (1 - self.momentum) * elite_mean
                    std = self.momentum * std + (1 - self.momentum) * elite_std
                    top_index = elite_indices[torch.argmax(scores[elite_indices])]
                    if float(scores[top_index]) > best_score:
                        best_score = float(scores[top_index])
                        best_cost = float(costs[top_index])
                        best_sequence = candidates[top_index].clone()

        if not any_feasible or best_sequence is None:
            return CEMPlan(None, mean, float("-inf"), float("inf"), False, self.iterations)
        return CEMPlan(best_sequence[0].clone(), best_sequence, best_score,
                       best_cost, True, self.iterations)
