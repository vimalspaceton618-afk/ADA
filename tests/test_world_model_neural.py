import pytest

torch = pytest.importorskip("torch")
from torch import nn

from ada.world_model.environment_model import RecurrentStateSpaceModel, RSSMWorldModel
from ada.world_model.predictive_engine import ActionConditionedJEPA, EMATargetEncoder, CEMPlanner
from ada.world_model.simulation_engine import compare_rollouts


def test_rssm_observe_and_imagine_shapes():
    model = RecurrentStateSpaceModel(8, 3, deterministic_dim=16, stochastic_groups=4,
                                     classes=5, hidden_dim=16)
    observations = torch.randn(2, 6, 8)
    actions = torch.randn(2, 5, 3)
    sequence = model.observe_sequence(observations, actions, sample=False)
    assert sequence.posterior.deterministic.shape == (2, 6, 16)
    assert sequence.posterior.stochastic.shape == (2, 6, 4, 5)
    imagined = model.imagine(sequence.posterior.features[:, -1] if False else
                             type(sequence.posterior)(
                                 sequence.posterior.deterministic[:, -1],
                                 sequence.posterior.stochastic[:, -1],
                                 sequence.posterior.logits[:, -1],
                             ), torch.randn(2, 4, 3), sample=False)
    assert imagined.deterministic.shape == (2, 4, 16)


def test_rssm_episode_reset_blocks_previous_action_history():
    model = RecurrentStateSpaceModel(6, 2, deterministic_dim=12, stochastic_groups=3,
                                     classes=4, hidden_dim=12).eval()
    obs = torch.randn(2, 3, 6)
    actions_a = torch.randn(2, 2, 2)
    actions_b = actions_a.clone()
    actions_b[:, 0] += 5.0
    is_first = torch.zeros(2, 3, dtype=torch.bool)
    is_first[:, 1] = True
    result_a = model.observe_sequence(obs, actions_a, is_first=is_first, sample=False)
    result_b = model.observe_sequence(obs, actions_b, is_first=is_first, sample=False)
    assert torch.allclose(result_a.posterior.deterministic[:, 1], result_b.posterior.deterministic[:, 1])


def test_rssm_world_model_loss_backpropagates_and_imagines():
    model = RSSMWorldModel(10, 3, embedding_dim=8, deterministic_dim=16,
                           stochastic_groups=4, classes=4, hidden_dim=16, free_nats=0.1)
    obs = torch.randn(2, 6, 10)
    actions = torch.randn(2, 5, 3)
    rewards = torch.randn(2, 5)
    terminated = torch.zeros(2, 5, dtype=torch.bool)
    output = model(obs, actions, sample=True)
    losses = model.loss(output, rewards=rewards, terminated=terminated)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    posterior = output["posterior"]
    initial = type(posterior)(
        posterior.deterministic[:, -1], posterior.stochastic[:, -1], posterior.logits[:, -1]
    )
    initial_copy = initial.deterministic.clone()
    rollout = model.imagine(initial, torch.randn(2, 4, 3))
    assert rollout["latent_features"].shape == (2, 4, 32)
    assert torch.equal(initial.deterministic, initial_copy)


def test_jepa_predictor_is_causal_and_target_is_detached():
    model = ActionConditionedJEPA(12, 3, hidden_dim=24, num_layers=2, num_heads=4).eval()
    context = torch.randn(2, 5, 12)
    actions = torch.randn(2, 5, 3)
    baseline = model(context, actions)
    changed = context.clone()
    changed[:, 3:] += 100.0
    future_changed = model(changed, actions)
    assert torch.allclose(baseline[:, :3], future_changed[:, :3], atol=1e-5)
    target = torch.randn_like(baseline, requires_grad=True)
    loss = model.loss(baseline, target)["loss"]
    loss.backward()
    assert target.grad is None


def test_ema_target_encoder_tracks_online_parameters():
    online = nn.Linear(4, 3)
    target = EMATargetEncoder(online, momentum=0.5)
    before = next(target.encoder.parameters()).clone()
    with torch.no_grad():
        next(online.parameters()).add_(2.0)
    target.update(online)
    after = next(target.encoder.parameters())
    assert torch.allclose(after, before + 1.0)


def test_cem_planner_respects_action_bounds():
    planner = CEMPlanner(horizon=3, samples=64, elite_count=8, iterations=3)
    initial = torch.zeros(1, 2)
    transition = lambda state, action: state + action
    reward = lambda state, action: -((state - 0.5) ** 2).sum(dim=-1)
    result = planner.plan(initial, transition, reward,
                          action_low=[-1.0, -1.0], action_high=[1.0, 1.0])
    assert result.feasible
    assert result.first_action is not None
    assert torch.all(result.action_sequence <= 1.0)
    assert torch.all(result.action_sequence >= -1.0)


def test_cem_infeasible_cost_returns_no_action():
    planner = CEMPlanner(horizon=2, samples=24, elite_count=4, iterations=2)
    result = planner.plan(
        torch.zeros(1, 2), lambda s, a: s + a, lambda s, a: torch.ones(s.shape[0]),
        action_low=[-1.0, -1.0], action_high=[1.0, 1.0],
        cost_fn=lambda s, a: torch.ones(s.shape[0]), max_cost=0.0,
    )
    assert not result.feasible
    assert result.first_action is None


def test_compare_rollouts_branches_one_state_into_candidates():
    model = RSSMWorldModel(5, 2, embedding_dim=6, deterministic_dim=8,
                           stochastic_groups=2, classes=3, hidden_dim=8)
    state = model.rssm.initial_state(1)
    candidates = torch.randn(3, 4, 2)
    results = compare_rollouts(model, state, candidates)
    assert len(results) == 3
    assert all(item.latent_features.shape == (1, 4, 14) for item in results)
