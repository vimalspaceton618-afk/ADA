"""ADA's symbolic and learned world-model interfaces.

The default import path remains dependency-light. Learned PyTorch modules are
resolved lazily only when explicitly requested.
"""
from importlib import import_module

from .contracts import (
    SCHEMA_VERSION, FactCondition, FactEffect, PredictionStatus, SimulationResult,
    TransitionRule, WorldAction, WorldFact, WorldPrediction, WorldSnapshot,
)
from .engine import WorldModelEngine

__all__ = [
    "SCHEMA_VERSION", "FactCondition", "FactEffect", "PredictionStatus",
    "SimulationResult", "TransitionRule", "WorldAction", "WorldFact",
    "WorldPrediction", "WorldSnapshot", "WorldModelEngine",
]

_LAZY = {
    "RSSMState": "environment_model.rssm",
    "RSSMSequence": "environment_model.rssm",
    "RecurrentStateSpaceModel": "environment_model.rssm",
    "RSSMWorldModel": "environment_model.model",
    "ActionConditionedJEPA": "predictive_engine.jepa",
    "EMATargetEncoder": "predictive_engine.jepa",
    "update_ema": "predictive_engine.jepa",
    "CEMPlanner": "predictive_engine.cem",
    "CEMPlan": "predictive_engine.cem",
    "LatentRollout": "simulation_engine.rollouts",
    "imagine_rollout": "simulation_engine.rollouts",
    "compare_rollouts": "simulation_engine.rollouts",
    "CausalGraph": "causal_model.graph",
    "EpisodeTrajectory": "temporal_model.trajectory",
    "EpisodeWindow": "temporal_model.trajectory",
    "mean_squared_error": "model_validation.metrics",
    "mean_absolute_error": "model_validation.metrics",
    "expected_calibration_error": "model_validation.metrics",
    "horizon_mse": "model_validation.metrics",
    "PhysicalState": "physical_model.contracts",
    "ObservedBehaviorEvent": "human_behavior_model.contracts",
    "EmpiricalActionPrior": "human_behavior_model.baseline",
    "ModelRuntimeState": "internal_state.contracts",
}

def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(name)
    module = import_module(f"07_WORLD_MODEL.{module_path}")
    value = getattr(module, name)
    globals()[name] = value
    return value

def __dir__():
    return sorted(set(globals()) | set(_LAZY))
