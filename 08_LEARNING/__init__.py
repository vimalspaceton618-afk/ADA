"""ADA Learning: validated experience, replay and gated adaptation."""
from .contracts import (
    SCHEMA_VERSION, ArtifactStatus, ExperienceKind, ExperienceRecord,
    ExperienceStatus, LearningArtifact, LearningDecision, LearningProposal,
    MetricSet, OutcomeKind,
)
from .experience_learning.ledger import ExperienceLedger
from .learning_controller.controller import LearningAttempt, LearningController
from .learning_evaluation.gates import GateResult, MetricPolicy, RegressionGate
from .integration import LearningIntegration, from_cognitive_evidence, from_cognitive_result, from_episode_window, from_memory_record, from_memory_search_results
from .replay_mechanism.buffer import ReplayBuffer

__all__ = [
    "SCHEMA_VERSION", "ArtifactStatus", "ExperienceKind", "ExperienceRecord",
    "ExperienceStatus", "LearningArtifact", "LearningDecision", "LearningProposal",
    "MetricSet", "OutcomeKind", "ExperienceLedger", "LearningAttempt",
    "LearningController", "GateResult", "MetricPolicy", "RegressionGate",
    "PyTorchLearningAdapter", "TorchBatch", "TrainingStepResult", "train_replay_step",
    "ReplayBuffer", "LearningIntegration", "from_memory_record",
    "from_cognitive_evidence", "from_cognitive_result", "from_memory_search_results", "from_episode_window",
]


def __getattr__(name: str):
    """Load the optional PyTorch adapter only when a caller requests it."""
    if name in {
        "PyTorchLearningAdapter", "TorchBatch",
        "TrainingStepResult", "train_replay_step",
    }:
        from . import pytorch_adapter
        return getattr(pytorch_adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
