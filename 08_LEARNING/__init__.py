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
    "ReplayBuffer", "LearningIntegration", "from_memory_record",
    "from_cognitive_evidence", "from_cognitive_result", "from_memory_search_results", "from_episode_window",
]
