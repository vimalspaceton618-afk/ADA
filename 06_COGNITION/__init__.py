"""ADA cognition layer: explicit task orchestration and auditable decisions."""
from .contracts import (
    SCHEMA_VERSION, CognitiveDecision, CognitiveResult, CognitiveStatus,
    CognitiveTask, DecisionStatus, EvidenceItem, PlanStep,
)
from .engine import CognitionEngine

__all__ = [
    "SCHEMA_VERSION", "CognitiveDecision", "CognitiveResult",
    "CognitiveStatus", "CognitiveTask", "DecisionStatus", "EvidenceItem",
    "PlanStep", "CognitionEngine",
]
