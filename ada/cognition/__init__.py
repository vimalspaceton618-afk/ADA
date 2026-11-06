"""Stable public import facade for ADA Cognition."""
from importlib import import_module

_module = import_module("06_COGNITION")
SCHEMA_VERSION = _module.SCHEMA_VERSION
CognitiveDecision = _module.CognitiveDecision
CognitiveResult = _module.CognitiveResult
CognitiveStatus = _module.CognitiveStatus
CognitiveTask = _module.CognitiveTask
DecisionStatus = _module.DecisionStatus
EvidenceItem = _module.EvidenceItem
PlanStep = _module.PlanStep
CognitionEngine = _module.CognitionEngine
CognitionAnalysis = _module.CognitionAnalysis
analyze_cognition = _module.analyze_cognition

# Expose the eleven cognition subsystems through the stable ada.cognition facade.
_SUBSYSTEMS = (
    "cognitive_controller", "attention_mechanism", "abstraction_engine",
    "concept_formation", "relationship_engine", "causal_reasoning",
    "logical_reasoning", "mathematical_reasoning", "hypothesis_generation",
    "uncertainty_estimation", "metacognition",
)
for _name in _SUBSYSTEMS:
    globals()[_name] = import_module(f"06_COGNITION.{_name}")

__all__ = [
    "SCHEMA_VERSION", "CognitiveDecision", "CognitiveResult",
    "CognitiveStatus", "CognitiveTask", "DecisionStatus", "EvidenceItem",
    "PlanStep", "CognitionEngine", "CognitionAnalysis", "analyze_cognition", *_SUBSYSTEMS,
]
