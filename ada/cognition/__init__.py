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

__all__ = [
    "SCHEMA_VERSION", "CognitiveDecision", "CognitiveResult",
    "CognitiveStatus", "CognitiveTask", "DecisionStatus", "EvidenceItem",
    "PlanStep", "CognitionEngine",
]
