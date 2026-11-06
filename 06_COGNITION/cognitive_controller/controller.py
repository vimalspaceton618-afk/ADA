"""Coordinate the bounded cognition pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class CognitiveStage:
    name: str
    output: Any

class CognitiveController:
    """Run registered stages in order; preserve outputs and fail visibly."""
    def __init__(self, stages=()):
        self._stages = tuple(stages)
        if any(not callable(stage) for stage in self._stages):
            raise TypeError("every cognitive stage must be callable")

    def run(self, context: Any) -> tuple[CognitiveStage, ...]:
        current = context
        results = []
        for stage in self._stages:
            value = stage(current)
            name = getattr(stage, "__name__", type(stage).__name__)
            results.append(CognitiveStage(name, value))
            current = value
        return tuple(results)
