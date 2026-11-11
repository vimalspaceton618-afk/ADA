"""Bounded continual-training schedules with replay and regression gates."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ContinualStage:
    stage_id: str
    dataset_digest: str
    max_steps: int
    replay_fraction: float
    require_regression_pass: bool = True
    def __post_init__(self):
        if not self.stage_id.strip() or len(self.dataset_digest) != 64: raise ValueError("stage ID and dataset digest required")
        if type(self.max_steps) is not int or self.max_steps < 1: raise ValueError("max_steps must be positive")
        if not 0 <= self.replay_fraction <= 1: raise ValueError("replay_fraction must be [0,1]")

class ContinualSchedule:
    def __init__(self, stages):
        self.stages = tuple(stages)
        if not self.stages or any(not isinstance(x, ContinualStage) for x in self.stages):
            raise ValueError("schedule requires ContinualStage values")
        if len({x.stage_id for x in self.stages}) != len(self.stages): raise ValueError("stage IDs must be unique")
    def stage(self, index: int) -> ContinualStage:
        return self.stages[index]

__all__ = ["ContinualStage", "ContinualSchedule"]
