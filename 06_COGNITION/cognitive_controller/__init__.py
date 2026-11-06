"""Coordinate cognition stages in an explicit, inspectable sequence."""
from dataclasses import dataclass
@dataclass(frozen=True, slots=True)
class CognitiveStage:
    name: str
    output: object
class CognitiveController:
    def __init__(self, stages=()):
        self.stages=tuple(stages)
        if any(not callable(s) for s in self.stages): raise TypeError("stages must be callable")
    def run(self, context):
        output=context; trace=[]
        for stage in self.stages:
            output=stage(output); trace.append(CognitiveStage(getattr(stage,"__name__",type(stage).__name__),output))
        return tuple(trace)
