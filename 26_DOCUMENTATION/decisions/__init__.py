"""Architecture decision lifecycle and supersession tracking."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class DecisionState(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"

@dataclass(frozen=True, slots=True)
class Decision:
    decision_id: str
    title: str
    context: str
    chosen_option: str
    alternatives: tuple[str, ...]
    consequences: str
    evidence: tuple[str, ...]
    state: DecisionState = DecisionState.PROPOSED
    supersedes: str | None = None
    def __post_init__(self):
        if any(not x.strip() for x in (self.decision_id, self.title, self.context, self.chosen_option, self.consequences)):
            raise ValueError("decision content required")
        if self.state is DecisionState.ACCEPTED and not self.evidence: raise ValueError("accepted decision needs evidence")

class DecisionLog:
    def __init__(self): self._items = {}
    def add(self, decision: Decision):
        if not isinstance(decision, Decision): raise TypeError("Decision required")
        if decision.decision_id in self._items: raise ValueError("decision ID already exists")
        if decision.supersedes and decision.supersedes not in self._items: raise KeyError(decision.supersedes)
        self._items[decision.decision_id] = decision
        return decision
    def get(self, decision_id): return self._items[decision_id]
    @property
    def items(self): return tuple(self._items[k] for k in sorted(self._items))

__all__ = ["DecisionState", "Decision", "DecisionLog"]
