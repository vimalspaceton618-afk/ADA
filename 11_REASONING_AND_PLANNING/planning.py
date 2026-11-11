"""Framework-independent deterministic task-graph planning contracts."""
from __future__ import annotations
import hashlib, json, math, threading, uuid
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

def _text(v, n):
    if not isinstance(v, str) or not v.strip(): raise ValueError(f"{n} must be non-empty")
    return v.strip()
def _freeze(v):
    if v is None or type(v) in (str, int, bool): return v
    if isinstance(v, float):
        if not math.isfinite(v): raise ValueError("non-finite value")
        return v
    if isinstance(v, Mapping):
        if any(not isinstance(k, str) for k in v): raise TypeError("mapping keys must be strings")
        return MappingProxyType({k: _freeze(x) for k, x in v.items()})
    if isinstance(v, (list, tuple)): return tuple(_freeze(x) for x in v)
    raise TypeError("value is not JSON-compatible")
def _thaw(v):
    if isinstance(v, Mapping): return {k: _thaw(x) for k, x in v.items()}
    if isinstance(v, tuple): return [_thaw(x) for x in v]
    return v

class PlanStatus(str, Enum):
    DRAFT="draft"; VERIFIED="verified"; REJECTED="rejected"

@dataclass(frozen=True, slots=True)
class PlanStep:
    step_id: str
    description: str
    capability: str
    depends_on: tuple[str, ...] = ()
    inputs: Mapping[str, Any] = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    def __post_init__(self):
        for n in ("step_id", "description", "capability"): _text(getattr(self, n), n)
        deps=tuple(self.depends_on)
        if self.step_id in deps or len(set(deps)) != len(deps): raise ValueError("invalid step dependencies")
        object.__setattr__(self, "depends_on", deps); object.__setattr__(self, "inputs", _freeze(self.inputs))
        object.__setattr__(self, "preconditions", tuple(_text(x, "precondition") for x in self.preconditions))
        object.__setattr__(self, "expected_outputs", tuple(_text(x, "expected_output") for x in self.expected_outputs))

@dataclass(frozen=True, slots=True)
class Plan:
    objective: str
    steps: tuple[PlanStep, ...]
    max_steps: int
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: PlanStatus = PlanStatus.DRAFT
    def __post_init__(self):
        _text(self.objective, "objective"); _text(self.plan_id, "plan_id")
        if type(self.max_steps) is not int or self.max_steps < 1: raise ValueError("max_steps must be positive integer")
        steps=tuple(self.steps)
        if not steps or len(steps)>self.max_steps: raise ValueError("steps must be non-empty and within max_steps")
        if any(not isinstance(x, PlanStep) for x in steps): raise TypeError("steps must be PlanStep")
        if len({x.step_id for x in steps}) != len(steps): raise ValueError("step ids must be unique")
        object.__setattr__(self, "steps", steps)
        if not isinstance(self.status, PlanStatus): raise TypeError("status must be PlanStatus")
    @property
    def digest(self):
        body={"objective":self.objective,"max_steps":self.max_steps,
              "steps":[{"id":s.step_id,"description":s.description,"capability":s.capability,
                        "depends_on":s.depends_on,"inputs":_thaw(s.inputs),
                        "preconditions":s.preconditions,"expected_outputs":s.expected_outputs} for s in self.steps]}
        return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest()

@dataclass(frozen=True, slots=True)
class PlanVerification:
    accepted: bool
    reason: str
    order: tuple[str, ...]
    unresolved_capabilities: tuple[str, ...] = ()

class PlanVerifier:
    def verify(self, plan: Plan, available_capabilities: Iterable[str]) -> PlanVerification:
        if not isinstance(plan, Plan): raise TypeError("plan must be Plan")
        available=set(available_capabilities); by_id={s.step_id:s for s in plan.steps}
        missing={d for s in plan.steps for d in s.depends_on if d not in by_id}
        if missing: return PlanVerification(False, f"unknown dependencies: {sorted(missing)}", ())
        unresolved=tuple(sorted({s.capability for s in plan.steps if s.capability not in available}))
        if unresolved: return PlanVerification(False, "required capabilities unavailable", (), unresolved)
        order=[]; pending=set(by_id)
        while pending:
            ready=sorted(sid for sid in pending if set(by_id[sid].depends_on).issubset(order))
            if not ready: return PlanVerification(False, "dependency cycle detected", ())
            order.extend(ready); pending.difference_update(ready)
        return PlanVerification(True, "plan graph is structurally valid", tuple(order))

class GoalManager:
    """Tracks explicit caller-supplied objectives; never invents intent."""
    def __init__(self): self._goals={}; self._lock=threading.RLock()
    def add(self, objective: str, *, goal_id: str|None=None):
        objective=_text(objective,"objective"); gid=goal_id or str(uuid.uuid4())
        with self._lock:
            if gid in self._goals: raise ValueError("goal id already exists")
            self._goals[gid]=objective
        return gid
    def get(self, goal_id):
        with self._lock: return self._goals[goal_id]

class DecisionEngine:
    """Selects from explicit scored alternatives with deterministic tie-break and abstention."""
    def choose(self, scores: Mapping[str,float], *, min_score: float=0.0):
        if isinstance(min_score,bool) or not isinstance(min_score,(int,float)) or not math.isfinite(float(min_score)):
            raise ValueError("min_score must be finite")
        if not isinstance(scores,Mapping) or not scores: return None
        for name,value in scores.items():
            _text(name,"candidate")
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)):
                raise ValueError("scores must be finite numeric values")
        best=max(scores,key=lambda name:(float(scores[name]),name))
        return best if scores[best]>=min_score else None
