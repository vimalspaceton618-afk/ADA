"""Controlled architecture development and candidate lifecycle for ADA."""
from __future__ import annotations
import hashlib, json, math, re, threading, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

def _text(v: str, n: str) -> str:
    if not isinstance(v, str) or not v.strip(): raise ValueError(f"{n} must be non-empty")
    return v.strip()

def _digest(v: str, n: str) -> str:
    if not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None: raise ValueError(f"{n} must be lowercase SHA-256")
    return v

def _freeze(v: Any) -> Any:
    if v is None or type(v) in (str, bool, int): return v
    if isinstance(v, float):
        if not math.isfinite(v): raise ValueError("non-finite number")
        return v
    if isinstance(v, Mapping):
        if any(not isinstance(k, str) for k in v): raise TypeError("mapping keys must be strings")
        return MappingProxyType({k: _freeze(x) for k, x in v.items()})
    if isinstance(v, (list, tuple)): return tuple(_freeze(x) for x in v)
    raise TypeError(f"unsupported declarative value: {type(v).__name__}")

def canonical_digest(v: Any) -> str:
    def thaw(x):
        if isinstance(x, Mapping): return {k: thaw(x[k]) for k in sorted(x)}
        if isinstance(x, tuple): return [thaw(y) for y in x]
        return x
    raw = json.dumps(thaw(_freeze(v)), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()

class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"

@dataclass(frozen=True, slots=True)
class Capability:
    capability_id: str
    version: str
    description: str
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]
    limitations: tuple[str, ...] = ()
    def __post_init__(self):
        for n in ("capability_id", "version", "description"): _text(getattr(self, n), n)
        if not isinstance(self.inputs, Mapping) or not isinstance(self.outputs, Mapping): raise TypeError("inputs/outputs must be mappings")
        object.__setattr__(self, "inputs", _freeze(self.inputs))
        object.__setattr__(self, "outputs", _freeze(self.outputs))
        object.__setattr__(self, "limitations", tuple(_text(x, "limitation") for x in self.limitations))

@dataclass(frozen=True, slots=True)
class ModuleDescriptor:
    module_id: str
    version: str
    artifact_digest: str
    dependencies: tuple[str, ...] = ()
    capabilities: tuple[Capability, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        _text(self.module_id, "module_id"); _text(self.version, "version"); _digest(self.artifact_digest, "artifact_digest")
        deps = tuple(_text(x, "dependency") for x in self.dependencies)
        if self.module_id in deps or len(set(deps)) != len(deps): raise ValueError("dependencies must be unique and cannot include self")
        object.__setattr__(self, "dependencies", deps)
        caps = tuple(self.capabilities)
        if any(not isinstance(x, Capability) for x in caps): raise TypeError("capabilities must be Capability")
        object.__setattr__(self, "capabilities", caps)
        object.__setattr__(self, "metadata", _freeze(self.metadata))

@dataclass(frozen=True, slots=True)
class ChangeProposal:
    target: str
    parent_digest: str
    operations: tuple[Mapping[str, Any], ...]
    rationale: str
    proposer_id: str
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    def __post_init__(self):
        for n in ("target", "rationale", "proposer_id", "proposal_id"): _text(getattr(self, n), n)
        _digest(self.parent_digest, "parent_digest")
        ops = tuple(self.operations)
        if not ops: raise ValueError("operations cannot be empty")
        frozen = []
        for op in ops:
            if not isinstance(op, Mapping): raise TypeError("operations must be mappings")
            if set(op) - {"op", "path", "value", "expected"}: raise ValueError("unsupported operation fields")
            if op.get("op") not in {"set", "remove", "append_unique"}: raise ValueError("operation must be set/remove/append_unique")
            _text(op.get("path"), "operation.path")
            if op.get("op") != "remove" and "value" not in op: raise ValueError("operation requires value")
            frozen.append(_freeze(op))
        object.__setattr__(self, "operations", tuple(frozen))
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None: raise ValueError("created_at must be timezone-aware")
    @property
    def digest(self):
        return canonical_digest({"target": self.target, "parent_digest": self.parent_digest, "operations": self.operations,
          "rationale": self.rationale, "proposer_id": self.proposer_id, "proposal_id": self.proposal_id, "created_at": self.created_at.isoformat()})

@dataclass(frozen=True, slots=True)
class CandidateRecord:
    proposal: ChangeProposal
    candidate_digest: str
    status: CandidateStatus
    validation_report: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    def __post_init__(self):
        _digest(self.candidate_digest, "candidate_digest")
        if not isinstance(self.status, CandidateStatus): raise TypeError("status must be CandidateStatus")
        object.__setattr__(self, "validation_report", _freeze(self.validation_report))

@dataclass(frozen=True, slots=True)
class PromotionApproval:
    candidate_digest: str
    approver_id: str
    reason: str
    approval_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    approved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    def __post_init__(self):
        _digest(self.candidate_digest, "candidate_digest"); _text(self.approver_id, "approver_id"); _text(self.reason, "reason")

class DevelopmentEngine:
    """Versioned registry and guarded candidate promotion state machine."""
    def __init__(self, *, validator: Callable[[ChangeProposal], Mapping[str, Any]] | None = None):
        self._modules = {}; self._candidates = {}; self._active = {}; self._history = {}
        self._validator = validator or (lambda p: {"status": "not_configured"})
        self._used_approvals = set(); self._lock = threading.RLock()
    @property
    def modules(self):
        with self._lock: return tuple(self._modules[k] for k in sorted(self._modules))
    @property
    def candidates(self):
        with self._lock: return tuple(self._candidates[k] for k in sorted(self._candidates))
    def register_module(self, module):
        if not isinstance(module, ModuleDescriptor): raise TypeError("module must be ModuleDescriptor")
        with self._lock:
            if module.module_id in self._modules: raise ValueError("module already registered")
            missing = set(module.dependencies) - set(self._modules)
            if missing: raise ValueError(f"unknown dependencies: {sorted(missing)}")
            self._modules[module.module_id] = module
            self._active[module.module_id] = module.artifact_digest
            self._history[module.module_id] = [module.artifact_digest]
    def dependency_order(self, module_ids: Iterable[str]):
        wanted = set(module_ids)
        with self._lock:
            missing = wanted - set(self._modules)
            if missing: raise KeyError(f"unknown modules: {sorted(missing)}")
            out = []; visiting = set(); visited = set()
            def visit(mid):
                if mid in visiting: raise ValueError("dependency cycle detected")
                if mid in visited: return
                visiting.add(mid)
                for dep in self._modules[mid].dependencies: visit(dep)
                visiting.remove(mid); visited.add(mid)
                if mid in wanted: out.append(mid)
            for mid in sorted(wanted): visit(mid)
            return tuple(out)
    def propose(self, proposal):
        if not isinstance(proposal, ChangeProposal): raise TypeError("proposal must be ChangeProposal")
        with self._lock:
            if proposal.target not in self._active: raise KeyError(proposal.target)
            if proposal.parent_digest != self._active[proposal.target]: raise ValueError("proposal parent is not active")
            key = proposal.digest
            if key not in self._candidates: self._candidates[key] = CandidateRecord(proposal, key, CandidateStatus.PROPOSED)
            return self._candidates[key]
    def validate(self, candidate_digest):
        _digest(candidate_digest, "candidate_digest")
        with self._lock:
            current = self._candidates[candidate_digest]
            if current.status is not CandidateStatus.PROPOSED: raise ValueError("only proposed candidates may be validated")
            proposal = current.proposal
        report = self._validator(proposal)
        if not isinstance(report, Mapping) or not report: raise ValueError("validator must return evidence mapping")
        status = CandidateStatus.VALIDATED if report.get("status") == "pass" else CandidateStatus.REJECTED
        result = CandidateRecord(proposal, candidate_digest, status, report, current.created_at)
        with self._lock:
            if self._candidates[candidate_digest].status is not CandidateStatus.PROPOSED: raise RuntimeError("candidate state changed")
            self._candidates[candidate_digest] = result
        return result
    def promote(self, candidate_digest, approval):
        _digest(candidate_digest, "candidate_digest")
        if not isinstance(approval, PromotionApproval): raise TypeError("PromotionApproval required")
        if approval.candidate_digest != candidate_digest: raise ValueError("approval candidate mismatch")
        with self._lock:
            if approval.approval_id in self._used_approvals: raise ValueError("approval ticket already consumed")
            candidate = self._candidates[candidate_digest]
            if candidate.status is not CandidateStatus.VALIDATED: raise ValueError("candidate is not validated")
            target = candidate.proposal.target
            if self._active[target] != candidate.proposal.parent_digest: raise RuntimeError("stale candidate")
            promoted = CandidateRecord(candidate.proposal, candidate_digest, CandidateStatus.PROMOTED,
                {**dict(candidate.validation_report), "approval_id": approval.approval_id, "approver_id": approval.approver_id}, candidate.created_at)
            self._used_approvals.add(approval.approval_id); self._candidates[candidate_digest] = promoted
            self._active[target] = candidate_digest; self._history[target].append(candidate_digest)
            return promoted
    def rollback(self, target, *, expected_current):
        _text(target, "target"); _digest(expected_current, "expected_current")
        with self._lock:
            if target not in self._active: raise KeyError(target)
            history = self._history[target]
            if self._active[target] != expected_current: raise RuntimeError("active digest changed; stale rollback refused")
            if len(history) < 2: raise ValueError("no prior approved artifact")
            current = history.pop(); prior = history[-1]; self._active[target] = prior
            rec = self._candidates.get(current)
            if rec: self._candidates[current] = CandidateRecord(rec.proposal, current, CandidateStatus.ROLLED_BACK, rec.validation_report, rec.created_at)
            return prior
    def active_digest(self, target):
        with self._lock: return self._active[target]
