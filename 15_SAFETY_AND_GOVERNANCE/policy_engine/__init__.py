"""Versioned policy bundles and deterministic policy evaluation."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from .. import PolicyEngine, PolicyRule, Principal, PolicyDecision

@dataclass(frozen=True, slots=True)
class PolicyBundle:
    bundle_id: str
    version: str
    rules: tuple[PolicyRule, ...]
    def __post_init__(self):
        if not self.bundle_id.strip() or not self.version.strip(): raise ValueError("bundle ID/version required")
        if any(not isinstance(x, PolicyRule) for x in self.rules): raise TypeError("rules must contain PolicyRule values")
        ids = [x.rule_id for x in self.rules]
        if len(ids) != len(set(ids)): raise ValueError("rule IDs must be unique")
    @property
    def digest(self):
        body = [(r.rule_id, r.action, r.resource_prefix, r.effect.value, sorted(r.roles), r.priority) for r in self.rules]
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

class PolicyBundleEngine:
    def __init__(self, bundle: PolicyBundle):
        if not isinstance(bundle, PolicyBundle): raise TypeError("bundle must be PolicyBundle")
        self.bundle = bundle
        self.engine = PolicyEngine(bundle.rules)
    def evaluate(self, principal: Principal, action: str, resource: str) -> PolicyDecision:
        return self.engine.evaluate(principal, action, resource)

__all__ = ["PolicyBundle", "PolicyBundleEngine"]
