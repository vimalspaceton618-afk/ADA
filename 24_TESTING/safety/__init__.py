"""Safety regression probes for policy and approval boundaries."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from importlib import import_module

@dataclass(frozen=True, slots=True)
class SafetyProbe:
    probe_id: str
    run: Callable[[], bool]
    expected_safe: bool = True
    def __post_init__(self):
        if not self.probe_id.strip() or not callable(self.run): raise ValueError("probe ID and callable required")

def run_safety_probes(probes):
    results = []
    for probe in probes:
        if not isinstance(probe, SafetyProbe): raise TypeError("probes must be SafetyProbe")
        try: observed = bool(probe.run())
        except Exception: observed = False
        results.append((probe.probe_id, observed == probe.expected_safe))
    return tuple(results)

def default_deny_probe():
    safety = import_module("15_SAFETY_AND_GOVERNANCE")
    principal = safety.Principal("probe")
    decision = safety.PolicyEngine().evaluate(principal, "tool.execute", "unknown")
    return decision.status is safety.AuthorizationStatus.DENIED

__all__ = ["SafetyProbe", "run_safety_probes", "default_deny_probe"]
