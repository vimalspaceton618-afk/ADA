"""Policy-gated public inference facade."""
from __future__ import annotations
from typing import Any
from importlib import import_module

class ADAInferenceAPI:
    def __init__(self, inference_engine, policy_engine):
        inf = import_module("18_INFERENCE")
        safe = import_module("15_SAFETY_AND_GOVERNANCE")
        if not isinstance(inference_engine, inf.InferenceEngine) or not isinstance(policy_engine, safe.PolicyEngine):
            raise TypeError("inference engine and policy engine are required")
        self.inference, self.policy = inference_engine, policy_engine
    def generate(self, request, *, principal, version: str | None = None):
        safe = import_module("15_SAFETY_AND_GOVERNANCE")
        if not isinstance(principal, safe.Principal): raise TypeError("principal must be Principal")
        decision = self.policy.evaluate(principal, "model.infer", request.model_id)
        if decision.status is not safe.AuthorizationStatus.ALLOWED:
            raise PermissionError(decision.reason)
        return self.inference.generate(request, version=version)

__all__ = ["ADAInferenceAPI"]
