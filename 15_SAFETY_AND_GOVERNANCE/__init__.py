"""Fail-closed policy, approval and tamper-evident audit primitives.

This package supplies process-local reference controls. Production identity,
key management and OS/container isolation remain deployment responsibilities.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


def _canonical(value: Any) -> bytes:
    def thaw(item):
        if isinstance(item, Mapping):
            return {str(k): thaw(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [thaw(v) for v in item]
        if isinstance(item, Enum):
            return item.value
        return item
    return json.dumps(thaw(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


class PolicyEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class AuthorizationStatus(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass(frozen=True, slots=True)
class Principal:
    principal_id: str
    roles: frozenset[str] = frozenset()
    permissions: frozenset[str] = frozenset()

    def __post_init__(self):
        if not self.principal_id.strip():
            raise ValueError("principal_id must be non-empty")
        if any(not x.strip() for x in self.roles | self.permissions):
            raise ValueError("roles and permissions must be non-empty strings")


@dataclass(frozen=True, slots=True)
class PolicyRule:
    rule_id: str
    action: str
    resource_prefix: str
    effect: PolicyEffect
    roles: frozenset[str] = frozenset()
    priority: int = 0

    def __post_init__(self):
        if not self.rule_id.strip() or not self.action.strip() or not self.resource_prefix.strip():
            raise ValueError("rule identity, action and resource prefix are required")
        if not isinstance(self.effect, PolicyEffect):
            raise TypeError("effect must be PolicyEffect")
        if type(self.priority) is not int:
            raise TypeError("priority must be int")


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    status: AuthorizationStatus
    reason: str
    matched_rule_ids: tuple[str, ...] = ()


class PolicyEngine:
    """Explicit rule engine; no matching allow means deny."""

    def __init__(self, rules=()):
        self._rules = tuple(rules)
        if any(not isinstance(r, PolicyRule) for r in self._rules):
            raise TypeError("rules must contain PolicyRule values")

    def evaluate(self, principal: Principal, action: str, resource: str) -> PolicyDecision:
        if not isinstance(principal, Principal):
            raise TypeError("principal must be Principal")
        if not action.strip() or not resource.strip():
            raise ValueError("action and resource must be non-empty")
        matches = [
            rule for rule in self._rules
            if (rule.action in {"*", action}
                and (resource == rule.resource_prefix
                     or resource.startswith(rule.resource_prefix.rstrip("/") + "/"))
                and (not rule.roles or bool(rule.roles & principal.roles)))
        ]
        matches.sort(key=lambda r: (-r.priority, r.rule_id))
        ids = tuple(r.rule_id for r in matches)
        if any(r.effect is PolicyEffect.DENY for r in matches):
            return PolicyDecision(AuthorizationStatus.DENIED, "explicit deny rule matched", ids)
        if any(r.effect is PolicyEffect.REQUIRE_APPROVAL for r in matches):
            return PolicyDecision(AuthorizationStatus.AWAITING_APPROVAL, "approval rule matched", ids)
        if any(r.effect is PolicyEffect.ALLOW for r in matches):
            return PolicyDecision(AuthorizationStatus.ALLOWED, "explicit allow rule matched", ids)
        return PolicyDecision(AuthorizationStatus.DENIED, "default deny: no policy rule matched", ())


class PermissionManager:
    def __init__(self, role_permissions: Mapping[str, frozenset[str]]):
        self._roles = {k: frozenset(v) for k, v in role_permissions.items()}
        if any(not k.strip() for k in self._roles):
            raise ValueError("role names must be non-empty")

    def has_permission(self, principal: Principal, permission: str) -> bool:
        if not isinstance(principal, Principal):
            raise TypeError("principal must be Principal")
        return permission in principal.permissions or any(
            permission in self._roles.get(role, ()) for role in principal.roles
        )


@dataclass(frozen=True, slots=True)
class ApprovalTicket:
    approval_id: str
    principal_id: str
    approver_id: str
    action: str
    resource: str
    arguments_digest: str
    issued_at: float
    expires_at: float
    signature: str


class ApprovalAuthority:
    """HMAC-bound, expiring, single-use approvals; keys must stay outside model context."""

    def __init__(self, secret: bytes, authorized_approvers: frozenset[str]):
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("approval secret must contain at least 32 bytes")
        if not authorized_approvers:
            raise ValueError("authorized_approvers cannot be empty")
        self._secret = secret
        self._approvers = frozenset(authorized_approvers)
        self._used = set()
        self._lock = threading.RLock()

    def _payload(self, approval_id, principal_id, approver_id, action, resource,
                 arguments_digest, issued_at, expires_at):
        return (approval_id, principal_id, approver_id, action, resource,
                arguments_digest, issued_at, expires_at)

    def issue(self, *, principal_id: str, approver_id: str, action: str,
              resource: str, arguments: Any, ttl_seconds: float = 300.0,
              now: float | None = None) -> ApprovalTicket:
        if approver_id not in self._approvers:
            raise PermissionError("approver is not authorized")
        if not all(isinstance(v, str) and v.strip()
                   for v in (principal_id, action, resource)):
            raise ValueError("principal, action and resource are required")
        if isinstance(ttl_seconds, bool) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        issued = time.time() if now is None else float(now)
        expires = issued + float(ttl_seconds)
        approval_id = str(uuid.uuid4())
        args_digest = digest(arguments)
        payload = self._payload(approval_id, principal_id, approver_id, action,
                                resource, args_digest, issued, expires)
        signature = hmac.new(self._secret, _canonical(payload), hashlib.sha256).hexdigest()
        return ApprovalTicket(approval_id, principal_id, approver_id, action,
                              resource, args_digest, issued, expires, signature)

    def consume(self, ticket: ApprovalTicket, *, principal_id: str, action: str,
                resource: str, arguments: Any, now: float | None = None) -> bool:
        if not isinstance(ticket, ApprovalTicket):
            return False
        current = time.time() if now is None else float(now)
        payload = self._payload(
            ticket.approval_id, ticket.principal_id, ticket.approver_id,
            ticket.action, ticket.resource, ticket.arguments_digest,
            ticket.issued_at, ticket.expires_at,
        )
        expected = hmac.new(self._secret, _canonical(payload), hashlib.sha256).hexdigest()
        valid = (
            hmac.compare_digest(ticket.signature, expected)
            and ticket.approver_id in self._approvers
            and ticket.principal_id == principal_id
            and ticket.action == action and ticket.resource == resource
            and ticket.arguments_digest == digest(arguments)
            and ticket.issued_at <= current < ticket.expires_at
        )
        if not valid:
            return False
        with self._lock:
            if ticket.approval_id in self._used:
                return False
            self._used.add(ticket.approval_id)
            return True


@dataclass(frozen=True, slots=True)
class AuditEntry:
    sequence: int
    event_type: str
    subject_id: str
    payload_digest: str
    previous_hash: str
    entry_hash: str
    timestamp: float


class AuditLedger:
    """Append-only API with hash chaining; external storage is needed for durability."""

    def __init__(self):
        self._entries = []
        self._lock = threading.RLock()

    def append(self, event_type: str, subject_id: str, payload: Any,
               *, timestamp: float | None = None) -> AuditEntry:
        if not event_type.strip() or not subject_id.strip():
            raise ValueError("event_type and subject_id are required")
        with self._lock:
            seq = len(self._entries)
            prev = self._entries[-1].entry_hash if self._entries else "0" * 64
            ts = time.time() if timestamp is None else float(timestamp)
            body = (seq, event_type, subject_id, digest(payload), prev, ts)
            entry = AuditEntry(seq, event_type, subject_id, body[3], prev,
                               hashlib.sha256(_canonical(body)).hexdigest(), ts)
            self._entries.append(entry)
            return entry

    @property
    def entries(self):
        with self._lock:
            return tuple(self._entries)

    def verify(self) -> bool:
        with self._lock:
            previous = "0" * 64
            for index, entry in enumerate(self._entries):
                body = (entry.sequence, entry.event_type, entry.subject_id,
                        entry.payload_digest, previous, entry.timestamp)
                if entry.sequence != index or entry.previous_hash != previous:
                    return False
                if hashlib.sha256(_canonical(body)).hexdigest() != entry.entry_hash:
                    return False
                previous = entry.entry_hash
            return True



DEFAULT_ROLE_PERMISSIONS = {
    "reader": frozenset({"artifact.read", "report.read", "telemetry.read"}),
    "researcher": frozenset({"artifact.read", "candidate.create", "sandbox.run"}),
    "evaluator": frozenset({"artifact.read", "evaluation.run", "report.sign"}),
    "release_approver": frozenset({"artifact.read", "release.approve"}),
    "administrator": frozenset({"identity.manage", "policy.manage", "audit.read"}),
}


@dataclass(frozen=True, slots=True)
class GovernedActionResult:
    policy: PolicyDecision
    action_record: Any | None
    reason: str


class GovernedActionController:
    """Enforce a separate policy/approval plane before the existing tool gateway."""

    def __init__(self, *, policy: PolicyEngine, gateway: Any,
                 audit: AuditLedger, approval_authority: ApprovalAuthority,
                 permissions: PermissionManager | None = None):
        from importlib import import_module
        gateway_type = import_module("12_ACTION_SYSTEM").ToolGateway
        if not isinstance(policy, PolicyEngine) or not isinstance(audit, AuditLedger):
            raise TypeError("policy and audit must use the safety reference implementations")
        if not isinstance(gateway, gateway_type) or not isinstance(approval_authority, ApprovalAuthority):
            raise TypeError("gateway and approval_authority are invalid")
        if permissions is not None and not isinstance(permissions, PermissionManager):
            raise TypeError("permissions must be PermissionManager")
        self.policy = policy
        self.gateway = gateway
        self.audit = audit
        self.approvals = approval_authority
        self.permissions = permissions

    def dispatch(self, tool_id: str, arguments: Mapping[str, Any], *,
                 principal: Principal, call_id: str,
                 approval: ApprovalTicket | None = None,
                 now: float | None = None) -> GovernedActionResult:
        action = "tool.execute"
        if self.permissions is not None and not self.permissions.has_permission(principal, action):
            decision = PolicyDecision(AuthorizationStatus.DENIED, "principal lacks tool.execute permission")
            self.audit.append("action.denied", principal.principal_id,
                              {"tool": tool_id, "call_id": call_id, "reason": decision.reason})
            return GovernedActionResult(decision, None, decision.reason)

        decision = self.policy.evaluate(principal, action, tool_id)
        inner_approval = None
        if decision.status is AuthorizationStatus.DENIED:
            self.audit.append("action.denied", principal.principal_id,
                              {"tool": tool_id, "call_id": call_id, "reason": decision.reason})
            return GovernedActionResult(decision, None, decision.reason)

        if decision.status is AuthorizationStatus.AWAITING_APPROVAL:
            if not self.approvals.consume(
                approval, principal_id=principal.principal_id, action=action,
                resource=tool_id, arguments=arguments, now=now,
            ):
                self.audit.append("action.awaiting_approval", principal.principal_id,
                                  {"tool": tool_id, "call_id": call_id})
                return GovernedActionResult(
                    decision, None, "a valid, unexpired, single-use human approval is required"
                )
            inner_approval = self.gateway.issue_approval(
                call_id=call_id, tool_id=tool_id, principal=principal.principal_id,
                arguments=arguments, approver=approval.approver_id,
            )

        record = self.gateway.execute(
            tool_id, arguments, principal=principal.principal_id,
            call_id=call_id, approval=inner_approval,
        )
        self.audit.append(
            f"action.{record.status.value}", principal.principal_id,
            {"tool": tool_id, "call_id": call_id, "argument_digest": record.argument_digest,
             "result_digest": record.result_digest, "error": record.error},
        )
        return GovernedActionResult(decision, record, record.status.value)


class ArtifactPromotionController:
    """Atomic digest-reference promotion after independent gates and human review."""

    REQUIRED_GATES = frozenset({
        "integrity", "structural", "numerical", "target_capability",
        "protected_regression", "robustness", "safety", "efficiency",
        "provenance", "review",
    })

    def __init__(self):
        self._active = {}
        self._history = {}
        self._lock = threading.RLock()

    def register_approved(self, model_id: str, artifact_digest: str):
        if not model_id.strip():
            raise ValueError("model_id is required")
        _validate_digest(artifact_digest, "artifact_digest")
        with self._lock:
            if model_id in self._active:
                raise ValueError("model already has an approved artifact")
            self._active[model_id] = artifact_digest
            self._history[model_id] = [artifact_digest]

    def promote(self, model_id: str, candidate_digest: str,
                gate_results: Mapping[str, bool], approver: Principal):
        _validate_digest(candidate_digest, "candidate_digest")
        if not isinstance(approver, Principal) or "release_approver" not in approver.roles:
            raise PermissionError("promotion requires a release_approver principal")
        if not isinstance(gate_results, Mapping):
            raise TypeError("gate_results must be a mapping")
        missing = self.REQUIRED_GATES - set(gate_results)
        invalid = {name for name, result in gate_results.items() if type(result) is not bool}
        failed = {name for name in self.REQUIRED_GATES if gate_results.get(name) is not True}
        if missing or invalid or failed:
            raise PermissionError(
                f"promotion blocked missing={sorted(missing)} invalid={sorted(invalid)} failed={sorted(failed)}"
            )
        with self._lock:
            if model_id not in self._active:
                raise KeyError(model_id)
            self._active[model_id] = candidate_digest
            self._history[model_id].append(candidate_digest)
            return candidate_digest

    def rollback(self, model_id: str, *, expected_current: str):
        _validate_digest(expected_current, "expected_current")
        with self._lock:
            if self._active.get(model_id) != expected_current:
                raise RuntimeError("active artifact changed; stale rollback refused")
            history = self._history[model_id]
            if len(history) < 2:
                raise ValueError("no prior approved artifact")
            history.pop()
            self._active[model_id] = history[-1]
            return history[-1]

    def active_digest(self, model_id: str):
        with self._lock:
            return self._active[model_id]



__all__ = [
    "PolicyEffect", "AuthorizationStatus", "Principal", "PolicyRule",
    "PolicyDecision", "PolicyEngine", "PermissionManager", "ApprovalTicket",
    "ApprovalAuthority", "AuditEntry", "AuditLedger", "GovernedActionResult",
    "GovernedActionController", "ArtifactPromotionController",
    "DEFAULT_ROLE_PERMISSIONS", "digest",
]
