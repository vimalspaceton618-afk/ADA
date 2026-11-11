"""Role-aware human approval service for governed actions."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
from .. import ApprovalAuthority, ApprovalTicket, Principal, PermissionManager

@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    principal_id: str
    action: str
    resource: str
    arguments_digest: str
    approver_id: str

class ActionApprovalService:
    def __init__(self, authority: ApprovalAuthority, permissions: PermissionManager):
        if not isinstance(authority, ApprovalAuthority) or not isinstance(permissions, PermissionManager):
            raise TypeError("approval authority and permission manager are required")
        self.authority, self.permissions = authority, permissions
    def issue(self, *, principal: Principal, approver: Principal, action: str,
              resource: str, arguments: Mapping[str, Any], ttl_seconds: float = 300.0,
              now: float | None = None) -> ApprovalTicket:
        if not isinstance(principal, Principal) or not isinstance(approver, Principal):
            raise TypeError("principal and approver must be Principal values")
        if not self.permissions.has_permission(approver, "action.approve"):
            raise PermissionError("approver lacks action.approve")
        return self.authority.issue(
            principal_id=principal.principal_id, approver_id=approver.principal_id,
            action=action, resource=resource, arguments=arguments,
            ttl_seconds=ttl_seconds, now=now,
        )
    def consume(self, ticket: ApprovalTicket, *, principal: Principal, action: str,
                resource: str, arguments: Mapping[str, Any], now: float | None = None) -> bool:
        if not isinstance(principal, Principal):
            raise TypeError("principal must be Principal")
        return self.authority.consume(
            ticket, principal_id=principal.principal_id, action=action,
            resource=resource, arguments=arguments, now=now,
        )

__all__ = ["ApprovalRequest", "ActionApprovalService"]
