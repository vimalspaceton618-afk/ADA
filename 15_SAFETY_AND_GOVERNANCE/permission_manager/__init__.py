"""Resource-scoped, expiring permission grants layered over role permissions."""
from __future__ import annotations
import time
from dataclasses import dataclass
from .. import Principal, PermissionManager

@dataclass(frozen=True, slots=True)
class PermissionGrant:
    principal_id: str
    action: str
    resource_prefix: str
    expires_at: float
    grant_id: str
    def __post_init__(self):
        for value in (self.principal_id, self.action, self.resource_prefix, self.grant_id):
            if not value.strip(): raise ValueError("grant fields must be non-empty")

class ScopedPermissionManager:
    def __init__(self, roles: PermissionManager, grants=()):
        if not isinstance(roles, PermissionManager): raise TypeError("roles must be PermissionManager")
        self.roles, self._grants = roles, tuple(grants)
    def grant(self, grant: PermissionGrant):
        if not isinstance(grant, PermissionGrant): raise TypeError("grant must be PermissionGrant")
        if any(x.grant_id == grant.grant_id for x in self._grants): raise ValueError("grant ID already exists")
        self._grants += (grant,)
    def revoke(self, grant_id: str):
        before = len(self._grants)
        self._grants = tuple(x for x in self._grants if x.grant_id != grant_id)
        return len(self._grants) != before
    def allows(self, principal: Principal, action: str, resource: str, *, now: float | None = None) -> bool:
        if not isinstance(principal, Principal): raise TypeError("principal must be Principal")
        if self.roles.has_permission(principal, action): return True
        current = time.time() if now is None else float(now)
        return any(g.principal_id == principal.principal_id and g.action == action
                   and (resource == g.resource_prefix or resource.startswith(g.resource_prefix.rstrip("/") + "/"))
                   and current < g.expires_at for g in self._grants)

__all__ = ["PermissionGrant", "ScopedPermissionManager"]
