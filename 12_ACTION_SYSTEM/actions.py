"""Deny-by-default action gateway. Production sandboxing is an adapter boundary."""
from __future__ import annotations
import hashlib, json, math, threading, time, uuid
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping

def _text(v,n):
    if not isinstance(v,str) or not v.strip(): raise ValueError(f"{n} must be non-empty")
    return v.strip()
def _freeze(v):
    if v is None or type(v) in (str,int,bool): return v
    if isinstance(v,float):
        if not math.isfinite(v): raise ValueError("non-finite value")
        return v
    if isinstance(v,Mapping):
        if any(not isinstance(k,str) for k in v): raise TypeError("argument keys must be strings")
        return MappingProxyType({k:_freeze(x) for k,x in v.items()})
    if isinstance(v,(list,tuple)): return tuple(_freeze(x) for x in v)
    raise TypeError(f"unsupported value: {type(v).__name__}")
def _thaw(v):
    if isinstance(v,Mapping): return {k:_thaw(x) for k,x in v.items()}
    if isinstance(v,tuple): return [_thaw(x) for x in v]
    return v

class RiskLevel(str,Enum):
    LOW="low"; MEDIUM="medium"; HIGH="high"; CRITICAL="critical"
class ActionStatus(str,Enum):
    DENIED="denied"; AWAITING_APPROVAL="awaiting_approval"; SUCCEEDED="succeeded"; FAILED="failed"; TIMED_OUT="timed_out"

@dataclass(frozen=True,slots=True)
class ToolSpec:
    tool_id:str
    version:str
    handler:Callable[[Mapping[str,Any]],Any]
    allowed_principals:frozenset[str]
    required_arguments:frozenset[str]=frozenset()
    optional_arguments:frozenset[str]=frozenset()
    risk:RiskLevel=RiskLevel.HIGH
    timeout_seconds:float=10.0
    requires_approval:bool=True
    def __post_init__(self):
        _text(self.tool_id,"tool_id"); _text(self.version,"version")
        if not callable(self.handler): raise TypeError("handler must be callable")
        if not isinstance(self.allowed_principals,frozenset) or not self.allowed_principals: raise ValueError("allowed_principals must be non-empty frozenset")
        if any(not isinstance(x,str) or not x.strip() for x in self.allowed_principals): raise ValueError("principal names must be non-empty")
        if self.required_arguments & self.optional_arguments: raise ValueError("argument sets overlap")
        if not isinstance(self.risk,RiskLevel): raise TypeError("risk must be RiskLevel")
        if isinstance(self.timeout_seconds,bool) or not isinstance(self.timeout_seconds,(int,float)) or not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds<=0:
            raise ValueError("timeout_seconds must be positive finite")
        if type(self.requires_approval) is not bool: raise TypeError("requires_approval must be bool")

@dataclass(frozen=True,slots=True)
class ApprovalTicket:
    call_id:str
    tool_id:str
    principal:str
    argument_digest:str
    approver:str
    ticket_id:str=field(default_factory=lambda:str(uuid.uuid4()))
    issued_at:float=field(default_factory=time.time)

@dataclass(frozen=True,slots=True)
class ActionRecord:
    call_id:str; tool_id:str; principal:str; status:ActionStatus; argument_digest:str
    started_at:float; finished_at:float; result_digest:str|None=None; error:str|None=None

class ToolGateway:
    """In-process reference. Hard timeouts and process isolation must be supplied externally."""
    def __init__(self):
        self._tools={}; self._records=[]; self._used_calls=set(); self._used_tickets=set(); self._lock=threading.RLock()
    @property
    def audit_log(self):
        with self._lock: return tuple(self._records)
    def register(self,spec):
        if not isinstance(spec,ToolSpec): raise TypeError("spec must be ToolSpec")
        with self._lock:
            if spec.tool_id in self._tools: raise ValueError("tool already registered")
            self._tools[spec.tool_id]=spec
    @staticmethod
    def digest(value):
        raw=json.dumps(_thaw(_freeze(value)),sort_keys=True,separators=(",",":"))
        return hashlib.sha256(raw.encode()).hexdigest()
    def issue_approval(self,*,call_id,tool_id,principal,arguments,approver):
        # Constructs a record only; production approver identity must be authenticated externally.
        for n,v in (("call_id",call_id),("tool_id",tool_id),("principal",principal),("approver",approver)): _text(v,n)
        return ApprovalTicket(call_id,tool_id,principal,self.digest(arguments),approver)
    def execute(self,tool_id,arguments,*,principal,call_id=None,approval=None):
        _text(tool_id,"tool_id"); _text(principal,"principal")
        if not isinstance(arguments,Mapping): raise TypeError("arguments must be mapping")
        cid=call_id or str(uuid.uuid4()); _text(cid,"call_id"); ad=self.digest(arguments)
        with self._lock:
            if cid in self._used_calls: raise ValueError("call_id replay blocked")
            spec=self._tools.get(tool_id)
            if spec is None: return self._record(cid,tool_id,principal,ActionStatus.DENIED,ad,error="unknown tool")
            if principal not in spec.allowed_principals: return self._record(cid,tool_id,principal,ActionStatus.DENIED,ad,error="principal unauthorized")
            keys=set(arguments); missing=spec.required_arguments-keys; unknown=keys-spec.required_arguments-spec.optional_arguments
            if missing or unknown: return self._record(cid,tool_id,principal,ActionStatus.DENIED,ad,error=f"argument schema mismatch missing={sorted(missing)} unknown={sorted(unknown)}")
            if spec.requires_approval:
                valid=(isinstance(approval,ApprovalTicket) and approval.call_id==cid and approval.tool_id==tool_id
                       and approval.principal==principal and approval.argument_digest==ad and bool(approval.approver.strip()))
                if not valid:
                    if approval is None:
                        return self._record(cid,tool_id,principal,ActionStatus.AWAITING_APPROVAL,ad,error="matching approval ticket required",consume=False)
                    return self._record(cid,tool_id,principal,ActionStatus.DENIED,ad,error="supplied approval ticket does not match this action")
                if approval.ticket_id in self._used_tickets: return self._record(cid,tool_id,principal,ActionStatus.DENIED,ad,error="approval ticket replay blocked")
                self._used_tickets.add(approval.ticket_id)
            self._used_calls.add(cid)
        start=time.monotonic()
        try:
            result=spec.handler(_freeze(arguments))
            if time.monotonic()-start>spec.timeout_seconds:
                return self._record(cid,tool_id,principal,ActionStatus.TIMED_OUT,ad,error="timeout observed after handler returned; execution was not preempted",started_at=start)
            return self._record(cid,tool_id,principal,ActionStatus.SUCCEEDED,ad,result_digest=self.digest(result),started_at=start)
        except Exception as exc:
            return self._record(cid,tool_id,principal,ActionStatus.FAILED,ad,error=f"{type(exc).__name__}: {exc}",started_at=start)
    def _record(self,cid,tool,principal,status,ad,*,result_digest=None,error=None,consume=True,started_at=None):
        finish=time.monotonic(); record=ActionRecord(cid,tool,principal,status,ad,started_at or finish,finish,result_digest,error)
        with self._lock:
            self._records.append(record)
            if consume: self._used_calls.add(cid)
        return record

class ActionController:
    def __init__(self,gateway):
        if not isinstance(gateway,ToolGateway): raise TypeError("gateway must be ToolGateway")
        self.gateway=gateway
    def dispatch(self,*args,**kwargs): return self.gateway.execute(*args,**kwargs)
