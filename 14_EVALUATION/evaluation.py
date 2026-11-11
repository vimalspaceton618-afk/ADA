"""Multi-dimensional evaluation contracts with fail-closed evidence gates."""
from __future__ import annotations
import hashlib, json, math, threading, uuid
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

def _text(v,n):
    if not isinstance(v,str) or not v.strip(): raise ValueError(f"{n} must be non-empty")
    return v.strip()
def _digest(v,n):
    if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v): raise ValueError(f"{n} must be lowercase SHA-256")
    return v
def _freeze(v):
    if v is None or type(v) in (str,int,bool): return v
    if isinstance(v,float):
        if not math.isfinite(v): raise ValueError("non-finite value")
        return v
    if isinstance(v,Mapping):
        if any(not isinstance(k,str) for k in v): raise TypeError("mapping keys must be strings")
        return MappingProxyType({k:_freeze(x) for k,x in v.items()})
    if isinstance(v,(list,tuple)): return tuple(_freeze(x) for x in v)
    raise TypeError("not JSON-compatible")
def _thaw(v):
    if isinstance(v,Mapping): return {k:_thaw(x) for k,x in v.items()}
    if isinstance(v,tuple): return [_thaw(x) for x in v]
    return v
class Verdict(str,Enum):
    PASS="pass"; FAIL="fail"; INCONCLUSIVE="inconclusive"
@dataclass(frozen=True,slots=True)
class EvaluationCase:
    case_id:str
    input:Mapping[str,Any]
    expected:Mapping[str,Any]
    tags:tuple[str,...]=()
    def __post_init__(self):
        _text(self.case_id,"case_id")
        if not isinstance(self.input,Mapping) or not isinstance(self.expected,Mapping): raise TypeError("input/expected must be mappings")
        object.__setattr__(self,"input",_freeze(self.input)); object.__setattr__(self,"expected",_freeze(self.expected))
        object.__setattr__(self,"tags",tuple(_text(x,"tag") for x in self.tags))
@dataclass(frozen=True,slots=True)
class EvaluationDataset:
    name:str
    cases:tuple[EvaluationCase,...]
    version:str
    dataset_id:str=field(default_factory=lambda:str(uuid.uuid4()))
    def __post_init__(self):
        _text(self.name,"name"); _text(self.version,"version")
        cases=tuple(self.cases)
        if not cases or any(not isinstance(c,EvaluationCase) for c in cases): raise ValueError("dataset needs EvaluationCase values")
        if len({c.case_id for c in cases})!=len(cases): raise ValueError("case IDs must be unique")
        object.__setattr__(self,"cases",cases)
    @property
    def digest(self):
        body={"name":self.name,"version":self.version,"cases":[{"id":c.case_id,"input":_thaw(c.input),"expected":_thaw(c.expected),"tags":c.tags} for c in self.cases]}
        return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest()
@dataclass(frozen=True,slots=True)
class Metric:
    name:str
    value:float
    direction:str="higher"
    def __post_init__(self):
        _text(self.name,"name")
        if isinstance(self.value,bool) or not isinstance(self.value,(int,float)) or not math.isfinite(float(self.value)): raise ValueError("metric must be finite numeric")
        if self.direction not in {"higher","lower"}: raise ValueError("direction must be higher/lower")
@dataclass(frozen=True,slots=True)
class EvaluationReport:
    subject_digest:str
    dataset_digest:str
    evaluator_id:str
    metrics:tuple[Metric,...]
    sample_count:int
    verdict:Verdict
    limitations:tuple[str,...]=()
    report_id:str=field(default_factory=lambda:str(uuid.uuid4()))
    def __post_init__(self):
        _digest(self.subject_digest,"subject_digest"); _digest(self.dataset_digest,"dataset_digest"); _text(self.evaluator_id,"evaluator_id")
        if type(self.sample_count) is not int or self.sample_count<0: raise ValueError("sample_count must be non-negative")
        metrics=tuple(self.metrics)
        if any(not isinstance(m,Metric) for m in metrics): raise TypeError("metrics must be Metric")
        if len({m.name for m in metrics})!=len(metrics): raise ValueError("metric names must be unique")
        object.__setattr__(self,"metrics",metrics); object.__setattr__(self,"limitations",tuple(_text(x,"limitation") for x in self.limitations))
        if not isinstance(self.verdict,Verdict): raise TypeError("verdict must be Verdict")
    @property
    def digest(self):
        body={"subject":self.subject_digest,"dataset":self.dataset_digest,"evaluator":self.evaluator_id,
              "metrics":[(m.name,m.value,m.direction) for m in self.metrics],"count":self.sample_count,
              "verdict":self.verdict.value,"limitations":self.limitations}
        return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest()
@dataclass(frozen=True,slots=True)
class GatePolicy:
    minimum_samples:int
    required_metrics:frozenset[str]
    thresholds:Mapping[str,tuple[str,float]]
    def __post_init__(self):
        if type(self.minimum_samples) is not int or self.minimum_samples<1: raise ValueError("minimum_samples must be positive")
        if not self.required_metrics: raise ValueError("required_metrics cannot be empty")
        object.__setattr__(self,"required_metrics",frozenset(self.required_metrics)); object.__setattr__(self,"thresholds",_freeze(self.thresholds))
        for name,pair in self.thresholds.items():
            _text(name,"metric"); direction,value=pair
            if direction not in {"min","max"} or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)): raise ValueError("invalid threshold")
@dataclass(frozen=True,slots=True)
class GateResult:
    verdict:Verdict
    reason:str
    missing_metrics:tuple[str,...]=()
    failed_metrics:tuple[str,...]=()
class EvaluationGate:
    def __init__(self,policy):
        if not isinstance(policy,GatePolicy): raise TypeError("policy must be GatePolicy")
        self.policy=policy
    def evaluate(self,report):
        if not isinstance(report,EvaluationReport): raise TypeError("report must be EvaluationReport")
        missing=tuple(sorted(self.policy.required_metrics-{m.name for m in report.metrics}))
        if report.verdict is Verdict.INCONCLUSIVE or report.sample_count<self.policy.minimum_samples or missing:
            return GateResult(Verdict.INCONCLUSIVE,"insufficient or incomplete evidence",missing)
        if report.verdict is Verdict.FAIL: return GateResult(Verdict.FAIL,"evaluator reported failure")
        vals={m.name:m.value for m in report.metrics}; failed=[]
        for name,(kind,threshold) in self.policy.thresholds.items():
            if name not in vals: return GateResult(Verdict.INCONCLUSIVE,f"threshold metric missing: {name}",(name,))
            if kind=="min" and vals[name]<threshold: failed.append(name)
            if kind=="max" and vals[name]>threshold: failed.append(name)
        return GateResult(Verdict.FAIL,"metric threshold failed",(),tuple(sorted(failed))) if failed else GateResult(Verdict.PASS,"all evidence and thresholds passed")
class EvaluatorRegistry:
    def __init__(self): self._evaluators={}; self._lock=threading.RLock()
    def register(self,evaluator_id,evaluator):
        _text(evaluator_id,"evaluator_id")
        if not callable(evaluator): raise TypeError("evaluator must be callable")
        with self._lock:
            if evaluator_id in self._evaluators: raise ValueError("evaluator already registered")
            self._evaluators[evaluator_id]=evaluator
    def evaluate(self,evaluator_id,dataset,subject_digest):
        _text(evaluator_id,"evaluator_id"); _digest(subject_digest,"subject_digest")
        if not isinstance(dataset,EvaluationDataset): raise TypeError("dataset must be EvaluationDataset")
        with self._lock: evaluator=self._evaluators[evaluator_id]
        report=evaluator(dataset,subject_digest)
        if not isinstance(report,EvaluationReport): raise TypeError("evaluator must return EvaluationReport")
        if report.evaluator_id!=evaluator_id or report.dataset_digest!=dataset.digest or report.subject_digest!=subject_digest: raise ValueError("report provenance mismatch")
        return report
class EvaluationSuite:
    """Runs suites and turns missing/failed evidence into INCONCLUSIVE reports."""
    def __init__(self): self._registries={}; self._reports=[]; self._lock=threading.RLock()
    def add(self,name,registry):
        _text(name,"name")
        if not isinstance(registry,EvaluatorRegistry): raise TypeError("registry must be EvaluatorRegistry")
        with self._lock:
            if name in self._registries: raise ValueError("suite already exists")
            self._registries[name]=registry
    def run(self,requests:Sequence[tuple[str,str,EvaluationDataset,str]]):
        results=[]
        for suite,evaluator,dataset,subject_digest in requests:
            try:
                with self._lock: registry=self._registries[suite]
                results.append(registry.evaluate(evaluator,dataset,subject_digest))
            except Exception as exc:
                dd=dataset.digest if isinstance(dataset,EvaluationDataset) else "0"*64
                results.append(EvaluationReport(subject_digest,dd,evaluator,(),0,Verdict.INCONCLUSIVE,(f"{type(exc).__name__}: {exc}",)))
        with self._lock: self._reports.extend(results)
        return tuple(results)
    @property
    def reports(self):
        with self._lock: return tuple(self._reports)
