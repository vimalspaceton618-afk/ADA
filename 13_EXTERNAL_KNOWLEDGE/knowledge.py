"""Provenance-first external knowledge intake; no network access is performed."""
from __future__ import annotations
import hashlib, math, re, threading, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

def _text(v,n):
    if not isinstance(v,str) or not v.strip(): raise ValueError(f"{n} must be non-empty")
    return v.strip()
def _freeze(v):
    if v is None or type(v) in (str,bool,int): return v
    if isinstance(v,float):
        if not math.isfinite(v): raise ValueError("non-finite metadata")
        return v
    if isinstance(v,Mapping):
        if any(not isinstance(k,str) for k in v): raise TypeError("mapping keys must be strings")
        return MappingProxyType({k:_freeze(x) for k,x in v.items()})
    if isinstance(v,(list,tuple)): return tuple(_freeze(x) for x in v)
    raise TypeError("metadata must be JSON-compatible")
class KnowledgeStatus(str,Enum):
    PENDING="pending"; VERIFIED="verified"; REJECTED="rejected"; QUARANTINED="quarantined"
@dataclass(frozen=True,slots=True)
class SourceRecord:
    source_id:str
    uri:str
    publisher:str
    retrieved_at:datetime=field(default_factory=lambda:datetime.now(timezone.utc))
    content_digest:str|None=None
    license_id:str|None=None
    terms_uri:str|None=None
    def __post_init__(self):
        for n in ("source_id","uri","publisher"): _text(getattr(self,n),n)
        if self.content_digest is not None and (len(self.content_digest)!=64 or any(c not in "0123456789abcdef" for c in self.content_digest)):
            raise ValueError("content_digest must be lowercase SHA-256")
        if not isinstance(self.retrieved_at,datetime) or self.retrieved_at.tzinfo is None: raise ValueError("retrieved_at must be timezone-aware")
@dataclass(frozen=True,slots=True)
class KnowledgeDocument:
    source:SourceRecord
    content:str
    title:str
    document_id:str=field(default_factory=lambda:str(uuid.uuid4()))
    metadata:Mapping[str,Any]=field(default_factory=dict)
    status:KnowledgeStatus=KnowledgeStatus.PENDING
    def __post_init__(self):
        if not isinstance(self.source,SourceRecord): raise TypeError("source must be SourceRecord")
        _text(self.content,"content"); _text(self.title,"title"); _text(self.document_id,"document_id")
        if not isinstance(self.status,KnowledgeStatus): raise TypeError("status must be KnowledgeStatus")
        object.__setattr__(self,"metadata",_freeze(self.metadata))
        actual=hashlib.sha256(self.content.encode()).hexdigest()
        if self.source.content_digest is not None and self.source.content_digest!=actual: raise ValueError("source digest mismatch")
    @property
    def digest(self): return hashlib.sha256(self.content.encode()).hexdigest()
@dataclass(frozen=True,slots=True)
class VerificationResult:
    status:KnowledgeStatus
    verifier_id:str
    reason:str
    document_digest:str
    checked_at:datetime=field(default_factory=lambda:datetime.now(timezone.utc))
    def __post_init__(self):
        if self.status not in {KnowledgeStatus.VERIFIED,KnowledgeStatus.REJECTED,KnowledgeStatus.QUARANTINED}: raise ValueError("verification must be terminal")
        _text(self.verifier_id,"verifier_id"); _text(self.reason,"reason")
        if not isinstance(self.document_digest,str) or len(self.document_digest)!=64 or any(c not in "0123456789abcdef" for c in self.document_digest):
            raise ValueError("document_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.checked_at,datetime) or self.checked_at.tzinfo is None: raise ValueError("checked_at must be timezone-aware")
@dataclass(frozen=True,slots=True)
class KnowledgeChunk:
    document_id:str
    ordinal:int
    text:str
    digest:str
    def __post_init__(self):
        _text(self.document_id,"document_id")
        if type(self.ordinal) is not int or self.ordinal<0: raise ValueError("ordinal must be non-negative")
        _text(self.text,"text")
        if self.digest!=hashlib.sha256(self.text.encode()).hexdigest(): raise ValueError("chunk digest mismatch")
class KnowledgeStore:
    """Bounded process-local store; persistence and network acquisition are adapter seams."""
    def __init__(self,*,capacity=10000):
        if type(capacity) is not int or capacity<1: raise ValueError("capacity must be positive")
        self.capacity=capacity; self._documents={}; self._chunks={}; self._verification={}; self._lock=threading.RLock()
    def ingest(self,document):
        if not isinstance(document,KnowledgeDocument): raise TypeError("document must be KnowledgeDocument")
        with self._lock:
            prior=self._documents.get(document.document_id)
            if prior and prior.digest!=document.digest: raise ValueError("document id collision")
            if prior: return prior
            if len(self._documents)>=self.capacity: raise OverflowError("knowledge store capacity reached")
            self._documents[document.document_id]=document; return document
    def verify(self,document_id,result):
        if not isinstance(result,VerificationResult): raise TypeError("result must be VerificationResult")
        with self._lock:
            current=self._documents[document_id]
            if current.status is not KnowledgeStatus.PENDING: raise ValueError("only pending documents can be reviewed")
            if result.document_digest != current.digest: raise ValueError("verification result is bound to a different document digest")
            updated=KnowledgeDocument(current.source,current.content,current.title,current.document_id,current.metadata,result.status)
            self._documents[document_id]=updated; self._verification[document_id]=result
            return updated
    def chunk(self,document_id,*,size=1200,overlap=120):
        if type(size) is not int or size<1 or type(overlap) is not int or overlap<0 or overlap>=size: raise ValueError("invalid chunk size/overlap")
        with self._lock:
            doc=self._documents[document_id]
            if doc.status is not KnowledgeStatus.VERIFIED: raise PermissionError("only verified documents may be chunked")
            chunks=[]; start=0
            while start<len(doc.content):
                end=min(len(doc.content),start+size); piece=doc.content[start:end]
                chunks.append(KnowledgeChunk(document_id,len(chunks),piece,hashlib.sha256(piece.encode()).hexdigest()))
                if end==len(doc.content): break
                start=end-overlap
            self._chunks[document_id]=tuple(chunks); return self._chunks[document_id]
    def get_chunks(self,document_id):
        """Return stored chunks only while their source document remains verified."""
        with self._lock:
            doc=self._documents[document_id]
            if doc.status is not KnowledgeStatus.VERIFIED:
                raise PermissionError("only verified documents may expose chunks")
            return tuple(self._chunks.get(document_id, ()))
    def get(self,document_id):
        with self._lock: return self._documents[document_id]
    def search(self,query,*,limit=10):
        query=_text(query,"query")
        if type(limit) is not int or limit<1: raise ValueError("limit must be positive")
        terms=set(re.findall(r"[a-z0-9_]+",query.casefold())); ranked=[]
        with self._lock:
            for did,chunks in self._chunks.items():
                if self._documents[did].status is not KnowledgeStatus.VERIFIED: continue
                for ch in chunks:
                    score=len(terms & set(re.findall(r"[a-z0-9_]+",ch.text.casefold())))
                    if score: ranked.append((score,did,ch.ordinal,ch))
        ranked.sort(key=lambda x:(-x[0],x[1],x[2])); return tuple(x[3] for x in ranked[:limit])
class TextCleaner:
    @staticmethod
    def clean(text): return re.sub(r"\s+"," ",_text(text,"text")).strip()
class LicensePolicy:
    """Explicit allowlist; unknown license metadata is denied by default."""
    def __init__(self,allowed_license_ids=frozenset()): self.allowed=frozenset(allowed_license_ids)
    def check(self,source):
        if not isinstance(source,SourceRecord): raise TypeError("source must be SourceRecord")
        return source.license_id is not None and source.license_id in self.allowed
