"""Typed provenance-linked relationships; no automatic inference."""
from dataclasses import dataclass
@dataclass(frozen=True, slots=True)
class Relation:
    subject_id: str
    predicate: str
    object_id: str
    evidence_id: str
    def __post_init__(self):
        for name in ("subject_id","predicate","object_id","evidence_id"):
            if not isinstance(getattr(self,name),str) or not getattr(self,name).strip(): raise ValueError(f"{name} required")
        if self.subject_id==self.object_id and self.predicate.casefold()=="causes": raise ValueError("self-causation rejected")
def build_relation(subject_id,predicate,object_id,evidence_id):
    return Relation(subject_id,predicate,object_id,evidence_id)
