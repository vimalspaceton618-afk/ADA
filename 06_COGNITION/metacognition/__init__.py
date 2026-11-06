"""Metacognitive quality and provenance monitor."""
from dataclasses import dataclass
@dataclass(frozen=True, slots=True)
class CognitiveAudit:
    evidence_count: int
    decision_abstained: bool
    has_rationale: bool
    warnings: tuple[str,...]
def audit(result):
    evidence=tuple(getattr(result,"evidence",()))
    decision=getattr(result,"decision",None)
    rationale=getattr(decision,"rationale","")
    status=getattr(getattr(decision,"status",None),"value",None)
    return CognitiveAudit(len(evidence),status=="abstained",bool(isinstance(rationale,str) and rationale.strip()),tuple(getattr(result,"warnings",())))
