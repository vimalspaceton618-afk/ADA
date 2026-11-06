"""Generate candidate hypotheses with explicit evidence context."""
from dataclasses import dataclass
@dataclass(frozen=True, slots=True)
class Hypothesis:
    statement: str
    supporting_evidence: tuple[str,...]
    missing_evidence: tuple[str,...]=()
    def __post_init__(self):
        if not isinstance(self.statement,str) or not self.statement.strip(): raise ValueError("statement must be non-empty")
        if not self.supporting_evidence and not self.missing_evidence: raise ValueError("evidence context required")
def propose(statement,*,supporting_evidence=(),missing_evidence=()):
    return Hypothesis(statement,tuple(supporting_evidence),tuple(missing_evidence))
