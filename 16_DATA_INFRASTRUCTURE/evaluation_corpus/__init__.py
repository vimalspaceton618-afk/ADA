"""Access-controlled evaluation corpus separated from training datasets."""
from __future__ import annotations
from dataclasses import dataclass
import secrets
from .. import DatasetManifest, DatasetSplit

@dataclass(frozen=True, slots=True)
class EvaluationAccess:
    evaluator_id: str
    token: str

class EvaluationCorpus:
    def __init__(self, manifest: DatasetManifest, records):
        if not isinstance(manifest, DatasetManifest) or manifest.split is not DatasetSplit.EVALUATION:
            raise ValueError("evaluation corpus requires an EVALUATION manifest")
        self.manifest, self._records = manifest, tuple(records)
        if len(self._records) != manifest.sample_count: raise ValueError("record count does not match manifest")
        self._tokens = {}
    def authorize(self, evaluator_id: str):
        if not evaluator_id.strip(): raise ValueError("evaluator_id required")
        token = secrets.token_urlsafe(32)
        self._tokens[token] = evaluator_id
        return EvaluationAccess(evaluator_id, token)
    def read(self, access: EvaluationAccess):
        if not isinstance(access, EvaluationAccess) or self._tokens.get(access.token) != access.evaluator_id:
            raise PermissionError("invalid evaluation corpus access")
        return self._records
    def revoke(self, access: EvaluationAccess):
        return self._tokens.pop(access.token, None) is not None

__all__ = ["EvaluationAccess", "EvaluationCorpus"]
