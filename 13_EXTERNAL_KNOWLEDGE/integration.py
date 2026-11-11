"""Verified external-knowledge adapters for downstream cognition."""

from __future__ import annotations

import math
from collections.abc import Iterable
from importlib import import_module

from .knowledge import KnowledgeChunk, KnowledgeStatus, KnowledgeStore


def to_cognitive_evidence(
    store: KnowledgeStore,
    chunks: Iterable[KnowledgeChunk],
    *,
    score: float,
):
    """Adapt stored, verified chunks into cognition evidence.

    The score is an explicit caller-supplied relevance score in [0, 1]; this
    adapter does not infer confidence or calibrate source reliability.
    """
    if not isinstance(store, KnowledgeStore):
        raise TypeError("store must be a KnowledgeStore")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score must be numeric")
    if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
        raise ValueError("score must be finite and in [0, 1]")

    evidence_type = import_module("06_COGNITION.contracts").EvidenceItem
    output = []
    seen = set()
    for chunk in tuple(chunks):
        if not isinstance(chunk, KnowledgeChunk):
            raise TypeError("chunks must contain KnowledgeChunk values")
        document = store.get(chunk.document_id)
        if document.status is not KnowledgeStatus.VERIFIED:
            raise PermissionError("unverified knowledge cannot become cognition evidence")
        stored = store.get_chunks(chunk.document_id)
        if chunk.ordinal >= len(stored) or stored[chunk.ordinal] != chunk:
            raise ValueError("chunk is not present in the verified document's stored chunk set")
        evidence_id = f"knowledge:{chunk.document_id}:{chunk.ordinal}"
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        output.append(evidence_type(
            evidence_id=evidence_id,
            source_kind="external_knowledge",
            source_id=document.source.source_id,
            source_digest=chunk.digest,
            content=chunk.text,
            score=float(score),
        ))
    return tuple(output)
