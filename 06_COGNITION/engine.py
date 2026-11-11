"""Deterministic, evidence-grounded cognitive orchestration for ADA v1."""
from __future__ import annotations

import json
import re
from collections.abc import Iterable

from .contracts import (
    CognitiveDecision, CognitiveResult, CognitiveStatus, CognitiveTask,
    DecisionStatus, EvidenceItem, PlanStep,
)


_WORD = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _WORD.findall(value) if len(item) > 1}


def _json_text(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class CognitionEngine:
    """Compose explicit task, representations and retrieved memories.

    v1 is a transparent orchestration baseline, not a trained reasoning model.
    It does not execute plans, infer hidden facts, or call external tools.
    Candidate ranking uses lexical evidence overlap and abstains by default
    when evidence is absent, weak, or ambiguous.
    """

    def __init__(self, *, memory_engine=None, min_evidence_score: float = 0.05,
                 min_decision_margin: float = 0.02) -> None:
        for name, value in (("min_evidence_score", min_evidence_score),
                            ("min_decision_margin", min_decision_margin)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        self.memory_engine = memory_engine
        self.min_evidence_score = float(min_evidence_score)
        self.min_decision_margin = float(min_decision_margin)

    @staticmethod
    def _representation_evidence(result) -> list[EvidenceItem]:
        from importlib import import_module
        module = import_module("04_REPRESENTATION.contracts")
        if not isinstance(result, module.RepresentationResult):
            raise TypeError("representations must contain RepresentationResult values")
        if result.status.value not in {"complete", "partial"}:
            return []
        evidence = []
        for item in result.representations:
            evidence.append(EvidenceItem(
                evidence_id=item.representation_id,
                source_kind="representation",
                source_id=item.source_input_id,
                source_digest=item.source_content_sha256,
                content=_json_text(item.to_dict()["payload"]),
            ))
        return evidence

    @staticmethod
    def _memory_evidence(results) -> list[EvidenceItem]:
        evidence = []
        for match in results:
            record = match.record
            evidence.append(EvidenceItem(
                evidence_id=record.memory_id,
                source_kind="memory",
                source_id=record.source_id,
                source_digest=record.source_digest,
                content=_json_text(record.to_dict()["content"]),
                score=float(match.score),
            ))
        return evidence

    def _decide(self, task: CognitiveTask, evidence: tuple[EvidenceItem, ...]) -> CognitiveDecision:
        candidates = task.candidate_actions
        if not candidates:
            return CognitiveDecision(
                DecisionStatus.ABSTAINED, None, 0.0,
                "No candidate actions were supplied; cognition will not invent an action.",
            )
        if not evidence:
            return CognitiveDecision(
                DecisionStatus.ABSTAINED, None, 0.0,
                "No admissible evidence was available for action selection.",
                tuple((item, 0.0) for item in candidates),
            )

        evidence_terms = set()
        for item in evidence:
            evidence_terms.update(_tokens(item.content))
        constraint_terms = set().union(*(_tokens(x) for x in task.constraints)) if task.constraints else set()
        scores = []
        for candidate in candidates:
            terms = _tokens(candidate)
            if not terms:
                score = 0.0
            else:
                overlap = len(terms & evidence_terms) / len(terms)
                # Constraints are explicit task input; matching terms modestly
                # increase traceability but never substitute for evidence.
                constraint_match = len(terms & constraint_terms) / len(terms) if constraint_terms else 0.0
                score = min(1.0, 0.85 * overlap + 0.15 * constraint_match)
            scores.append((candidate, round(score, 6)))
        ranked = sorted(scores, key=lambda pair: (-pair[1], pair[0]))
        top_action, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - runner_up
        if top_score < self.min_evidence_score or margin < self.min_decision_margin:
            reason = (
                "Evidence overlap is below the configured threshold."
                if top_score < self.min_evidence_score
                else "Top candidate is insufficiently separated from alternatives."
            )
            return CognitiveDecision(
                DecisionStatus.ABSTAINED, None, top_score, reason, tuple(scores)
            )
        return CognitiveDecision(
            DecisionStatus.SELECTED, top_action, top_score,
            "Selected from caller-supplied candidates using lexical overlap with provenance-linked evidence; this is a heuristic score, not calibrated probability.",
            tuple(scores),
        )

    def analyze(self, result: CognitiveResult, query: str, **subsystem_inputs):
        """Run the opt-in subsystem pipeline over a completed cognition result."""
        from .integration import analyze_cognition
        if not isinstance(result, CognitiveResult):
            raise TypeError("result must be a CognitiveResult")
        return analyze_cognition(result, query, **subsystem_inputs)

    def process(self, task: CognitiveTask, *, representations: Iterable = (),
                additional_evidence: Iterable[EvidenceItem] = (),
                retrieve_memory: bool = True, namespace: str = "default",
                memory_limit: int = 20) -> CognitiveResult:
        if not isinstance(task, CognitiveTask):
            raise TypeError("task must be a CognitiveTask")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be non-empty")
        if type(memory_limit) is not int or not 1 <= memory_limit <= 1000:
            raise ValueError("memory_limit must be an integer between 1 and 1000")

        evidence = []
        warnings = []
        for result in tuple(representations):
            evidence.extend(self._representation_evidence(result))
        supplied_evidence = tuple(additional_evidence)
        if any(not isinstance(item, EvidenceItem) for item in supplied_evidence):
            raise TypeError("additional_evidence must contain EvidenceItem values")
        evidence.extend(supplied_evidence)

        query = task.query or task.objective
        if retrieve_memory and self.memory_engine is not None:
            from importlib import import_module
            contracts = import_module("05_MEMORY.contracts")
            matches = self.memory_engine.retrieve(
                query,
                contracts.MemoryQuery(namespace=namespace, limit=memory_limit),
            )
            evidence.extend(self._memory_evidence(matches))
        elif retrieve_memory and self.memory_engine is None:
            warnings.append("Memory retrieval requested but no MemoryEngine is configured.")

        # Preserve deterministic evidence ordering and avoid duplicate IDs.
        unique = {}
        for item in evidence:
            previous = unique.get(item.evidence_id)
            if previous is not None and previous != item:
                raise ValueError(f"conflicting evidence identity: {item.evidence_id}")
            unique.setdefault(item.evidence_id, item)
        evidence_tuple = tuple(unique.values())
        plan = tuple(PlanStep(index, step) for index, step in enumerate(task.plan_steps))
        decision = self._decide(task, evidence_tuple)

        if not evidence_tuple:
            status = CognitiveStatus.INSUFFICIENT_EVIDENCE
        elif not task.candidate_actions:
            status = CognitiveStatus.NO_CANDIDATES
        else:
            status = CognitiveStatus.READY
        return CognitiveResult(
            task_id=task.task_id, status=status, evidence=evidence_tuple,
            plan=plan, decision=decision, warnings=tuple(warnings),
        )
