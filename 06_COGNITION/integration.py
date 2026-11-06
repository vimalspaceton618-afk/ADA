"""Opt-in integration pipeline for the eleven deterministic cognition subsystems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cognitive_controller import CognitiveController
from . import (
    attention_mechanism, abstraction_engine, concept_formation,
    relationship_engine, causal_reasoning, logical_reasoning,
    mathematical_reasoning, hypothesis_generation,
    uncertainty_estimation, metacognition,
)


@dataclass(frozen=True, slots=True)
class CognitionAnalysis:
    """Outputs from a subsystem pass; empty tuples mean no inputs were supplied."""
    focus: tuple
    abstractions: tuple
    concepts: tuple
    relations: tuple
    causal_reachable: tuple
    logical_results: tuple
    calculations: tuple
    hypotheses: tuple
    normalized_uncertainty: float | None
    audit: Any
    stage_trace: tuple


def analyze_cognition(
    result,
    query: str,
    *,
    focus_limit: int = 5,
    concepts=(),
    relations=(),
    causal_edges=(),
    causal_source: str | None = None,
    logical_premises=(),
    arithmetic=(),
    hypotheses=(),
) -> CognitionAnalysis:
    """Run supplied cognition subsystems without fabricating missing inputs.

    Structured inputs use tuples:
      concepts: (label, evidence_ids[, attributes])
      relations: (subject_id, predicate, object_id, evidence_id)
      logical_premises: (antecedent_bool, implication_bool)
      arithmetic: (left_number, operator, right_number)
      hypotheses: (statement[, supporting_evidence[, missing_evidence]])
    """
    from .contracts import CognitiveResult, EvidenceItem
    if not isinstance(result, CognitiveResult):
        raise TypeError("result must be a CognitiveResult")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    concepts = tuple(concepts)
    relations = tuple(relations)
    causal_edges = tuple(causal_edges)
    logical_premises = tuple(logical_premises)
    arithmetic = tuple(arithmetic)
    hypotheses = tuple(hypotheses)
    evidence_items = tuple(result.evidence)
    if any(not isinstance(item, EvidenceItem) for item in evidence_items):
        raise TypeError("result.evidence must contain EvidenceItem values")
    evidence_texts = tuple(item.content for item in evidence_items)
    focus_items = tuple((item.evidence_id, item.content) for item in result.evidence)

    def attention_stage(ctx):
        return {**ctx, "focus": attention_mechanism.select_focus(
            query, focus_items, limit=focus_limit
        )}

    def abstraction_stage(ctx):
        return {**ctx, "abstractions": abstraction_engine.extract_terms(evidence_texts)}

    def concept_stage(ctx):
        values = tuple(concept_formation.form_concept(*item) for item in concepts)
        return {**ctx, "concepts": values}

    def relationship_stage(ctx):
        values = tuple(relationship_engine.build_relation(*item) for item in relations)
        return {**ctx, "relations": values}

    def causal_stage(ctx):
        values = ()
        if causal_source is not None:
            values = causal_reasoning.causal_reachability(causal_edges, causal_source)
        elif causal_edges:
            raise ValueError("causal_source is required when causal_edges are supplied")
        return {**ctx, "causal_reachable": values}

    def logic_stage(ctx):
        values = tuple(logical_reasoning.modus_ponens(*item) for item in logical_premises)
        return {**ctx, "logical_results": values}

    def math_stage(ctx):
        values = tuple(mathematical_reasoning.calculate(*item) for item in arithmetic)
        return {**ctx, "calculations": values}

    def hypothesis_stage(ctx):
        values = []
        for item in hypotheses:
            if not isinstance(item, (tuple, list)) or not 1 <= len(item) <= 3:
                raise ValueError("each hypothesis must contain statement and optional supporting/missing evidence")
            statement = item[0]
            supporting = item[1] if len(item) >= 2 else ()
            missing = item[2] if len(item) == 3 else ()
            values.append(hypothesis_generation.propose(
                statement, supporting_evidence=supporting, missing_evidence=missing
            ))
        return {**ctx, "hypotheses": tuple(values)}

    def uncertainty_stage(ctx):
        scores = tuple(score for _, score in result.decision.candidate_scores)
        uncertainty = None
        if scores and sum(scores) > 0:
            uncertainty = uncertainty_estimation.normalized_entropy(scores)
        return {**ctx, "normalized_uncertainty": uncertainty}

    def metacognition_stage(ctx):
        return {**ctx, "audit": metacognition.audit(result)}

    controller = CognitiveController((
        attention_stage, abstraction_stage, concept_stage, relationship_stage,
        causal_stage, logic_stage, math_stage, hypothesis_stage,
        uncertainty_stage, metacognition_stage,
    ))
    trace = controller.run({})
    output = trace[-1].output if trace else {}
    return CognitionAnalysis(
        focus=output.get("focus", ()),
        abstractions=output.get("abstractions", ()),
        concepts=output.get("concepts", ()),
        relations=output.get("relations", ()),
        causal_reachable=output.get("causal_reachable", ()),
        logical_results=output.get("logical_results", ()),
        calculations=output.get("calculations", ()),
        hypotheses=output.get("hypotheses", ()),
        normalized_uncertainty=output.get("normalized_uncertainty"),
        audit=output.get("audit"),
        stage_trace=trace,
    )
