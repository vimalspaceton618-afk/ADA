"""Evaluation-gated validation adapter for the development engine."""

from __future__ import annotations

from importlib import import_module
from typing import Any


class EvaluationBackedValidator:
    """Validate a development proposal using a registered evaluator and gate.

    The proposal digest is the evaluated subject digest. A proposal is marked
    as passing only when the evaluator's report matches the requested
    evaluator, dataset, and proposal and the configured EvaluationGate passes.
    """

    def __init__(self, *, registry: Any, evaluator_id: str, dataset: Any, gate: Any):
        evaluation = import_module("14_EVALUATION")
        if not isinstance(registry, evaluation.EvaluatorRegistry):
            raise TypeError("registry must be an EvaluatorRegistry")
        if not isinstance(dataset, evaluation.EvaluationDataset):
            raise TypeError("dataset must be an EvaluationDataset")
        if not isinstance(gate, evaluation.EvaluationGate):
            raise TypeError("gate must be an EvaluationGate")
        if not isinstance(evaluator_id, str) or not evaluator_id.strip():
            raise ValueError("evaluator_id must be non-empty")
        self.registry = registry
        self.evaluator_id = evaluator_id
        self.dataset = dataset
        self.gate = gate

    def __call__(self, proposal: Any) -> dict[str, Any]:
        from .development_engine import ChangeProposal

        if not isinstance(proposal, ChangeProposal):
            raise TypeError("proposal must be a ChangeProposal")

        report = self.registry.evaluate(
            self.evaluator_id, self.dataset, proposal.digest
        )
        result = self.gate.evaluate(report)
        return {
            "status": result.verdict.value,
            "reason": result.reason,
            "evaluator_id": report.evaluator_id,
            "subject_digest": report.subject_digest,
            "dataset_digest": report.dataset_digest,
            "report_digest": report.digest,
            "sample_count": report.sample_count,
            "metrics": {metric.name: metric.value for metric in report.metrics},
            "missing_metrics": result.missing_metrics,
            "failed_metrics": result.failed_metrics,
            "limitations": report.limitations,
        }
