"""Integrity validation for provenance and source-grounded evidence."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import PerceptionInput, PerceptionResult, PerceptionStatus, content_digest


@dataclass(frozen=True, slots=True)
class PerceptionValidationReport:
    valid: bool
    input_id: str
    checks: tuple[str, ...]
    errors: tuple[str, ...]


class PerceptionValidator:
    """Checks provenance and structural invariants, not semantic truth."""

    def validate(self, item: PerceptionInput, result: PerceptionResult) -> PerceptionValidationReport:
        if not isinstance(item, PerceptionInput) or not isinstance(result, PerceptionResult):
            raise TypeError("validate requires PerceptionInput and PerceptionResult")
        checks: list[str] = []
        errors: list[str] = []

        if result.input_id == item.input_id:
            checks.append("input_id_matches")
        else:
            errors.append("result input_id does not match source input")

        if result.modality is item.modality:
            checks.append("modality_matches")
        else:
            errors.append("result modality does not match source input")

        if result.content_sha256 == content_digest(item.content):
            checks.append("source_digest_matches")
        else:
            errors.append("result digest does not match source content")

        if result.correlation_id == item.correlation_id:
            checks.append("correlation_id_matches")
        else:
            errors.append("correlation_id does not match source input")

        observation = result.observation
        if observation is None:
            if result.status in {PerceptionStatus.OBSERVED, PerceptionStatus.PARTIAL}:
                errors.append("observed or partial result is missing its observation")
            else:
                checks.append("no_observation_consistent_with_status")
        else:
            if observation.source_input_id != item.input_id:
                errors.append("observation provenance does not match source input")
            else:
                checks.append("observation_provenance_matches")
            if observation.content_sha256 != result.content_sha256:
                errors.append("observation digest does not match result digest")
            else:
                checks.append("observation_digest_matches")

        return PerceptionValidationReport(
            valid=not errors,
            input_id=item.input_id,
            checks=tuple(checks),
            errors=tuple(errors),
        )
