# ADA

ADA is an experimental research project exploring adaptive intelligent systems through explicit, testable architecture boundaries.

## Repository status

The repository contains implemented research components alongside planned domains. A directory or design document is not evidence of a trained or production-ready capability. Consult each domain's README and implementation record for its exact scope and verification status.

## Architecture map

- `03_PERCEPTION` — source observation and modality-aware perception contracts.
- `04_REPRESENTATION` — immutable, typed representations with source provenance.
- `05_MEMORY` — provenance-aware record memory and differentiable associative memory.
- `06_COGNITION` — bounded task orchestration, evidence aggregation, caller-defined planning, heuristic candidate ranking, explicit abstention, integrated subsystems and gated strategy-level RSI.
- `07_WORLD_MODEL` — downstream world-state modelling domain; implementation status is tracked separately.
- `09_NEURAL_SUBSTRATE` — experimental neural components and model assembly.

Stable Python facades are exposed under `ada.*` where implemented. Optional neural components require the `neural` extra.

## Research discipline

- Treat architectural capabilities as hypotheses until implemented and evaluated.
- Keep learning and structural changes versioned, measurable, reversible, and permission-controlled.
- Preserve data provenance, licensing, privacy constraints, and evaluation results.
- Distinguish deterministic baselines from learned capabilities.
- Do not describe ADA as human-level or superintelligent without reproducible evidence.

See `CONTRIBUTING.md` for contribution guidelines.
