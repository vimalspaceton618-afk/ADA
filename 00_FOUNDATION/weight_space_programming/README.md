# WSPPI — Weight-Space Programming and Parameter Intelligence

**Status:** Proposed research subsystem; specification only  
**Parent:** ADA

WSPPI investigates how ADA can inspect, program, transform, compose, validate, and evolve neural parameter spaces. It unifies two capabilities:

1. **Weight-space programming:** explicit, typed, reproducible operations on model parameters and parameter deltas.
2. **Controlled self-evolution:** using evaluation evidence to propose parameter changes, test them in isolation, and promote only validated artifacts.

## Documents

- [Architecture](WSPPI_ARCHITECTURE.md)
- [Research specification](WSPPI_RESEARCH_SPECIFICATION.md)
- [Weight Program Language](WEIGHT_PROGRAM_LANGUAGE.md)
- [Safety and governance](WSPPI_SAFETY_GOVERNANCE.md)

## Non-negotiable invariant

**Production weights are immutable during ordinary inference.** WSPPI creates candidate artifacts in isolated workspaces. A candidate can become a release only after integrity checks, capability and regression evaluations, safety review, and explicit authorized approval. The model itself cannot grant that approval.

## Core terms

- **Weight program:** declarative specification of parameter-space operations and constraints.
- **Weight transformation:** deterministic or explicitly stochastic operation on an artifact or parameter delta.
- **Candidate artifact:** isolated result that has not been promoted.
- **Self-evolution proposal:** candidate transformation proposed by ADA against a defined objective and evidence.
- **Promotion:** governed change of the approved model artifact reference, not an in-place mutation.
