# 00 — Foundation

**Status:** Research baseline  
**Maturity:** Foundational specification; implementation pending  
**Scope:** Identity, principles, mathematical vocabulary, hypotheses, contracts, system specification, and shared terminology.

This directory is the constitutional and technical starting point for ADA. It defines what the project is trying to build, the constraints under which it may evolve, and the evidence required before a proposal becomes an engineering decision.

## Documents

| Area | Document | Purpose |
|---|---|---|
| Identity | [Project identity](identity/PROJECT_IDENTITY.md) | Mission, scope, non-goals, intended users, and definition of success |
| Principles | [Design principles](design_principles/DESIGN_PRINCIPLES.md) | Normative rules for architecture, research, safety, and operations |
| Mathematics | [Mathematical foundations](mathematical_foundations/MATHEMATICAL_FOUNDATIONS.md) | Notation and mathematical building blocks used across the project |
| Research | [Research hypotheses](research_hypotheses/RESEARCH_HYPOTHESES.md) | Falsifiable claims, evidence requirements, and experiment gates |
| Contracts | [Architecture contracts](architecture_contracts/ARCHITECTURE_CONTRACTS.md) | Interfaces, invariants, compatibility, and failure behavior |
| Specification | [System specification](system_specification/SYSTEM_SPECIFICATION.md) | System-level view and proposed architecture boundaries |
| Language | [Terminology](terminology/TERMINOLOGY.md) | Canonical terms and meanings |
| Weight-space intelligence | [WSPPI](weight_space_programming/README.md) | Parameter-space programming and controlled self-evolution |

## Decision hierarchy

1. **Verified constraints** — reproducible facts, resource limits, safety requirements, and interface guarantees.
2. **Accepted decisions** — choices approved after review and recorded with rationale.
3. **Active hypotheses** — plausible research directions that require experiments.
4. **Exploratory ideas** — concepts not yet mature enough to constrain implementation.

A lower-level item must never be presented as a higher-confidence item. An experiment can promote or reject a hypothesis; ambition alone cannot.

## Change control

Every material architectural change should record: problem, alternatives, expected benefit, cost, risks, experiment or evidence, compatibility impact, rollback plan, owner, and status. Use an Architecture Decision Record (ADR) for decisions with long-lived consequences.

## Current baseline

ADA is an experimental research program for adaptive, efficient, reliable intelligent systems. The working research direction combines state-space sequence modeling, selective attention, sparse expert computation, domain-specialized components, and controlled orchestration. No architecture in this document is claimed to be optimal or implemented merely because it is specified.

## What this foundation does not claim

- It does not claim that ADA is already generally intelligent, autonomous in an unrestricted sense, or superhuman.
- It does not claim that any proposed model size, token budget, layer ratio, or benchmark target guarantees capability.
- It does not authorize unrestricted online learning, self-modification, tool execution, or deployment.
- It does not replace empirical validation, security review, legal review, or human approval.

**North star:** build systems whose capabilities are measurable, whose changes are inspectable, whose behavior is controllable, and whose efficiency is demonstrated rather than assumed.
