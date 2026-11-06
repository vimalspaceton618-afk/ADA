# ADA Cognition

06_COGNITION is the bounded orchestration layer between 05_MEMORY and 07_WORLD_MODEL. It consumes explicit tasks, provenance-preserving representations and optional memory retrieval, then emits evidence, a caller-authored plan and an auditable decision record.

## v1 scope

- Immutable task, evidence, plan, decision and result contracts.
- Representation handoff validates the upstream `RepresentationResult` type and preserves source input ID and source SHA-256.
- Optional lexical retrieval through the existing `MemoryEngine`, respecting its namespace filter and lifecycle rules.
- Deterministic evidence collection and duplicate evidence-ID suppression.
- Caller-supplied candidate ranking using transparent lexical overlap.
- Explicit abstention when evidence is missing, weak or ambiguous.
- Structured result status, candidate scores, rationale and evidence provenance.

## Quick start

```python
from ada.cognition import CognitionEngine, CognitiveTask
from ada.memory import MemoryEngine

memory = MemoryEngine()
cognition = CognitionEngine(memory_engine=memory)

task = CognitiveTask(
    objective="Choose the next documented maintenance action",
    query="maintenance inspection",
    constraints=("do not operate unverified equipment",),
    candidate_actions=("inspect the cooling system", "restart the equipment"),
    plan_steps=("collect inspection evidence", "review safety constraints"),
)
result = cognition.process(task, retrieve_memory=True)
print(result.status, result.decision.status, result.decision.rationale)
```

Supply representation results from `RepresentationEngine.transform(...)` through the `representations` argument. The engine accepts only explicit `CognitiveTask` candidates and plan steps; it does not invent task intent or execute plans.

## Decision semantics

The v1 score is a lexical heuristic over candidate terms and the union of evidence terms, with a small explicit-constraint match contribution. It is not a probability, learned reasoning score, semantic similarity metric or benchmark result. Selection requires both a minimum score and a minimum margin over the runner-up. Ties and weak matches abstain. Configure thresholds on `CognitionEngine`.

## Boundaries and non-capabilities

- No LLM inference, chain-of-thought generation or hidden reasoning claims.
- No autonomous task decomposition; plan steps are supplied by the caller.
- No tool execution, external side effects, approvals or action verification.
- No world-state prediction, simulation or causal model; those belong to 07_WORLD_MODEL.
- No learned semantic retrieval, embedding model, vector database or persistent memory.
- No automatic memory writes, fact extraction, memory consolidation or conflict resolution.
- No safety guarantee: heuristic candidate selection must not authorize hazardous or consequential actions.

## Verification

Tests cover input validation, provenance preservation, namespace-scoped memory retrieval, deterministic scoring, tie/low-evidence abstention, no-candidate behavior and caller-authored plan ordering. Passing tests verifies software contracts, not general intelligence or reasoning quality.
