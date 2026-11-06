# ADA Cognition

06_COGNITION is the bounded cognitive processing layer between 05_MEMORY and 07_WORLD_MODEL. It consumes explicit tasks, provenance-preserving representations and optional memory retrieval, then emits evidence, caller-authored plans and auditable decisions.

## Subsystem map

Each directory is an independent implementation boundary with its public v1 baseline in `__init__.py`.

| Subsystem | Responsibility | Current implementation |
|---|---|---|
| `cognitive_controller/` | Coordinate cognitive stages | Ordered callable pipeline with stage trace |
| `attention_mechanism/` | Select relevant focus | Deterministic lexical ranking baseline |
| `abstraction_engine/` | Extract reusable abstractions | Frequency-ranked lexical term extraction |
| `concept_formation/` | Form concepts | Explicit concepts linked to evidence IDs |
| `relationship_engine/` | Represent relations | Typed subject-predicate-object evidence links |
| `causal_reasoning/` | Inspect causal structure | Reachability over caller-supplied directed edges |
| `logical_reasoning/` | Apply logical rules | Explicit propositional primitives |
| `mathematical_reasoning/` | Compute | Safe bounded arithmetic primitives |
| `hypothesis_generation/` | Propose explanations | Evidence-labelled hypothesis records |
| `uncertainty_estimation/` | Quantify score dispersion | Entropy and normalized entropy |
| `metacognition/` | Monitor outputs | Evidence, rationale, abstention and warning audit |

## Orchestration API

The root `contracts.py` and `engine.py` define the stable cognition task/result API. `ada.cognition` is the public Python facade. Cognition can optionally use `ada.memory.MemoryEngine` and accepts typed `RepresentationResult` values from 04_REPRESENTATION.

## Integrated subsystem pipeline\n\n`analyze_cognition(result, query, ...)` runs the subsystem sequence through `CognitiveController` and returns a `CognitionAnalysis` report with stage trace, focus ranking, lexical abstractions, evidence-linked concepts and relations, supplied-edge causal reachability, explicit logical results, bounded arithmetic results, caller-proposed hypotheses, normalized candidate-score entropy, and a metacognitive audit. Structured inputs are opt-in; absent inputs produce empty outputs rather than invented facts. The pipeline does not execute plans or external tools.\n\nExample:\n\n```python\nfrom ada.cognition import CognitionEngine, CognitiveTask, analyze_cognition\n\nresult = CognitionEngine().process(\n    CognitiveTask(objective="review evidence"), retrieve_memory=False\n)\nreport = analyze_cognition(result, "review evidence")\nprint(report.audit, report.stage_trace)\n```\n\n## Decision semantics

The v1 action score is a lexical heuristic over candidate terms and evidence terms, with a small explicit-constraint contribution. It is not a probability, learned reasoning score, semantic similarity metric or benchmark result. Selection requires a minimum score and margin over the runner-up. Ties and weak matches abstain.

## Important limits

- The subfolder implementations are transparent deterministic baselines, not trained neural reasoning systems.
- Attention and abstraction are lexical; they do not perform semantic understanding.
- Concept and relationship records require caller-supplied labels and evidence; they do not discover entities or relations.
- Causal reachability traverses supplied edges; it does not establish causation or discover causal graphs.
- Logical utilities implement only the documented propositional primitives, not a theorem prover.
- Mathematical reasoning currently provides safe scalar arithmetic, not symbolic algebra or a proof engine.
- Hypothesis generation stores caller-proposed hypotheses; it does not autonomously generate validated explanations.
- Entropy summarizes supplied non-negative scores; it does not calibrate model confidence.
- Metacognition reports structural quality signals; it does not introspect hidden model states.
- No subfolder executes external tools, performs side effects, authorizes hazardous actions, or guarantees safety.
- No world-state prediction or simulation is implemented here; that belongs to 07_WORLD_MODEL.

## Verification

The cognition test suite covers core orchestration and each subfolder's input/output contracts. Passing tests establishes implementation behavior only, not general intelligence or reasoning quality.
