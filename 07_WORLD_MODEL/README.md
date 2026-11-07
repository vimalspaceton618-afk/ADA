# ADA World Model

07_WORLD_MODEL is the state-and-dynamics boundary after perception, representation, memory, and cognition.

## Responsibility split
- 03_PERCEPTION produces source-grounded observations.
- 04_REPRESENTATION produces typed projections while preserving source identity and digest.
- 05_MEMORY stores provenance-aware records over time.
- 06_COGNITION orchestrates explicit evidence, candidate hypotheses, and decisions.
- 07_WORLD_MODEL maintains a time-indexed state and evaluates possible state transitions.

Cognition may request a prediction. The world model does not decide the user's objective, select or execute actions, or silently turn uncertain observations into facts.

## v1 implementation
- WorldFact: immutable subject-predicate-value assertion with source kind, source ID, SHA-256 digest, observation time, optional expiry, and confidence.
- WorldSnapshot: immutable point-in-time fact set with content digest and active-fact filtering.
- WorldAction: caller-supplied hypothetical action descriptor; it is never executed.
- TransitionRule: explicit declarative preconditions and effects.
- WorldModelEngine: monotonic observation ingestion, bounded snapshot history, rule matching, prediction, and isolated simulation.

## Research basis and adapter targets
- DreamerV3: recurrent state-space latent dynamics and imagination-based model-predictive control. https://github.com/danijar/dreamerv3
- V-JEPA 2: representation-space video prediction and action-conditioned planning. https://github.com/facebookresearch/vjepa2
- NVIDIA Cosmos: multimodal, action-conditioned physical-world prediction and simulation. https://github.com/NVIDIA/cosmos
- TD-MPC2: compact latent dynamics and trajectory optimization. https://github.com/nicklashansen/tdmpc2

Hugging Face discovery identified V-JEPA 2 checkpoints such as facebook/vjepa2-vitl-fpc64-256 and world-model datasets including NVIDIA PhysicalAI synthetic driving and physical-interaction collections. These are research/data references, not runtime dependencies. No checkpoints or third-party code are copied into this milestone.

## Prediction semantics
The v1 baseline evaluates only caller-authored transition rules. It is not a learned P(s[t+h] | s[t], a) distribution. Confidence is the minimum declared confidence among the rule, matched evidence, and predicted effects; it is a conservative provenance bound, not calibrated probability.

- No action rule: NO_MODEL.
- Preconditions fail: INAPPLICABLE.
- Multiple rules match: AMBIGUOUS and abstain.
- Exactly one rule matches: emit predicted facts tied to the base snapshot digest.
- Predictions are hypothetical and isolated; only observations update current state.

## Invariants and limitations
- No mandatory third-party dependencies and no GPU required.
- Facts require explicit provenance and finite confidence in [0, 1].
- Snapshot and fact timestamps must be timezone-aware.
- Observation time is monotonic; expired facts are not carried forward.
- State values are immutable finite JSON-compatible structures.
- Prediction never mutates the current snapshot or history.
- No fact inference from raw image, video, text, or memory data.
- No action execution, physics guarantee, learned causal model, or safety-critical control authorization.

## Verification
tests/test_world_model.py covers provenance validation, immutable values, time ordering, expiry, state updates, prediction preconditions, ambiguity abstention, uncertainty bounds, simulation isolation, and public imports.
