# ADA Learning

`08_LEARNING` is the controlled adaptation plane between Memory, Cognition,
World Model and the neural training substrate. It captures experience,
curates reusable evidence, builds candidate updates and promotes only
independently evaluated improvements.

The core package is framework-agnostic. PyTorch, JAX, NeMo/Megatron and other
training backends are optional adapters and are not imported by this folder's
core contracts.

## Architecture

```text
03_PERCEPTION -> 04_REPRESENTATION -> 05_MEMORY
                                         |
                                         v
07_WORLD_MODEL -----------------> Experience Ledger
                                         |
                                         v
                                  Quality / Provenance
                                         |
                                         v
                                   Replay Buffer
                                         |
                                         v
                                  Learning Proposal
                                         |
                                         v
                              Candidate Artifact Builder
                                         |
                                         v
                              Held-out Evaluation Gate
                                  /             \
                               reject          promote
                                 |                |
                              retain          Artifact Registry
                              parent          (external storage)
```

## Implemented modules

- `contracts.py`: immutable experience, proposal, metric, artifact and decision contracts.
- `experience_learning/ledger.py`: bounded process-local evidence ledger with explicit candidate, validate and reject transitions.
- `replay_mechanism/buffer.py`: bounded deterministic stratified replay over validated records.
- `learning_evaluation/gates.py`: direction-aware metric policies and regression gates; evaluation identity and dataset digest must remain fixed.
- `learning_controller/controller.py`: proposal/evaluation/promotion orchestration. It only changes the active artifact descriptor after the evaluator passes.
- `ada.learning`: stable public facade.

## Integration boundaries

### Memory

Use `MemoryEngine` to retain provenance-bearing long-term records. The Learning
ledger is a task-oriented, bounded intake/review queue; it is not a replacement
for durable memory or a second source of truth. The caller passes an
`ExperienceRecord` with source identifiers and the original source SHA-256.

### World Model

World Model `EpisodeTrajectory` provides the episode-safe observation/action
sequence contract. Convert accepted trajectory windows into learning records
with the original episode digest and episode/task identity. Never shuffle
transitions across episode boundaries.

### Cognition and RSI

Cognition's bounded RSI controller optimizes declarative strategy parameters.
Learning provides the broader evidence and model-artifact lifecycle. They are
complementary: Learning does not bypass RSI's bounds, the proposal parent digest,
or any safety policy.

### Neural Substrate

The core Learning layer does not import neural dependencies. A future training
adapter can map replay samples to PyTorch/JAX batches, compute loss, write a
candidate checkpoint and return a content digest. This layer then evaluates
that candidate. No optimizer step, checkpoint write, source-code mutation or
deployment is executed by this initial implementation.

## Learning modes and roadmap

The current milestone establishes the common evidence/control plane. The
following directories remain extension points until their algorithms and
benchmarks are implemented:

| Domain | Future implementation |
|---|---|
| `supervised_learning` | batch construction, loss adapters, supervised trainer |
| `self_supervised_learning` | masking/contrastive/predictive objectives |
| `reinforcement_learning` | offline/online policy optimization and reward contracts |
| `imitation_learning` | behavior-cloning and trajectory filtering |
| `observation_learning` | unsupervised environment/representation adaptation |
| `continual_learning` | task stream, replay/regularization/distillation |
| `transfer_learning` | adapter/LoRA transfer and domain adaptation |
| `meta_learning` | rapid adaptation and task-distribution evaluation |
| `curriculum_learning` | difficulty scheduling and mastery progression |
| `forgetting_control` | old-task retention, drift and rollback checks |

These modes are not claimed as completed algorithms in this milestone.

## Research grounding

The implementation uses design principles from:

- [Lifelong Learning of Large Language Model based Agents: A Roadmap](https://arxiv.org/abs/2501.07278): treats perception, memory, action and evaluation as connected parts of a lifelong agent.
- [Continual Learning for Large Language Models: A Survey](https://arxiv.org/abs/2402.01364): separates continual pretraining, instruction tuning and alignment.
- [Lifelong Language Pretraining with Distribution-Specialized Experts](https://arxiv.org/abs/2305.12281): motivates domain/task specialization and expert-aware retention.
- [Offline Experience Replay for Continual Offline Reinforcement Learning](https://arxiv.org/abs/2305.13804): motivates replay under distribution shift and task sequences.
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): motivates replay sampling by transition relevance; the current baseline uses explicit importance and stratification rather than TD-error priorities.
- [Rethinking Self-Evolving Agent Skills: Feedback Dynamics over Multiple Rounds](https://arxiv.org/abs/2608.02636): motivates sparse, validation-filtered skill revisions and storing failed as well as successful trajectories.

Hugging Face repositories were inspected as implementation references, including
`makltd1/continual-lm`, `Vir007/continual-ai-2b` and the Continual Learning
Exam dataset. These are reference artifacts, not runtime dependencies.
CLE Lite is a benchmark with 20 professional tasks and five connected levels
per task; confirm dataset license terms before redistribution or benchmark
reuse. The small synthetic RL instruction dataset inspected had Dataset Viewer
generation failures and was not adopted as a source of truth.

## Safety, reliability and non-goals

- Experience is a candidate until an explicit validation call.
- Only validated experience can enter replay.
- Provenance is mandatory; source digests are content identifiers, not proof of truth.
- Replay capacity is bounded and eviction is deterministic.
- Baseline/candidate evaluation must use the same evaluator identity, dataset
  digest and sample count.
- Every configured metric is checked with an explicit higher/lower-is-better
  direction and regression tolerance.
- No metric is assumed to be a proxy for safety unless explicitly configured.
- The controller only promotes a descriptor after the gate passes.
- No code execution, tool invocation, external action, model-weight update,
  or deployment is performed by this module.
- Stores are process-local; durable persistence, encryption, distributed
  replay and audit export are deferred.

## Verification

`tests/test_learning.py` covers immutable provenance contracts, explicit
validation, duplicate identity, bounded ledger capacity, replay eligibility,
stratified sampling, eviction, evaluation identity, metric direction,
regression rejection, accepted promotion and stale-parent protection.
