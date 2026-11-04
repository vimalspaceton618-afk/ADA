# WSPPI Research Specification

**Status:** Proposed research plan  
**Primary question:** Can parameter-space operations reliably modify or compose model capabilities under measurable and governable constraints?

## 1. Objectives

1. Establish reliable inspection and provenance for parameter artifacts.
2. Determine which operations are useful under explicit compatibility constraints.
3. Measure whether adapter and delta composition adds target capabilities while preserving unrelated capabilities.
4. Investigate whether automated systems can diagnose weaknesses and propose useful parameter updates.
5. Detect regressions, unsafe behavioral shifts, and evaluator overfitting.
6. Compare parameter updates against data, retrieval, prompt, tool, and architecture interventions.

## 2. Two research tracks

**Track A — Weight-space programming:** a researcher or approved system specifies a transformation; WSPPI compiles and executes it. Research asks whether the operation produces the intended measurable behavior.

**Track B — Controlled self-evolution:** ADA analyzes approved evaluation evidence, proposes a weight program, and tests candidate updates iteratively. Research asks whether proposal generation can identify improvements that generalize beyond observed failures.

Track B uses Track A infrastructure and receives no broader permissions.

## 3. Falsifiable hypotheses

### H-WSP-01 — Bounded parameter deltas
Small constrained deltas can adapt a task while preserving defined general capabilities.
Test full fine-tuning, LoRA/adapters, scaled deltas, and no-update baselines under matched data and compute.
Falsifier: gains fail to generalize, protected capabilities regress, or costs outweigh benefits.

### H-WSP-02 — Adapter composition
Compatible adapters can support multiple tasks with less storage or deployment cost than separate full checkpoints, without unacceptable interference.
Test independent adapters, sequential and weighted composition, routing, and joint training.
Falsifier: interference or routing overhead removes the benefit.

### H-WSP-03 — Weight interpolation and merging
Some compatible checkpoints can be interpolated or merged to improve a declared metric.
Test raw interpolation, alignment-aware merging, task-vector methods, and ensembling, including incompatible negative controls.
Falsifier: results are unstable or no better than simpler baselines.

### H-WSP-04 — Automated proposal quality
A constrained proposal generator can outperform random, heuristic, or researcher-authored baselines on held-out objectives.
Use blind candidate evaluation and compare no-change, bounded random, standard fine-tuning, and human proposals.
Falsifier: gains disappear on held-out tasks or exploit the evaluator.

### H-WSP-05 — Regression prediction
Artifact statistics and behavioral probes can predict some harmful transformations before expensive full evaluation.
Correlate layer-wise delta features and activation shifts with full-suite regressions.
Falsifier: predictors fail to generalize or miss critical regressions.

### H-WSP-06 — Iterative controlled evolution
A bounded propose-test-select loop can outperform a one-shot update under a fixed compute and evaluation budget.
Compare iterative proposals with one-shot fine-tuning, standard hyperparameter search, and no-change controls.
Falsifier: iteration overfits, destabilizes behavior, or yields no advantage at equal budget.

### H-WSP-07 — Non-weight interventions
Some capability failures are better addressed through retrieval, tool correction, data quality, or orchestration than parameter modification.
For each diagnosed failure, compare a parameter update with a relevant non-weight intervention.

## 4. Research questions

- What makes a parameter program more than tensor arithmetic?
- How should parameter identity be tracked across architecture revisions?
- Which transformations are valid for each model family?
- How can neuron/channel permutation symmetries be aligned before merging?
- When do low-rank updates interfere?
- Can sensitivity or Fisher-style estimates guide update magnitude?
- How can evaluations detect regressions and hidden backdoors?
- Can a proposal model predict useful updates without exploiting evaluators?
- Which components must remain immutable across evolution cycles?

## 5. Experimental protocol

Every experiment must include a fixed base artifact and digest; unchanged baseline; task definition; train/validation/held-out separation; data provenance and contamination checks; pre-registered primary metric and threshold; protected regression suites; matched budgets where appropriate; multiple seeds or uncertainty estimates where feasible; complete transformation logs; and failure analysis.

Candidate selection uses validation data only. Final held-out tests must not be repeatedly used to tune transformations.

## 6. Metrics

**Parameter:** absolute/relative delta norms, per-layer magnitude, update density, distribution shift, artifact size, non-finite tensor count.

**Capability:** target task success, general capability retention, cross-domain transfer, calibration, abstention, robustness, tool-use and instruction-following regression.

**Operations:** transformation runtime, peak memory, accelerator utilization, evaluation cost, candidate success rate, rollback time, reproducibility rate.

**Governance:** provenance completeness, unauthorized operation rejection, audit completeness, review time, promotion and rollback correctness.

No single metric establishes safe or useful evolution.

## 7. Research phases and exit criteria

**Phase 0 — Artifact science:** manifests, integrity checks, tensor inspection, immutable storage. Exit when digest verification and schema mismatch detection work and workers cannot overwrite source artifacts.

**Phase 1 — Deterministic operations:** inspection, delta statistics, bounded scaling, adapter application, compatibility checks. Exit when unit, numerical, and reproducibility tests pass on a small model.

**Phase 2 — Candidate evaluation:** protected regression suites, candidate reports, human-reviewed promotion. Exit when failed or inconclusive candidates cannot be promoted and rollback works.

**Phase 3 — Composition research:** adapter composition, task vectors, interpolation, alignment, merging. Exit with reproducible value against matched baselines, or a documented negative result.

**Phase 4 — Proposal generation:** ADA proposes typed WPL programs from evaluation evidence. Execution and promotion remain outside model authority. Exit when proposal quality is measured against human, heuristic, random, and no-change baselines on held-out objectives.

**Phase 5 — Controlled self-evolution:** bounded iterative loops with immutable releases, human approval, and rollback. Exit only after repeatable held-out gains without unacceptable regressions or governance failures.

## 8. Required baselines

No parameter change; standard supervised fine-tuning; LoRA or another parameter-efficient baseline; separate specialist checkpoint or routing; retrieval/tool intervention where relevant; human-authored program; random or heuristic bounded proposal.

## 9. Stop conditions

Stop if artifact integrity fails, data rights are unclear, resource limits are exceeded, safety or protected regressions cross limits, contamination is suspected, the proposal exploits evaluation, audit/rollback is unavailable, or authorized review is missing.

## 10. Expected outputs

Versioned artifact and tensor-manifest specification; WPL schema and compiler; transformation library; compatibility tests; evaluation suite; experiment reports; proposal-generation benchmark; safety/governance tests; and an evidence-based decision on whether self-evolution should progress beyond research prototypes.

A negative result is a valid research output. Retain WSPPI only where it demonstrates value over simpler methods.
