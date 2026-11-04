# ADA Research Hypotheses

**Status:** Active research register — none of the claims below are validated by this document.  
**Rule:** Every hypothesis needs a baseline, measurable outcome, resource budget, and a result that could falsify it.

## H1 — Hybrid sequence modeling

**Claim:** A hybrid architecture combining a selective state-space sequence mixer with strategically placed attention may improve the quality/efficiency trade-off for long-context workloads over suitable dense-attention baselines.

**Rationale:** State-space layers may provide efficient sequence processing, while attention may provide direct content-based retrieval at selected positions.

**Test:** Compare state-space-only, attention-only, and hybrid models under matched parameter and compute budgets. Evaluate language modeling, long-context retrieval, recall under distractors, throughput, latency, memory, and training stability.

**Falsifiers:** No meaningful quality/efficiency gain; attention placement creates bottlenecks; recurrent state degrades retrieval; or engineering overhead outweighs theoretical savings.

## H2 — Sparse attention placement

**Claim:** A small number of carefully placed attention layers may recover important content-addressing behavior without requiring attention in every layer.

**Candidate configuration:** The previously discussed 24-layer layout with 21 Mamba³-like layers and 3 attention layers is a starting experiment only.

**Test:** Sweep attention locations and counts, including uniform, early, middle, late, and learned or task-informed placement. Compare at matched compute and training data.

**Falsifiers:** Dense or more frequent attention consistently performs better at acceptable cost, or sparse placement causes unacceptable degradation on defined tasks.

## H3 — Dynamic sparse GQA

**Claim:** A dynamic sparse grouped-query attention mechanism may reduce memory and compute while preserving task-relevant retrieval.

**Test:** Define a deterministic dense reference, then compare sparse patterns, selection overhead, approximation error, quality, memory bandwidth, and serving latency.

**Falsifiers:** Dynamic selection overhead cancels savings; critical tokens are repeatedly omitted; or quality loss exceeds the predefined tolerance.

## H4 — Mixture-of-experts computation

**Claim:** Sparse expert layers may increase model capacity without proportional per-token compute growth, provided routing and communication are efficient.

**Test:** Measure expert utilization, token distribution, router entropy, overflow, dropped tokens, all-to-all traffic, active FLOPs, and downstream quality. Compare against dense feed-forward baselines.

**Falsifiers:** Expert collapse, severe imbalance, communication bottlenecks, unstable training, or no advantage at equal resource budgets.

## H5 — Domain-specialist portfolio

**Claim:** Several narrow models trained for clearly separated domains may provide stronger capability per unit resource than one similarly resourced general model on those domains.

**Candidate domains from project discussions:** cybersecurity (defensive and authorized), complex coding, mathematics, physics, and biology.

**Test:** Build clean domain datasets and held-out evaluations; compare each specialist with a general baseline, multi-task baseline, and routed portfolio. Measure cross-domain leakage and calibration.

**Falsifiers:** Specialists overfit, fail to transfer, create routing errors, or cost more to train and serve than the achieved gains justify.

## H6 — Specialist composition

**Claim:** A well-calibrated router or composition layer may combine specialists more reliably than naive output concatenation or majority voting.

**Test:** Compare routing, logit ensembling, structured delegation, distillation, and other explicitly named fusion strategies. Evaluate accuracy, calibration, latency, cost, and failure recovery.

**Falsifiers:** Composition increases error, latency, or complexity without a measurable advantage.

## H7 — Controlled adaptive memory

**Claim:** Versioned external memory and retrieval may improve continuity and task performance without modifying model weights during ordinary inference.

**Test:** Compare no memory, retrieval-only, structured memory, and bounded updates. Measure recall, stale-memory errors, privacy leakage, contamination, and deletion correctness.

**Falsifiers:** Memory causes unacceptable hallucination, leakage, stale-state errors, or operational burden.

## H8 — Compute-aware scaling

**Claim:** Architecture and data composition should be selected using measured quality under fixed budgets rather than parameter count alone.

**Test:** Record data tokens, data quality, model size, active compute, training FLOPs, hardware utilization, wall-clock duration, energy, and downstream quality across scale points.

**Falsifiers:** The proposed efficiency indicators do not predict downstream outcomes or are dominated by data and optimization effects.

## Experiment protocol

Every experiment record MUST include:

- Hypothesis ID and version.
- Question and expected mechanism.
- Baseline(s), matched constraints, and success threshold chosen before the run.
- Dataset manifest, provenance, license, filters, split, and contamination checks.
- Model configuration, code commit, dependency lock, and random seeds.
- Hardware, topology, precision, batch size, sequence length, and training schedule.
- Training cost, energy where available, memory, throughput, and failure logs.
- Evaluation protocol, confidence intervals where appropriate, and raw result location.
- Limitations, negative findings, conclusion, and next decision.

## Research queue

Priority order should be: correctness reference implementation; small-scale ablation; stability; reproducibility; efficiency measurement; then scaling. Do not begin expensive scaling runs until the corresponding small-scale experiment is informative and stable.


## WSPPI hypotheses — weight-space programming and controlled evolution

The WSPPI research register adds seven falsifiable hypotheses:

- **H-WSP-01:** bounded parameter deltas can adapt a task while retaining protected general capabilities.
- **H-WSP-02:** compatible adapters can be composed with lower storage or deployment cost and acceptable interference.
- **H-WSP-03:** compatible checkpoints may be interpolated or merged to improve a declared metric.
- **H-WSP-04:** constrained automated proposals can outperform random, heuristic, or human baselines on held-out objectives.
- **H-WSP-05:** parameter statistics and behavioral probes may predict some harmful transformations before full evaluation.
- **H-WSP-06:** bounded iterative propose-test-select loops may outperform one-shot updates at equal budgets.
- **H-WSP-07:** some failures are better addressed through retrieval, tools, data, or orchestration than weight changes.

These are hypotheses, not accepted findings. Protocols, falsifiers, metrics, baselines, and staged exit criteria are defined in [WSPPI Research Specification](../weight_space_programming/WSPPI_RESEARCH_SPECIFICATION.md).
