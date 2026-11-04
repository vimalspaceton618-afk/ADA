# Mathematical Foundations

**Status:** Shared notation and conceptual baseline  
**Purpose:** Establish a consistent mathematical language. This document is not a complete proof of any proposed architecture.

## 1. Notation

| Symbol | Meaning |
|---|---|
| (x_t) | Input at sequence position (t) |
| (h_t) | Latent or recurrent state at position (t) |
| (y_t) | Model output at position (t) |
| (d) | Hidden representation width |
| (L) | Number of network layers |
| (N) | Number of parameters or model components, as stated explicitly |
| (T) | Sequence length |
| (B) | Batch size |
| (E) | Number of experts |
| (k) | Number of selected experts per token |
| \\(\theta\\) | Trainable parameters |
| \\(\mathcal{D}\\) | Dataset or data distribution |
| \\(\mathcal{L}\\) | Training or evaluation loss |

Use explicit subscripts when a quantity is layer-, head-, token-, or expert-specific.

## 2. Learning objective

For parameters \\(\theta\\), a generic empirical objective is:

\[
\min_{\theta}\; \mathbb{E}_{(x,y)\sim\mathcal{D}}[\mathcal{L}(f_{\theta}(x),y)] + \lambda\,\Omega(\theta)
\]

Here \\(\Omega\\) is an optional regularizer and \\(\lambda\\) its coefficient. This expression does not prescribe a single loss function; objectives differ for language modeling, classification, retrieval, control, and multimodal tasks.

For autoregressive sequence modeling:

\[
\mathcal{L}_{\mathrm{NLL}}(\theta) =
-\frac{1}{T}\sum_{t=1}^{T}
\log p_{\theta}(x_t\mid x_{<t})
\]

Loss must be paired with task-relevant behavioral evaluations. Lower token loss does not guarantee better tool use, factuality, reasoning, or calibration.

## 3. State-space sequence modeling

A general discrete state-space form is:

\[
h_t = A_t h_{t-1} + B_t u_t, \qquad
y_t = C_t h_t + D_t u_t
\]

where (u_t) is the input, (h_t) is state, and (A_t,B_t,C_t,D_t) define state transition and readout. Selective or input-dependent variants make some parameters functions of the input.

Implementation details such as discretization, stability constraints, scan algorithms, numerical precision, state initialization, and sequence parallelism must be specified by the chosen model. The label **Mamba³** in ADA notes is a research direction name, not a claim that an established external implementation has been reproduced exactly. Any custom Mamba³-like block must have a mathematical and executable specification.

## 4. Attention

Scaled dot-product attention is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}+M\right)V
\]

where (M) represents masking or attention bias. Grouped-query attention shares key/value heads across groups of query heads. Sparse attention restricts the set of keys or positions considered.

Any sparse-attention proposal must specify its sparsity pattern, causal behavior, approximation error, mask semantics, and complexity. It must be evaluated against a correct dense reference.

## 5. Mixture of experts

A sparse expert layer can be represented as:

\[
\operatorname{MoE}(x)=\sum_{i\in\operatorname{TopK}(g(x),k)}p_i(x)E_i(x)
\]

where (g) is a router, (E_i) is expert (i), and (p_i) is its selected weight. The router must define normalization, tie-breaking, capacity, overflow behavior, and token dropping or fallback behavior.

Expert balance is not guaranteed by top-k routing. Auxiliary objectives or routing constraints may be used, but must be measured for their effect on quality, stability, and overhead.

## 6. Hybrid composition

A candidate hybrid block may compose state-space mixing, attention, and feed-forward or expert transformations:

\[
z^{(l)} = \operatorname{Norm}(x^{(l)})
\]
\[
u^{(l)} = \operatorname{SequenceMixer}_{l}(z^{(l)})
\]
\[
v^{(l)} = \operatorname{Norm}(x^{(l)}+u^{(l)})
\]
\[
x^{(l+1)} = x^{(l)}+u^{(l)}+\operatorname{FFN/MoE}_{l}(v^{(l)})
\]

This is a schematic pattern only. Residual topology, normalization placement, parallel versus sequential branches, gating, and dimensional projections must be made explicit in the concrete architecture specification.

## 7. Conditional compute and cost

Report both total parameter count and active parameter count. For sparse models, active compute depends on routing, token distribution, expert capacity, and communication.

A practical cost model should include:

\[
C_{\mathrm{total}} =
C_{\mathrm{compute}}+
C_{\mathrm{memory}}+
C_{\mathrm{communication}}+
C_{\mathrm{routing}}+
C_{\mathrm{serving}}
\]

These terms are conceptual accounting categories, not necessarily directly additive hardware counters. Actual measurement must use wall-clock and resource telemetry.

## 8. Domain specialization and composition

A collection of specialist models does not automatically form a better general model. Composition requires explicit decisions about:

- Shared versus separate tokenizers and representation spaces.
- Routing policy and confidence thresholds.
- Whether outputs are logits, hidden states, structured tool results, or natural-language messages.
- Calibration, disagreement handling, and conflict resolution.
- Distillation or joint training, where appropriate.
- Evaluation of the composed system against each component and a single-model baseline.

The term **fusion** must be qualified: parameter merging, feature fusion, logit ensembling, routing, distillation, and agent orchestration are different methods.

## 9. Evaluation mathematics

For a metric (m), report the estimate, sample size, confidence interval where applicable, and evaluation protocol. For paired comparisons, use matched examples and an appropriate paired statistical procedure. Avoid interpreting small score differences without uncertainty analysis.

Efficiency comparisons should report quality at a fixed resource budget and resource use at a fixed quality threshold where possible.

## 10. Open mathematical work

The following require formal derivation and experiments before adoption:

1. Stability and memory behavior of the proposed Mamba³-like recurrence.
2. Complexity and approximation properties of dynamic sparse GQA.
3. Router load balance, expert specialization, and communication cost.
4. Interaction between recurrent state and attention memory.
5. Training stability for hybrid blocks at target sequence lengths.
6. Calibration and error propagation across specialist model routing.
7. Scaling behavior across parameters, data, tokens, and compute.

No unproved scaling law or complexity claim should be used as an implementation guarantee.
