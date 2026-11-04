# ADA Terminology

Canonical meanings for terms used in specifications, code, experiments, and reports. When a term has multiple technical meanings, use the definition below or qualify it.

| Term | ADA meaning |
|---|---|
| ADA | Project identifier for this adaptive-intelligence research program |
| Adaptive system | A system that changes behavior or configuration through a defined, controlled mechanism |
| State-space model (SSM) | A sequence model using latent state transitions and readout |
| Mamba-like block | A selective state-space sequence-processing block; exact recurrence and implementation must be specified |
| Mamba³ | ADA working label for a proposed next-generation/custom state-space research direction; not an assertion of equivalence to a published implementation |
| Attention | Content-addressing operation over query, key, and value representations |
| GQA | Grouped-query attention, where groups of query heads share key/value heads |
| Dynamic sparse GQA | Proposed attention that dynamically restricts attended positions or entries; selection algorithm must be specified |
| RoPE | Rotary positional embedding applied to query/key representations in supported attention designs |
| MoE | Mixture of Experts: a model layer that routes inputs to a subset of expert networks |
| Top-k routing | Selecting k experts according to router scores, subject to defined tie-breaking and capacity behavior |
| Expert capacity | Maximum token load an expert can accept in a routing batch or dispatch unit |
| Active parameters | Parameters used for a particular input or token; report calculation method |
| Total parameters | All stored trainable model parameters, including inactive experts |
| Dense model | A model that activates its relevant parameter path for each input rather than sparsely selecting experts |
| Specialist model | A model trained and evaluated for a bounded domain or task family |
| General model | A model intended to cover a broader task distribution; this is a scope label, not a capability guarantee |
| Fusion | Must be qualified as parameter merging, hidden-state/feature fusion, logit ensembling, routing, distillation, or orchestration |
| Router | A component selecting a model, expert, tool, or processing path using a declared policy |
| Orchestrator | A component coordinating bounded multi-step tasks and component interactions |
| Agent | A system that observes context, plans, and takes bounded actions through approved interfaces |
| Tool | An external capability invoked through a permissioned interface |
| Memory | Persisted information retrieved or updated outside ordinary model-weight inference |
| Online learning | Updating model parameters during or near live operation; disabled by default in this project baseline |
| Retrieval-augmented generation (RAG) | Generation supported by retrieval of external documents or records |
| Calibration | Agreement between confidence estimates and observed correctness under a stated distribution |
| Hallucination | Unsupported or fabricated output presented as factual |
| Data provenance | Record of source, rights, collection, transformations, and lineage of data |
| Contamination | Unintended overlap or exposure between training data and evaluation material |
| Baseline | A reference system or method used for controlled comparison |
| Ablation | Experiment removing or changing one design element to estimate its contribution |
| FLOPs | Floating-point operation count; counting convention must be stated |
| Tokens per second | Throughput metric; disclose batch size, sequence settings, hardware, and measurement boundary |
| Time to first token | Time from request submission to first generated token under a stated serving setup |
| ADR | Architecture Decision Record documenting a significant decision and its rationale |
| Validated | Passed predefined tests under a documented evaluation protocol |
| Production-ready | Meets capability, reliability, security, operational, and maintenance release requirements |

## Naming rules

- Distinguish **model**, **system**, **agent**, **tool**, and **product**.
- Distinguish **parameter count** from **active parameters** and **compute**.
- Distinguish **memory retrieval** from **parameter learning**.
- Distinguish a **research hypothesis** from a **decision** and an implemented feature.
- Use exact benchmark names and versions; do not use informal labels as evidence.
- When citing another architecture, identify the source and explain which parts are independently implemented, adapted, or merely conceptually related.
