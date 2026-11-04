# Configuration Specification

**Status:** Proposed typed configuration contract

## Configuration precedence

Lowest to highest: version-controlled defaults; named profile; explicit config file; command-line overrides; allowlisted ADA_ environment variables.

Secrets MUST be supplied through a secret manager or runtime injection. They MUST NOT be stored in profiles, committed YAML, command history, or logs. Resolved configuration provenance is recorded with secrets redacted.

## Required sections

| Section | Fields | Validation |
|---|---|---|
| project | name, version, run_id, stage | identifiers and enum |
| model | architecture_id, vocab_size, d_model, n_layers, max_seq_len | positive dimensions and implementation compatibility |
| tokenizer | id, revision, vocab_size, special_token_ids | embedding/head vocabulary agreement |
| sequence | mixer schedule, state size, causal mode | explicitly supported implementation |
| attention | query heads, KV heads, RoPE theta, sparse fraction | divisibility and mask semantics |
| moe | experts, top_k, hidden dimension, routing | 1 ≤ top_k ≤ experts |
| memory | enabled, slots, memory width, temperature | positive dimensions and declared read/write semantics |
| training | precision, seed, batch, sequence, optimizer, schedule | valid ranges and backend support |
| data | manifests, split, packing, masking policy | provenance and split validation |
| runtime | backend, device, workers, compile mode | explicit availability and compatibility |
| resources | host/device memory limits, reserve fraction | budget bounds and admission |
| checkpoint | path, expected architecture digest, resume policy | integrity and compatibility |
| logging | level, output directory, redaction | safe writable location |

## Cross-field invariants

- d_model must be divisible by n_heads for equal-width attention heads.
- n_heads must be divisible by n_kv_heads for GQA.
- 0 < sparse_keep ≤ 1 and 0 < mod_capacity ≤ 1.
- 1 ≤ moe.top_k ≤ moe.n_experts.
- Tokenizer vocabulary and embedding/head vocabulary MUST match.
- Sequence length MUST not exceed declared implementation and positional-encoding limits.
- A Mamba-2 profile MUST resolve to a real Mamba-2 implementation, not a convolutional placeholder.
- TensorNet linear attention MUST resolve to its explicit tested implementation and supported scan path.
- Attention backend MUST declare causal and padding-mask behavior.
- BF16 MUST be rejected when selected device/backend does not support it adequately; no silent substitution.
- Checkpoint architecture, tokenizer and tensor-shape manifests MUST match before loading.

## Research profile vs executable profile

The paper-derived v4.5 profile is a reference of intended values. An executable run must select an implementation profile declaring component status, tests, backend and supported limits. Unsupported paper components cause rejection or an explicit experiment-specific override.

## Configuration identity

Canonicalize the fully resolved non-secret configuration and compute a SHA-256 digest. Store it in every run record and checkpoint metadata. Different data manifests, tokenizer revisions, model shapes, precision or optimizer schedules define different experiments.
