# Configuration Implementation Contract

**Status:** First executable foundation  
**Scope:** Runtime configuration parsing and validation only.

## Goals

- Parse configuration from a mapping or YAML file.
- Validate required model dimensions and architecture constraints.
- Reject unsupported or ambiguous configuration rather than silently changing it.
- Produce a canonical, stable JSON representation and SHA-256 digest.
- Keep research-target profiles non-executable unless explicitly converted into a supported implementation profile.

## Initial executable profile

The initial runtime supports a deliberately small, explicit configuration surface:

- model: vocabulary size, model width, layer count, maximum sequence length
- attention: query-head and key/value-head counts
- MoE: expert count and top-k
- runtime: backend and device selection

This is not an implementation of the ADA v4.5 paper architecture. Features such as Mamba2/3 selective scan, dynamic sparse GQA, mixture-of-depths, continuous external memory, and multi-token prediction remain unsupported until their implementations and tests exist.

## Validation invariants

1. Integer dimensions must be positive integers; booleans are not accepted as integers.
2. GQA requires query heads to be divisible by key/value heads.
3. MoE requires 1 <= top_k <= n_experts.
4. Device selection is explicit. No implicit CPU/CUDA/XLA fallback occurs.
5. Unknown keys are rejected to catch misspellings and unsupported features.
6. The paper-derived `ADA_V45_RESEARCH.yaml` profile remains `execution_allowed: false`.
7. Canonical serialization sorts keys and emits stable JSON; the digest is SHA-256 over UTF-8 canonical JSON.

## API

- `ADAConfig.from_mapping(mapping)`
- `ADAConfig.from_yaml(path)`
- `ADAConfig.canonical_json()`
- `ADAConfig.digest()`

YAML loading requires PyYAML. Importing the package does not require PyYAML; attempting YAML loading without it raises an actionable `ConfigurationError`.

## Explicit non-goals

No hardware probing, package installation, resource admission, training, model construction, checkpoint loading, or runtime startup is performed here. These are separate initialization stages and must not be inferred from configuration validity.
