# Configuration Test Plan

The first executable configuration implementation is accepted only when tests cover:

- Valid minimal executable config.
- Mapping and YAML parsing equivalence.
- Stable canonical JSON and SHA-256 digest.
- Missing required sections and keys.
- Unknown root and nested keys.
- Booleans rejected where integer dimensions are required.
- Zero/negative values rejected.
- GQA divisibility.
- MoE top-k bounds.
- Explicit backend/device values.
- Research profile rejected for execution.
- Missing PyYAML produces a clear configuration error.

Tests must not imply that the model architecture or runtime has been implemented.
