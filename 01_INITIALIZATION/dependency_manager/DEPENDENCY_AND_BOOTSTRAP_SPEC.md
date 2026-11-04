# Dependency and Bootstrap Specification

**Status:** Proposed

## Principles

- Runtime startup is verification-only: it MUST NOT run pip install, download arbitrary code, or mutate the environment.
- Development and production dependency sets MUST be separate.
- Pin versions for reproducible experiments; use lockfiles with hashes where supported.
- Keep optional packages optional and test each integration independently.
- Do not import cloud-provider or notebook packages from core runtime modules.

## Dependency groups

| Group | Examples | Required when |
|---|---|---|
| Core | Python standard library, PyTorch or JAX backend | model commands |
| Data | datasets, tokenizer library, parquet/JSON readers | data ingestion/training |
| Numerics | NumPy and scientific utilities | reference math/diagnostics |
| Visualization | Matplotlib, optional Seaborn | dashboards |
| Acceleration | Triton, CUDA extensions, Mamba kernels, XLA/Pallas | matching backend/component |
| Distributed | torch.distributed, FSDP/ZeRO, JAX distributed | distributed training |
| Cloud/notebook | Colab, Drive helpers | notebook-only workflows |

Exact versions MUST be selected in backend-specific lockfiles after compatibility testing. This specification intentionally does not invent versions.

## Verification report

For each required package, record distribution and import names, installed version, expected capability probe, result and whether required/optional/incompatible. A successful import does not prove CUDA kernel, XLA custom-call or distributed collective functionality; run minimal probes.

## Installation policy

Development may use documented setup commands or a container. Production should use a previously built, tested environment image. Dependency updates require lockfile changes, tests and a changelog entry.

## Notebook migration findings

The supplied notebook export mixes PyTorch, Hugging Face Datasets/Transformers, requests, Matplotlib, Seaborn, Graphviz, PyMuPDF and Colab helpers. Shell magics and notebook paths are present. Dependencies belong in optional groups, not one mandatory runtime requirements file.

## Network and trust

Dataset/tokenizer downloads MUST use pinned revisions or immutable manifests where available. Verify checksums. Never execute remote repository code during initialization. Network access is disabled by default for inference unless explicitly required.
