# ADA

ADA is an experimental research project exploring adaptive intelligent systems through explicit, testable architecture boundaries.

## Repository status

The repository contains implemented research components alongside planned domains. A directory or design document is not evidence of a trained or production-ready capability. Consult each domain's README and implementation record for its exact scope and verification status.

## Architecture map

- `03_PERCEPTION` — source observation and modality-aware perception contracts.
- `04_REPRESENTATION` — immutable, typed representations with source provenance.
- `05_MEMORY` — provenance-aware record memory and differentiable associative memory.
- `06_COGNITION` — bounded task orchestration, evidence aggregation, caller-defined planning, heuristic candidate ranking, explicit abstention, integrated subsystems and gated strategy-level RSI.
- `07_WORLD_MODEL` — downstream world-state modelling domain; implementation status is tracked separately.
- `09_NEURAL_SUBSTRATE` — experimental neural components and model assembly.

Stable Python facades are exposed under `ada.*` where implemented. Optional neural components require the `neural` extra.

## Official Colab pretraining notebook

[![Open ADA Pretraining in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vimalspaceton618-afk/ADA/blob/main/notebooks/colab/ADA_Pretraining.ipynb)

**Use this one notebook for ADA Colab training:** `notebooks/colab/ADA_Pretraining.ipynb`. It checks the GPU, clones the latest `main` branch, installs the pretraining dependencies, mounts Google Drive, previews real FineWeb-Edu records, launches causal language-model pretraining, plots train/validation loss, and verifies saved checkpoints.

The training script streams the English educational `sample-10BT` configuration from `HuggingFaceFW/fineweb-edu`, trains a byte-level BPE tokenizer on training-only documents, and initializes ADA from random weights. It uses a deterministic held-out validation split, next-token cross-entropy plus the MoE load-balancing term, periodic validation, and resumable optimizer/RNG checkpoints.

The default pilot processes 512,000 next-token targets (500 steps × batch 4 × sequence 256). This is a bounded real-data experiment, not frontier-scale training or evidence of broad model capability. FineWeb-Edu is English-only, not a multilingual or specialist corpus. Its dataset card declares ODC-By 1.0 and notes that Common Crawl terms also apply; preserve attribution and review those terms before redistribution. Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Local CUDA execution is also supported:

```bash
pip install -e ".[pretraining]"
python scripts/ada_pretrain_fineweb_edu.py --device cuda --output-dir ./artifacts/ada-fineweb-edu --max-steps 500
```

## Research discipline

- Treat architectural capabilities as hypotheses until implemented and evaluated.
- Keep learning and structural changes versioned, measurable, reversible, and permission-controlled.
- Preserve data provenance, licensing, privacy constraints, and evaluation results.
- Distinguish deterministic baselines from learned capabilities.
- Do not describe ADA as human-level or superintelligent without reproducible evidence.

See `CONTRIBUTING.md` for contribution guidelines.


## Infrastructure reference layers (15-27)

The remaining numbered domains now contain executable standard-library
reference contracts and automated tests: safety policy and human approval;
dataset manifests and quality gates; training lifecycle and checkpoints;
model registry and bounded inference adapters; conversation and feedback;
node scheduling and leases; observability events, metrics and spans; plugin
allowlists and serialization; deployment release state and rollback; test
runner aggregation; research protocols and result provenance; documentation
catalog and evidence-backed ADRs; and incident, backup and maintenance
lifecycles.

Import these surfaces through ada.systems. They are tested control-plane and
integration baselines, not proof of trained model capability, production
isolation, persistent distributed services, or live deployment.

The 15–27 nested feature packages are importable through ada.systems. Their
current implementation scope, test coverage, and remaining external adapters
are tracked in
[26_DOCUMENTATION/REMAINING_SUBSYSTEMS_BASELINE.md](26_DOCUMENTATION/REMAINING_SUBSYSTEMS_BASELINE.md).
