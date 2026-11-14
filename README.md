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

## One-click GPU validation

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vimalspaceton618-afk/ADA/blob/main/notebooks/colab/ADA_GPU_Smoke_Test.ipynb)

Open the notebook, select a GPU runtime, and choose **Runtime → Run all**. It clones the current `main` branch, installs `.[neural]`, then executes:

```bash
python scripts/ada_gpu_smoke_test.py --device cuda --steps 3 --batch-size 2 --sequence-length 64
```

The smoke test exercises the integrated causal LM with alternating Mamba-2 reference SSD and causal attention layers, sparse MoE routing, Energy Memory, backward gradients, an AdamW parameter update, and inference. It uses synthetic token IDs and writes a JSON report plus a checkpoint bundle under `artifacts/ada_gpu_smoke_test/`.

A PASS confirms only that the tested configuration executed and updated parameters on the selected device. It is not a convergence, quality, throughput-comparison, or production-kernel benchmark. The Mamba-2 path is a correctness-oriented PyTorch reference, not a fused CUDA/Triton implementation.

## Real-data pretraining (Colab)

[![Open ADA FineWeb-Edu pretraining in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vimalspaceton618-afk/ADA/blob/main/notebooks/colab/ADA_Real_Data_Pretraining_FineWeb_Edu.ipynb)

The first bounded pretraining recipe streams the English educational `sample-10BT` configuration from `HuggingFaceFW/fineweb-edu`, trains a byte-level BPE tokenizer on training-only documents, and trains `ADAForCausalLM` from random initialization. It uses a deterministic held-out validation split, next-token cross-entropy plus the MoE load-balancing term, periodic validation, resumable optimizer/RNG checkpoints, and a run manifest.

Install the optional dependencies and run locally on CUDA:

```bash
pip install -e ".[pretraining]"
python scripts/ada_pretrain_fineweb_edu.py --device cuda --output-dir ./artifacts/ada-fineweb-edu --max-steps 500
```

The default pilot processes 512,000 next-token targets (500 steps × batch 4 × sequence 256). This is a real-data pretraining experiment, but not frontier-scale training or evidence of broad model capability. FineWeb-Edu is English educational web text, not a multilingual or specialist corpus. Its dataset card declares ODC-By 1.0 and notes that Common Crawl terms also apply; preserve attribution and review those terms before redistribution. Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

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
