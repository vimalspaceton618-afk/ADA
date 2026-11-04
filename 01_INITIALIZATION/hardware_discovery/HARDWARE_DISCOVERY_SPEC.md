# Hardware Discovery Specification

**Status:** Proposed

## Discovery dimensions

| Domain | Measurements | Why it matters |
|---|---|---|
| CPU | cores, architecture, SIMD, NUMA | input pipeline, CPU fallback, offload |
| Host memory | total, available, cgroup limit | data loading, checkpoint staging |
| GPU | model, VRAM total/free, compute capability | model/activation/optimizer placement |
| GPU topology | NVLink/PCIe, device count, NUMA affinity | tensor/expert communication |
| TPU | chip count, HBM, ICI topology, XLA version | sharding and collective feasibility |
| Storage | free bytes, measured throughput | datasets, checkpoints, temporary files |
| Network | interface/bandwidth only when measured and authorized | distributed training/data streaming |

## Inventory vs feasibility

Discovery answers what is present. Resource allocation answers whether a declared workload can safely run. Hardware presence is not an admission decision.

## Workload estimate inputs

Consume total parameter count and parameter dtype; optimizer and state precision; gradient precision and sharding/offload; activation estimate or measured peak; sequence length, microbatch, accumulation and checkpointing; attention cache, SSM state and memory-bank tensors; temporary workspace, communication buffers and safety reserve.

Use conservative estimates. Prefer representative warm-up measurements over theoretical constants where available.

## Paper-derived target inventory

The paper discusses A100/H100/H200, TPU v5e and MI300X comparisons, and a proposed 4× H100 arrangement. These are planning references, not detected resources or measured performance. The uploaded notebook does not establish a validated multi-accelerator training run.

## Required report

Produce machine-readable inventory with timestamp, tool source, device identifiers (redacted in ordinary logs), per-device memory, software stack, topology and probe outcomes. Retain raw report alongside run config.
