# Resource Allocation Policy

**Status:** Proposed conservative admission policy

## Principles

- No workload starts until memory/resource admission passes.
- Reserve explicit headroom for fragmentation, runtime workspaces, communication buffers and system processes.
- Never assume all advertised VRAM/HBM is allocatable.
- Prefer measured peak memory from representative runs.
- Never silently reduce dimensions, sequence length, number of experts, precision or batch size.

## Parameter memory

For P parameters stored in b bytes each:

`weight_bytes = P × b`

This estimates weights only. Training also needs gradients, optimizer states, activations, temporary buffers and runtime overhead. An AdamW estimate must state moment dtype and whether master weights, gradients, optimizer sharding or offload are used.

## Admission calculation

```text
available_device_memory
  >= weights
   + gradients
   + optimizer_state
   + activations
   + workspaces
   + communication_buffers
   + safety_reserve
```

Return ADMIT, ADMIT_WITH_EXPLICIT_OVERRIDE, or REJECT, with every term and uncertainty recorded. An override is never automatic and must be approved by the run owner.

## Workload classes

- unit/smoke: tiny shapes; imports and correctness only.
- component verification: isolated forward/backward checks.
- short-run experiment: small corpus slice and bounded steps, with checkpoint/resume.
- ablation: controlled comparison at fixed compute/data budgets.
- scale run: distributed training after measured throughput, recovery and cost estimates.

## Practical development mode

A CPU-only laptop can support architecture development, unit tests, config validation, tokenizer/data logic and tiny shape tests. It should not be assumed to support the paper's production profile. If no supported accelerator is available, initialization rejects accelerator-required jobs.

## OOM policy

Classify out-of-memory errors at workload boundary, clean temporary allocations and record failed config. A retry uses a new, explicitly approved config and run record. Do not silently resume with altered shapes.
