# Environment Detection Specification

**Status:** Proposed

## Purpose

Report actual runtime capabilities. Detection distinguishes hardware presence, framework visibility, backend compatibility and workload feasibility.

## Host inventory

Collect OS/version/architecture and kernel; Python implementation/version and executable; CPU model, cores and instruction features when discoverable; host RAM total/available and process limits; filesystem free space; container/cgroup limits; framework and compiler/runtime versions. Unknown fields are recorded as unknown, not guessed.

## Accelerator inventory

### NVIDIA/CUDA
Record device name, compute capability, total/free memory, driver, CUDA runtime, visible-device mask, and PyTorch CUDA availability. Successful torch.cuda.is_available() does not validate custom kernels.

### AMD/ROCm
Record device identity, ROCm runtime and framework build compatibility. Do not classify ROCm as CUDA.

### Google TPU/XLA
Record XLA availability, topology, runtime versions and supported execution path. The research paper's TPU proposal requires a JAX/Flax + Pallas-compatible scan implementation; a CUDA-only Triton kernel is not a TPU backend.

### CPU fallback
CPU execution is allowed only for profiles explicitly permitting it and within stated sequence, batch and memory limits. CPU fallback is suitable for correctness smoke tests; it does not imply useful training throughput.

## Device selection

1. Honor explicit device/backend request.
2. Validate backend and dtype support.
3. Query resource inventory.
4. Run minimal allocation and operation probe.
5. Select only when all required probes pass.

Never select the first visible accelerator without considering free memory, device limits, interconnect, implementation support and workload size.

## Output schema

```json
{
  "host": {"os": "recorded", "python": "recorded", "ram_bytes": null},
  "accelerators": [],
  "frameworks": [],
  "selected_backend": null,
  "selected_device": null,
  "probes": [],
  "detected_at": "RFC3339 timestamp"
}
```

## Security

Hardware discovery MUST NOT enumerate unrelated user files, inspect credentials, or transmit inventory externally. Redact hostnames and identifiers from public reports unless disclosure is authorized.
