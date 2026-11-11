"""Accelerator inventory descriptors and optional PyTorch discovery."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class AcceleratorDescriptor:
    device_id: str
    backend: str
    name: str
    memory_bytes: int | None = None
    capability: str | None = None
    def __post_init__(self):
        if not self.device_id.strip() or not self.backend.strip() or not self.name.strip():
            raise ValueError("accelerator identity fields required")
        if self.memory_bytes is not None and self.memory_bytes < 0: raise ValueError("memory_bytes must be non-negative")

def discover_torch_devices():
    try:
        import torch
    except ImportError:
        return (AcceleratorDescriptor("cpu:0", "cpu", "CPU"),)
    result = [AcceleratorDescriptor("cpu:0", "cpu", "CPU")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            result.append(AcceleratorDescriptor(f"cuda:{i}", "cuda", props.name, int(props.total_memory)))
    return tuple(result)

__all__ = ["AcceleratorDescriptor", "discover_torch_devices"]
