"""Validated parallelism configuration and PyTorch distributed adapter."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ParallelismConfig:
    mode: str = "ddp"
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    def __post_init__(self):
        if self.mode not in {"ddp", "tp", "ep", "dp_ep"}: raise ValueError("unsupported parallelism mode")
        if self.tensor_parallel_size < 1 or self.expert_parallel_size < 1: raise ValueError("parallel sizes must be positive")

class TorchDistributedAdapter:
    def __init__(self):
        try:
            import torch.distributed as dist
        except ImportError as exc:
            raise RuntimeError("PyTorch distributed support is not installed") from exc
        self.dist = dist
    def require_initialized(self):
        if not self.dist.is_available() or not self.dist.is_initialized():
            raise RuntimeError("torch.distributed process group is not initialized")
        return self.dist.get_world_size(), self.dist.get_rank()

__all__ = ["ParallelismConfig", "TorchDistributedAdapter"]
