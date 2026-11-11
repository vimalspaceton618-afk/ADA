"""Hardened container command specification; Docker is not invoked."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ContainerSpec:
    image: str
    command: tuple[str, ...]
    user: str = "10001:10001"
    read_only_root: bool = True
    network_mode: str = "none"
    memory_mb: int = 4096
    cpu_quota: float = 2.0
    def __post_init__(self):
        if not self.image.strip() or not self.command: raise ValueError("image and command are required")
        if self.network_mode != "none": raise ValueError("network must be disabled by default")
        if not self.read_only_root: raise ValueError("root filesystem must be read-only")
        if self.memory_mb < 128 or self.cpu_quota <= 0: raise ValueError("invalid resource limit")

def docker_run_args(spec: ContainerSpec):
    if not isinstance(spec, ContainerSpec): raise TypeError("spec must be ContainerSpec")
    return ("docker", "run", "--rm", "--network", "none", "--read-only",
            "--user", spec.user, "--memory", f"{spec.memory_mb}m",
            "--cpus", str(spec.cpu_quota), spec.image, *spec.command)

__all__ = ["ContainerSpec", "docker_run_args"]
