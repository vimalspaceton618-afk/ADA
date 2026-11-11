"""Analytical helpers for scaling and parameter-budget calculations."""
from __future__ import annotations
import math

def chinchilla_tokens(active_parameters: int, *, tokens_per_parameter: float = 20.0):
    if type(active_parameters) is not int or active_parameters < 1: raise ValueError("active_parameters must be positive")
    if not math.isfinite(tokens_per_parameter) or tokens_per_parameter <= 0: raise ValueError("ratio must be positive")
    return int(active_parameters * tokens_per_parameter)

def active_moe_parameters(expert_parameters: int, experts: int, top_k: int):
    if min(expert_parameters, experts, top_k) < 1 or top_k > experts: raise ValueError("invalid MoE dimensions")
    return expert_parameters * top_k

def total_to_active_ratio(total_parameters: int, active_parameters: int):
    if total_parameters < 1 or active_parameters < 1 or active_parameters > total_parameters:
        raise ValueError("invalid parameter counts")
    return total_parameters / active_parameters

__all__ = ["chinchilla_tokens", "active_moe_parameters", "total_to_active_ratio"]
