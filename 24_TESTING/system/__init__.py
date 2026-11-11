"""System-level smoke checks across importable ADA domains."""
from __future__ import annotations
import importlib

REQUIRED_PACKAGES = (
    "02_CORE_RUNTIME", "03_PERCEPTION", "04_REPRESENTATION", "05_MEMORY",
    "06_COGNITION", "07_WORLD_MODEL", "08_LEARNING", "09_NEURAL_SUBSTRATE",
    "10_DEVELOPMENT_ENGINE", "11_REASONING_AND_PLANNING", "12_ACTION_SYSTEM",
    "13_EXTERNAL_KNOWLEDGE", "14_EVALUATION", "15_SAFETY_AND_GOVERNANCE",
    "16_DATA_INFRASTRUCTURE", "17_TRAINING", "18_INFERENCE", "19_INTERACTION",
    "20_DISTRIBUTED_INFRASTRUCTURE", "21_OBSERVABILITY", "22_INTEROPERABILITY",
    "23_DEPLOYMENT", "24_TESTING", "25_RESEARCH", "26_DOCUMENTATION", "27_OPERATIONS",
)

def import_all_domains():
    loaded, failures = [], []
    for name in REQUIRED_PACKAGES:
        try: importlib.import_module(name); loaded.append(name)
        except Exception as exc: failures.append((name, f"{type(exc).__name__}: {exc}"))
    return tuple(loaded), tuple(failures)

__all__ = ["REQUIRED_PACKAGES", "import_all_domains"]
