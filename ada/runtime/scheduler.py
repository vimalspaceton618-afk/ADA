"""Compatibility facade for the canonical FIFO scheduler."""

from importlib import import_module

FIFOScheduler = import_module("02_CORE_RUNTIME.task_scheduler.fifo").FIFOScheduler
__all__ = ["FIFOScheduler"]
