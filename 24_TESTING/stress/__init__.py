"""Bounded concurrency stress helper with deterministic iteration counts."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True, slots=True)
class StressReport:
    tasks: int
    successes: int
    failures: tuple[str, ...]

def run_concurrent(operation: Callable[[int], Any], *, tasks: int, workers: int):
    if not callable(operation) or type(tasks) is not int or tasks < 1 or type(workers) is not int or workers < 1:
        raise ValueError("callable, positive tasks and workers required")
    failures, successes = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(operation, i) for i in range(tasks)]
        for future in as_completed(futures):
            try: future.result(); successes += 1
            except Exception as exc: failures.append(f"{type(exc).__name__}: {exc}")
    return StressReport(tasks, successes, tuple(failures))

__all__ = ["StressReport", "run_concurrent"]
