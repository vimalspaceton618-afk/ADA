"""Cross-component contract graph checks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable

@dataclass(frozen=True, slots=True)
class IntegrationCheck:
    check_id: str
    upstream: str
    downstream: str
    run: Callable[[], bool]
    def __post_init__(self):
        if any(not x.strip() for x in (self.check_id, self.upstream, self.downstream)) or not callable(self.run):
            raise ValueError("integration check fields and runner required")

@dataclass(frozen=True, slots=True)
class IntegrationReport:
    passed: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]
    @property
    def success(self): return not self.failed

def run_integration_checks(checks: Iterable[IntegrationCheck]):
    passed, failed = [], []
    for check in checks:
        if not isinstance(check, IntegrationCheck): raise TypeError("checks must be IntegrationCheck")
        try:
            result = check.run()
            if result is True: passed.append(check.check_id)
            else: failed.append((check.check_id, "returned false"))
        except Exception as exc:
            failed.append((check.check_id, f"{type(exc).__name__}: {exc}"))
    return IntegrationReport(tuple(passed), tuple(failed))

__all__ = ["IntegrationCheck", "IntegrationReport", "run_integration_checks"]
