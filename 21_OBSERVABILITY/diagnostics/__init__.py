"""Named diagnostic checks with explicit pass/fail outcomes."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    check_id: str
    passed: bool
    detail: str
    duration_seconds: float

class DiagnosticRunner:
    def __init__(self): self._checks = {}
    def register(self, check_id: str, check: Callable[[], Any]):
        if not check_id.strip() or not callable(check): raise ValueError("check ID and callable required")
        if check_id in self._checks: raise ValueError("duplicate diagnostic check")
        self._checks[check_id] = check
    def run(self):
        import time
        output = []
        for name, fn in sorted(self._checks.items()):
            start = time.monotonic()
            try:
                value = fn()
                passed = value is True
                detail = "passed" if passed else f"check returned {value!r}"
            except Exception as exc:
                passed, detail = False, f"{type(exc).__name__}: {exc}"
            output.append(DiagnosticResult(name, passed, detail, time.monotonic() - start))
        return tuple(output)

__all__ = ["DiagnosticResult", "DiagnosticRunner"]
