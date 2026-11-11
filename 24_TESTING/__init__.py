"""Deterministic test registry and reliability result aggregation."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TestDefinition:
    test_id: str
    runner_id: str
    input: Mapping[str, Any]
    expected: Any
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.test_id.strip() or not self.runner_id.strip():
            raise ValueError("test_id and runner_id are required")


@dataclass(frozen=True, slots=True)
class TestCaseResult:
    test_id: str
    status: TestStatus
    duration_seconds: float
    output_digest: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TestRun:
    run_id: str
    results: tuple[TestCaseResult, ...]

    @property
    def passed(self):
        return all(x.status is TestStatus.PASSED for x in self.results)

    @property
    def counts(self):
        return {status.value: sum(x.status is status for x in self.results)
                for status in TestStatus}


class TestRegistry:
    def __init__(self):
        self._runners = {}
        self._lock = threading.RLock()

    def register(self, runner_id: str, runner: Callable[[Mapping[str, Any]], Any]):
        if not runner_id.strip() or not callable(runner):
            raise ValueError("runner_id and callable runner are required")
        with self._lock:
            if runner_id in self._runners:
                raise ValueError("runner_id already registered")
            self._runners[runner_id] = runner

    def resolve(self, runner_id: str):
        with self._lock:
            return self._runners[runner_id]


class TestHarness:
    def __init__(self, registry: TestRegistry):
        if not isinstance(registry, TestRegistry):
            raise TypeError("registry must be TestRegistry")
        self.registry = registry

    def run(self, definitions: Sequence[TestDefinition]):
        import uuid
        results = []
        for definition in definitions:
            if not isinstance(definition, TestDefinition):
                raise TypeError("definitions must contain TestDefinition")
            started = time.monotonic()
            try:
                output = self.registry.resolve(definition.runner_id)(definition.input)
                status = TestStatus.PASSED if output == definition.expected else TestStatus.FAILED
                raw = json.dumps(output, sort_keys=True, default=str,
                                 separators=(",", ":")).encode("utf-8")
                output_digest = hashlib.sha256(raw).hexdigest()
                error = None
            except Exception as exc:
                status, output_digest = TestStatus.ERROR, None
                error = f"{type(exc).__name__}: {exc}"
            results.append(TestCaseResult(
                definition.test_id, status, time.monotonic() - started,
                output_digest, error,
            ))
        return TestRun(str(uuid.uuid4()), tuple(results))


__all__ = [
    "TestStatus", "TestDefinition", "TestCaseResult", "TestRun",
    "TestRegistry", "TestHarness",
]
