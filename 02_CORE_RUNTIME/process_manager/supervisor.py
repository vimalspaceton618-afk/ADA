"""Explicit subprocess supervisor; commands are never run through a shell."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ProcessRecord:
    process_id: int
    command: tuple[str, ...]
    started_at_monotonic: float
    return_code: int | None
    running: bool


class ProcessSupervisor:
    """Owns only processes it starts and applies bounded graceful shutdown."""

    def __init__(self, *, allowed_executables: Sequence[str] = ()) -> None:
        self._allowed = frozenset(allowed_executables)
        self._lock = threading.RLock()
        self._processes: dict[int, tuple[subprocess.Popen[bytes], tuple[str, ...], float]] = {}

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        stdout: int = subprocess.DEVNULL,
        stderr: int = subprocess.DEVNULL,
    ) -> ProcessRecord:
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("command must be a non-empty sequence of strings")
        executable = Path(command[0]).name
        if not self._allowed or executable not in self._allowed:
            raise PermissionError(
                f"executable is not allowlisted (configure allowed_executables): {executable}"
            )
        if cwd is not None and not Path(cwd).is_dir():
            raise ValueError("cwd must be an existing directory")
        clean_env = None
        if env is not None:
            if any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
                raise TypeError("environment keys and values must be strings")
            clean_env = dict(os.environ)
            clean_env.update(env)
        process = subprocess.Popen(
            list(command), cwd=str(cwd) if cwd is not None else None,
            env=clean_env, stdin=subprocess.DEVNULL, stdout=stdout,
            stderr=stderr, shell=False, close_fds=True,
        )
        started = time.monotonic()
        with self._lock:
            self._processes[process.pid] = (process, tuple(command), started)
        return ProcessRecord(process.pid, tuple(command), started, None, True)

    def inspect(self, process_id: int) -> ProcessRecord | None:
        with self._lock:
            entry = self._processes.get(process_id)
            if entry is None:
                return None
            process, command, started = entry
            code = process.poll()
            return ProcessRecord(process_id, command, started, code, code is None)

    def stop(self, process_id: int, *, grace_seconds: float = 5.0) -> ProcessRecord | None:
        if isinstance(grace_seconds, bool) or grace_seconds < 0:
            raise ValueError("grace_seconds must be non-negative")
        with self._lock:
            entry = self._processes.get(process_id)
        if entry is None:
            return None
        process, command, started = entry
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        code = process.poll()
        with self._lock:
            self._processes.pop(process_id, None)
        return ProcessRecord(process_id, command, started, code, False)

    def stop_all(self, *, grace_seconds: float = 5.0) -> tuple[ProcessRecord, ...]:
        with self._lock:
            process_ids = tuple(self._processes)
        records = []
        for process_id in process_ids:
            record = self.stop(process_id, grace_seconds=grace_seconds)
            if record is not None:
                records.append(record)
        return tuple(records)
