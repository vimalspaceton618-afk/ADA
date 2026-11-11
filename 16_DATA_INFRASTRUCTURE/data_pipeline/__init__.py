"""Streaming JSONL ingestion and deterministic record batching."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator, Mapping, Any

class JsonlReader:
    def __init__(self, path: str | Path, *, max_line_bytes: int = 8_000_000):
        if max_line_bytes < 1: raise ValueError("max_line_bytes must be positive")
        self.path, self.max_line_bytes = Path(path), max_line_bytes
    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        with self.path.open("rb") as stream:
            for line_no, raw in enumerate(stream, 1):
                if not raw.strip(): continue
                if len(raw) > self.max_line_bytes: raise ValueError(f"line {line_no} exceeds size limit")
                try: value = json.loads(raw)
                except Exception as exc: raise ValueError(f"invalid JSON on line {line_no}") from exc
                if not isinstance(value, dict): raise TypeError(f"line {line_no} must be a JSON object")
                yield value

def batches(records, batch_size: int):
    if type(batch_size) is not int or batch_size < 1: raise ValueError("batch_size must be positive")
    batch = []
    for item in records:
        batch.append(item)
        if len(batch) == batch_size:
            yield tuple(batch); batch = []
    if batch: yield tuple(batch)

__all__ = ["JsonlReader", "batches"]
