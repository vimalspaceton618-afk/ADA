"""Durable JSONL audit journal with hash-chain verification.

The journal is tamper-evident, not tamper-proof; deploy it on append-only or
WORM storage for stronger retention guarantees.
"""
from __future__ import annotations
import json, os, hashlib, threading, time
from pathlib import Path
from typing import Any
from .. import AuditEntry, _canonical, digest

class FileAuditLedger:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._entries = self._load()
        if not self.verify():
            raise ValueError("audit journal hash chain is invalid")
    def _load(self):
        if not self.path.exists(): return []
        values = []
        with self.path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    item = json.loads(line)
                    values.append(AuditEntry(
                        item["sequence"], item["event_type"], item["subject_id"],
                        item["payload_digest"], item["previous_hash"], item["entry_hash"],
                        item["timestamp"],
                    ))
                except Exception as exc:
                    raise ValueError(f"invalid audit record at line {number}") from exc
        return values
    def append(self, event_type: str, subject_id: str, payload: Any, *, timestamp: float | None = None):
        if not event_type.strip() or not subject_id.strip():
            raise ValueError("event_type and subject_id are required")
        with self._lock:
            seq = len(self._entries)
            previous = self._entries[-1].entry_hash if self._entries else "0" * 64
            stamp = time.time() if timestamp is None else float(timestamp)
            payload_digest = digest(payload)
            body = (seq, event_type, subject_id, payload_digest, previous, stamp)
            entry_hash = hashlib.sha256(_canonical(body)).hexdigest()
            entry = AuditEntry(seq, event_type, subject_id, payload_digest, previous, entry_hash, stamp)
            record = json.dumps({
                "sequence": seq, "event_type": event_type, "subject_id": subject_id,
                "payload_digest": payload_digest, "previous_hash": previous,
                "entry_hash": entry_hash, "timestamp": stamp,
            }, sort_keys=True, separators=(",", ":")) + "\n"
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(record)
                stream.flush()
                os.fsync(stream.fileno())
            self._entries.append(entry)
            return entry
    @property
    def entries(self):
        with self._lock: return tuple(self._entries)
    def verify(self):
        with self._lock:
            previous = "0" * 64
            for i, entry in enumerate(self._entries):
                if entry.sequence != i or entry.previous_hash != previous:
                    return False
                body = (i, entry.event_type, entry.subject_id, entry.payload_digest, previous, entry.timestamp)
                if hashlib.sha256(_canonical(body)).hexdigest() != entry.entry_hash:
                    return False
                previous = entry.entry_hash
            return True

__all__ = ["FileAuditLedger"]
