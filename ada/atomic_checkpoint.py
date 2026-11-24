"""Versioned, integrity-checked *single-process* pretraining snapshots.

A completed snapshot is published only after all files have been written,
fsynced, renamed into place and indexed by an atomically replaced latest.json.
Unpublished staging directories are never considered resumable.

Atomic rename and durability are guaranteed only on filesystems that support
same-filesystem atomic rename and POSIX-style fsync; Google Drive FUSE and
network/object stores require separate fault-injection verification. This is
not a distributed or concurrent-writer checkpoint protocol.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

VERSION = 1
REQUIRED_FILES = (
    "tokenizer.json",
    "model/config.json",
    "model/model.pt",
    "training_state.pt",
    "run_manifest.json",
)
_NAME = re.compile(r"step-[0-9]{12}-[0-9a-f]{32}\Z")


def _digest(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            h.update(block)
    return h.hexdigest(), size


def _fsync_directory(path: Path) -> None:
    # A directory fsync persists rename metadata on supporting POSIX filesystems.
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_pointer(root: Path, payload: dict) -> None:
    temporary = root / f".latest-{uuid4().hex}.tmp"
    try:
        _write_json(temporary, payload)
        os.replace(temporary, root / "latest.json")
        _fsync_directory(root)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_committed_checkpoint(output_dir: Path) -> Path | None:
    """Verify all files referenced by the published pointer; fail closed.

    Return None only when no pointer exists (legacy run or no saved snapshot).
    An invalid or corrupted *published* checkpoint is never silently skipped.
    """
    root = Path(output_dir) / "checkpoints"
    pointer = root / "latest.json"
    if not pointer.exists():
        return None
    try:
        latest = json.loads(pointer.read_text(encoding="utf-8"))
        name = latest["checkpoint"]
        if (
            latest.get("version") != VERSION
            or not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or type(latest.get("step")) is not int
        ):
            raise ValueError("invalid published checkpoint pointer")
        checkpoint = root / name
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError("published checkpoint directory missing or symlinked")
        index_file = checkpoint / "checkpoint.json"
        if index_file.is_symlink() or not index_file.is_file():
            raise ValueError("checkpoint index is absent or symlinked")
        index_digest, _ = _digest(index_file)
        if index_digest != latest.get("index_sha256"):
            raise ValueError("checkpoint index digest mismatch")
        index = json.loads(index_file.read_text(encoding="utf-8"))
        if (
            index.get("version") != VERSION
            or index.get("step") != latest["step"]
            or type(index.get("tokens_seen")) is not int
            or index["tokens_seen"] < 0
            or set(index.get("files", {})) != set(REQUIRED_FILES)
        ):
            raise ValueError("checkpoint index has incompatible contents")
        for relative_path in REQUIRED_FILES:
            file = checkpoint / relative_path
            if file.is_symlink() or not file.is_file():
                raise ValueError(f"checkpoint artifact missing: {relative_path}")
            expected = index["files"][relative_path]
            digest, size = _digest(file)
            if (digest != expected["sha256"] or size != expected["size_bytes"]
                    or size <= 0):
                raise ValueError(f"checkpoint integrity mismatch: {relative_path}")
        return checkpoint
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot resume: corrupt or incomplete published checkpoint: {exc}") from exc


def commit_checkpoint(
    output_dir: Path,
    *,
    step: int,
    tokens_seen: int,
    writer: Callable[[Path], None],
) -> Path:
    """Publish a complete immutable checkpoint after verifying its components.

    writer(stage) must produce all REQUIRED_FILES. On failure the current
    published checkpoint remains unchanged and the staging tree is cleaned.
    """
    if type(step) is not int or step < 1:
        raise ValueError("step must be a positive integer")
    if type(tokens_seen) is not int or tokens_seen < 0:
        raise ValueError("tokens_seen must be a non-negative integer")
    if not callable(writer):
        raise TypeError("writer must be callable")
    output_dir = Path(output_dir)
    root = output_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    current = resolve_committed_checkpoint(output_dir)
    if current is not None:
        current_step = json.loads((root / "latest.json").read_text())["step"]
        if step <= current_step:
            raise ValueError("checkpoint step must advance beyond the published step")

    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    destination = root / f"step-{step:012d}-{uuid4().hex}"
    try:
        writer(stage)
        checksums = {}
        for relative_path in REQUIRED_FILES:
            file = stage / relative_path
            if file.is_symlink() or not file.is_file():
                raise ValueError(f"checkpoint writer omitted: {relative_path}")
            digest, size = _digest(file)
            if size <= 0:
                raise ValueError(f"checkpoint writer created empty file: {relative_path}")
            checksums[relative_path] = {"sha256": digest, "size_bytes": size}
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
        index_file = stage / "checkpoint.json"
        _write_json(index_file, {
            "version": VERSION, "step": step, "tokens_seen": tokens_seen,
            "files": checksums,
        })
        _fsync_directory(stage / "model")
        _fsync_directory(stage)
        # Unique immutable destination, followed by a single publication point.
        os.rename(stage, destination)
        _fsync_directory(root)
        digest, _ = _digest(destination / "checkpoint.json")
        _atomic_pointer(root, {
            "version": VERSION, "checkpoint": destination.name,
            "step": step, "index_sha256": digest,
        })
        return destination
    finally:
        if stage.exists():
            shutil.rmtree(stage)
