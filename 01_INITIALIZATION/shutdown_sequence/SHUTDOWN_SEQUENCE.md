# Shutdown Sequence

**Status:** Proposed graceful lifecycle contract

## Trigger sources

Normal command completion; user cancellation/interrupt; deadline or resource-budget exhaustion; unrecoverable runtime failure; external supervisor termination.

## Ordered shutdown

1. Stop accepting new work.
2. Signal cancellation to active tasks and data workers.
3. Wait for bounded cooperative completion.
4. If training is active and policy permits, flush a consistent checkpoint to a temporary path.
5. Verify checkpoint integrity, then atomically publish its final name/manifest.
6. Flush logs, metrics and experiment metadata.
7. Close dataset streams, file handles and network sessions.
8. Destroy distributed process groups and release device/runtime resources.
9. Remove only temporary files owned by this run.
10. Emit final status and exit code.

## Checkpoint safety

A checkpoint includes model, optimizer and scheduler state; global step; RNG states; sampler/data cursor; architecture/config digest; tokenizer/data manifest identities; code revision and framework versions.

Never overwrite the last known-good checkpoint before a new one is written and verified. Use temporary write, fsync where supported, and atomic rename. A partial checkpoint is not resumable.

## Failure handling

- Shutdown is idempotent.
- Cleanup failures are logged but do not hide original failure.
- Timeouts are bounded.
- Never claim checkpoint success unless verification passed.
- Do not delete user data, source datasets or prior approved artifacts.

## Exit status

Emit COMPLETED, CANCELLED, FAILED or FAILED_CHECKPOINT, plus run ID and checkpoint identity if valid.
