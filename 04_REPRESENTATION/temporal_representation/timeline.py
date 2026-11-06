"""Timestamp validation and stable temporal payload helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def validate_timed_records(
    records: Sequence[Mapping[str, Any]],
    *,
    start_key: str,
    end_key: str | None = None,
    duration_seconds: float | None = None,
) -> tuple[dict[str, Any], ...]:
    if isinstance(records, (str, bytes)):
        raise TypeError("timed records must be a sequence of mappings")
    if not isinstance(records, Sequence):
        raise TypeError("timed records must be a sequence")
    if duration_seconds is not None:
        if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)):
            raise TypeError("duration_seconds must be numeric or None")
        if not math.isfinite(float(duration_seconds)) or duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and non-negative")

    result: list[dict[str, Any]] = []
    previous_start = -math.inf
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"timed record {index} must be a mapping")
        if start_key not in record:
            raise ValueError(f"timed record {index} is missing {start_key}")
        start = record[start_key]
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise TypeError(f"timed record {index} {start_key} must be numeric")
        start_value = float(start)
        if not math.isfinite(start_value) or start_value < 0:
            raise ValueError(
                f"timed record {index} {start_key} must be finite and non-negative"
            )
        if start_value < previous_start:
            raise ValueError("timestamps must be non-decreasing")
        previous_start = start_value
        if duration_seconds is not None and start_value > duration_seconds + 1e-6:
            raise ValueError(f"timed record {index} starts beyond media duration")
        item = dict(record)
        item[start_key] = start_value
        if end_key is not None:
            if end_key not in record:
                raise ValueError(f"timed record {index} is missing {end_key}")
            end = record[end_key]
            if isinstance(end, bool) or not isinstance(end, (int, float)):
                raise TypeError(f"timed record {index} {end_key} must be numeric")
            end_value = float(end)
            if not math.isfinite(end_value) or end_value < start_value:
                raise ValueError(
                    f"timed record {index} {end_key} must be finite and >= start"
                )
            if duration_seconds is not None and end_value > duration_seconds + 1e-6:
                raise ValueError(f"timed record {index} ends beyond media duration")
            item[end_key] = end_value
        result.append(item)
    return tuple(result)
