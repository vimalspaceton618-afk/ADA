"""Validated episode trajectories that prevent cross-episode sequence mixing."""
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty numeric vector")
    result = tuple(float(x) for x in value)
    if any(not math.isfinite(x) for x in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("source_digest must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class EpisodeWindow:
    episode_id: str
    start_index: int
    observations: tuple[tuple[float, ...], ...]
    actions: tuple[tuple[float, ...], ...]
    rewards: tuple[float, ...]
    terminated: tuple[bool, ...]
    timestamps: tuple[datetime, ...]
    source_digest: str

    def __post_init__(self):
        if not self.episode_id or type(self.start_index) is not int or self.start_index < 0:
            raise ValueError("episode_id and start_index are invalid")
        _digest(self.source_digest)
        if len(self.observations) < 2 or len(self.actions) != len(self.observations) - 1:
            raise ValueError("window requires T observations and T-1 actions")
        if not (len(self.rewards) == len(self.terminated) == len(self.actions)):
            raise ValueError("rewards and terminated labels must align with actions")
        if len(self.timestamps) != len(self.observations):
            raise ValueError("timestamps must align with observations")


@dataclass(frozen=True, slots=True)
class EpisodeTrajectory:
    """One episode; samples are never allowed to cross its boundary."""
    episode_id: str
    observations: tuple[tuple[float, ...], ...]
    actions: tuple[tuple[float, ...], ...]
    rewards: tuple[float, ...]
    terminated: tuple[bool, ...]
    timestamps: tuple[datetime, ...]
    source_digest: str

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty")
        _digest(self.source_digest)
        observations = tuple(_vector(x, "observation") for x in self.observations)
        actions = tuple(_vector(x, "action") for x in self.actions)
        rewards = tuple(float(x) for x in self.rewards)
        if len(observations) < 2 or len(actions) != len(observations) - 1:
            raise ValueError("trajectory requires T observations and T-1 actions")
        if any(len(x) != len(observations[0]) for x in observations):
            raise ValueError("all observation vectors must have the same dimension")
        if any(len(x) != len(actions[0]) for x in actions):
            raise ValueError("all action vectors must have the same dimension")
        if len(rewards) != len(actions) or any(not math.isfinite(x) for x in rewards):
            raise ValueError("rewards must be finite and align with actions")
        terminated = tuple(self.terminated)
        if len(terminated) != len(actions) or any(type(x) is not bool for x in terminated):
            raise ValueError("terminated must contain one boolean per action")
        timestamps = tuple(self.timestamps)
        if len(timestamps) != len(observations):
            raise ValueError("timestamps must align with observations")
        for timestamp in timestamps:
            if not isinstance(timestamp, datetime) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("timestamps must be timezone-aware")
        if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("timestamps must be strictly increasing")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "timestamps", timestamps)

    def window(self, start: int, length: int) -> EpisodeWindow:
        if type(start) is not int or type(length) is not int or start < 0 or length < 2:
            raise ValueError("start must be non-negative and length must be at least two")
        stop = start + length
        if stop > len(self.observations):
            raise ValueError("requested window exceeds episode boundary")
        return EpisodeWindow(
            self.episode_id, start,
            self.observations[start:stop], self.actions[start:stop-1],
            self.rewards[start:stop-1], self.terminated[start:stop-1],
            self.timestamps[start:stop], self.source_digest,
        )
