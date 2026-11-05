"""Transparent surface-pattern detection for text observations.

This baseline finds repeated token n-grams. It is not semantic pattern learning,
causal discovery, anomaly detection, or a trained classifier.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..contracts import Modality, Observation

_TOKEN = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", re.UNICODE)


@dataclass(frozen=True, slots=True)
class SurfacePattern:
    tokens: tuple[str, ...]
    occurrences: int
    kind: str = "repeated_ngram"


class SurfacePatternDetector:
    def __init__(self, *, ngram_sizes: tuple[int, ...] = (1, 2, 3), minimum_occurrences: int = 2) -> None:
        if not ngram_sizes or any(type(size) is not int or size < 1 for size in ngram_sizes):
            raise ValueError("ngram_sizes must contain positive integers")
        if type(minimum_occurrences) is not int or minimum_occurrences < 2:
            raise ValueError("minimum_occurrences must be at least 2")
        self.ngram_sizes = tuple(sorted(set(ngram_sizes)))
        self.minimum_occurrences = minimum_occurrences

    def detect(self, observation: Observation) -> tuple[SurfacePattern, ...]:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        if observation.modality is not Modality.TEXT or observation.extracted_text is None:
            return ()
        tokens = [token.casefold() for token in _TOKEN.findall(observation.extracted_text)]
        counts: Counter[tuple[str, ...]] = Counter()
        for size in self.ngram_sizes:
            for start in range(0, len(tokens) - size + 1):
                counts[tuple(tokens[start:start + size])] += 1
        patterns = [
            SurfacePattern(tokens=ngram, occurrences=count)
            for ngram, count in counts.items()
            if count >= self.minimum_occurrences
        ]
        return tuple(sorted(patterns, key=lambda item: (-item.occurrences, len(item.tokens), item.tokens)))
