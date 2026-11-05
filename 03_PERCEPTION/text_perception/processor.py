"""Text perception without pretending to perform semantic model inference.

This adapter preserves source text and computes reproducible surface statistics.
Semantic interpretation, entity extraction, and language identification require
explicit downstream adapters and are intentionally not guessed here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ..contracts import (
    Modality, Observation, PerceptionInput, PerceptionStatus, content_digest,
)

_WORD = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", re.UNICODE)
_SENTENCE_END = re.compile(r"[.!?。！？]+")


class TextPerceptionAdapter:
    modality = Modality.TEXT

    def perceive(self, item: PerceptionInput) -> Observation:
        if not isinstance(item.content, str):
            raise TypeError("TextPerceptionAdapter requires normalized text content")
        text = item.content
        words = _WORD.findall(text)
        sentences = [part for part in _SENTENCE_END.split(text) if part.strip()]
        line_count = len(text.splitlines()) or 1
        attributes: Mapping[str, Any] = {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "line_count": line_count,
            "unique_word_count": len({word.casefold() for word in words}),
            "whitespace_only": not bool(text.strip()),
            "language": None,
            "semantic_analysis": "not_performed",
        }
        return Observation(
            modality=Modality.TEXT,
            status=PerceptionStatus.OBSERVED,
            summary=f"Text received: {len(words)} lexical tokens across {len(sentences)} detected sentence segments",
            source_input_id=item.input_id,
            content_sha256=content_digest(text),
            extracted_text=text,
            attributes=attributes,
        )
