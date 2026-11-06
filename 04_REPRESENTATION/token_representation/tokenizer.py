"""Deterministic lexical tokenization with source character offsets.

This is a surface tokenizer, not a model tokenizer and not a semantic parser.
It does not emit vocabulary IDs or claim compatibility with any model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_PATTERN = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class LexicalToken:
    index: int
    text: str
    normalized: str
    start: int
    end: int
    token_type: str


def tokenize_text(text: str) -> tuple[LexicalToken, ...]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    output = []
    for index, match in enumerate(_TOKEN_PATTERN.finditer(text)):
        token = match.group(0)
        token_type = (
            "word"
            if any(char.isalnum() or char == "_" for char in token)
            else "punctuation"
        )
        output.append(LexicalToken(
            index=index,
            text=token,
            normalized=token.casefold() if token_type == "word" else token,
            start=match.start(),
            end=match.end(),
            token_type=token_type,
        ))
    return tuple(output)


def tokens_to_payload(text: str) -> dict[str, object]:
    tokens = tokenize_text(text)
    return {
        "tokenizer_id": "ada.lexical.v1",
        "tokenizer_semantics": "unicode_surface_tokens_not_model_token_ids",
        "source_text": text,
        "token_count": len(tokens),
        "tokens": [
            {
                "index": token.index,
                "text": token.text,
                "normalized": token.normalized,
                "start": token.start,
                "end": token.end,
                "token_type": token.token_type,
            }
            for token in tokens
        ],
        "offset_unit": "unicode_code_points",
        "offset_convention": "zero_based_half_open",
    }
