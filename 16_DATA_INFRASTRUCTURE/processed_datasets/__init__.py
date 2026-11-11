"""Tokenized and packed sequences with boundary-aware labels."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True, slots=True)
class TokenizedExample:
    sample_id: str
    token_ids: tuple[int, ...]
    document_end: bool = True
    def __post_init__(self):
        if not self.sample_id.strip() or not self.token_ids: raise ValueError("sample_id and token_ids required")
        if any(type(x) is not int or x < 0 for x in self.token_ids): raise ValueError("token IDs must be non-negative integers")

@dataclass(frozen=True, slots=True)
class PackedSequence:
    input_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    sample_ids: tuple[str, ...]

def pack_examples(examples: Sequence[TokenizedExample], *, sequence_length: int, eos_token_id: int):
    if type(sequence_length) is not int or sequence_length < 2: raise ValueError("sequence_length must be >= 2")
    if type(eos_token_id) is not int or eos_token_id < 0: raise ValueError("invalid eos token")
    tokens, mask, ids, packed = [], [], [], []
    for item in examples:
        if not isinstance(item, TokenizedExample): raise TypeError("examples must be TokenizedExample")
        segment = item.token_ids + ((eos_token_id,) if item.document_end else ())
        for token in segment:
            tokens.append(token); mask.append(True); ids.append(item.sample_id)
            if len(tokens) == sequence_length:
                packed.append(PackedSequence(tuple(tokens), tuple(mask), tuple(ids)))
                tokens, mask, ids = [], [], []
    if tokens:
        pad = sequence_length - len(tokens)
        packed.append(PackedSequence(tuple(tokens + [eos_token_id] * pad),
                                     tuple(mask + [False] * pad), tuple(ids + [""] * pad)))
    return tuple(packed)

__all__ = ["TokenizedExample", "PackedSequence", "pack_examples"]
