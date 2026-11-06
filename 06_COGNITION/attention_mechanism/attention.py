"""Deterministic lexical attention baseline; not learned attention."""
from __future__ import annotations
import re
_WORD = re.compile(r"[a-z0-9_]+", re.I)

def select_focus(query: str, items, *, limit: int = 5):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be non-empty")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    terms = {w.casefold() for w in _WORD.findall(query) if len(w) > 1}
    scored = []
    for index, item in enumerate(items):
        text = item if isinstance(item, str) else str(item)
        words = {w.casefold() for w in _WORD.findall(text) if len(w) > 1}
        score = len(terms & words) / max(1, len(terms))
        scored.append((index, float(score), item))
    return tuple(sorted(scored, key=lambda x: (-x[1], x[0]))[:limit])
