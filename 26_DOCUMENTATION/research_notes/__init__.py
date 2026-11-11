"""Timestamped research notes linked to hypotheses and experiments."""
from __future__ import annotations
from dataclasses import dataclass
import time

@dataclass(frozen=True, slots=True)
class ResearchNote:
    note_id: str
    title: str
    body: str
    author: str
    created_at: float
    tags: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    def __post_init__(self):
        if any(not x.strip() for x in (self.note_id, self.title, self.body, self.author)):
            raise ValueError("research note fields required")

class ResearchNotebook:
    def __init__(self): self._notes = {}
    def add(self, title: str, body: str, author: str, *, tags=(), references=()):
        note = ResearchNote(f"note-{len(self._notes)+1}", title, body, author, time.time(), tuple(tags), tuple(references))
        self._notes[note.note_id] = note
        return note
    def search(self, term: str):
        q = term.casefold()
        return tuple(x for x in self._notes.values()
                     if q in x.title.casefold() or q in x.body.casefold() or any(q in t.casefold() for t in x.tags))

__all__ = ["ResearchNote", "ResearchNotebook"]
