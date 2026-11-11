"""Developer guide index with topic and audience tags."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class DeveloperGuide:
    guide_id: str
    title: str
    path: str
    audience: str
    topics: tuple[str, ...]
    def __post_init__(self):
        if any(not x.strip() for x in (self.guide_id, self.title, self.path, self.audience)):
            raise ValueError("guide fields required")
        if self.path.startswith("/") or ".." in self.path.split("/"): raise ValueError("guide path must be repository-relative")

class GuideIndex:
    def __init__(self): self._guides = {}
    def add(self, guide: DeveloperGuide):
        if not isinstance(guide, DeveloperGuide): raise TypeError("DeveloperGuide required")
        if guide.guide_id in self._guides: raise ValueError("guide ID already exists")
        self._guides[guide.guide_id] = guide
    def search(self, topic: str):
        term = topic.casefold()
        return tuple(g for g in self._guides.values() if any(term in x.casefold() for x in g.topics))

__all__ = ["DeveloperGuide", "GuideIndex"]
