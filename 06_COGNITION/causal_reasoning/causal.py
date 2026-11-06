"""Represent and inspect explicitly supplied causal edges."""
from __future__ import annotations
from collections import defaultdict, deque

def causal_reachability(edges, source: str):
    """Return nodes reachable in supplied directed edges; this is not causal discovery."""
    graph = defaultdict(list)
    for cause, effect in edges:
        if not cause or not effect:
            raise ValueError("causal edge endpoints must be non-empty")
        graph[cause].append(effect)
    seen, queue = {source}, deque([source])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    seen.discard(source)
    return tuple(sorted(seen))
