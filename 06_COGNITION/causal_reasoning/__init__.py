"""Inspect explicitly supplied causal edges; does not discover causality."""
from collections import defaultdict,deque
def causal_reachability(edges,source):
    graph=defaultdict(list)
    for cause,effect in edges:
        if not cause or not effect: raise ValueError("edge endpoints must be non-empty")
        graph[cause].append(effect)
    seen={source}; queue=deque([source])
    while queue:
        node=queue.popleft()
        for nxt in graph[node]:
            if nxt not in seen: seen.add(nxt); queue.append(nxt)
    seen.discard(source)
    return tuple(sorted(seen))
