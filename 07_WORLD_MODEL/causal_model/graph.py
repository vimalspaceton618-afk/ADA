"""Explicit acyclic dependency graphs; edges are hypotheses, not causal proof."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CausalGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self):
        nodes, edges = tuple(self.nodes), tuple(tuple(x) for x in self.edges)
        if any(not isinstance(x, str) or not x.strip() for x in nodes):
            raise ValueError("causal node identifiers must be non-empty strings")
        if len(set(nodes)) != len(nodes):
            raise ValueError("causal node identifiers must be unique")
        known = set(nodes)
        if any(len(edge) != 2 or edge[0] not in known or edge[1] not in known for edge in edges):
            raise ValueError("causal edges must connect declared nodes")
        if any(source == target for source, target in edges):
            raise ValueError("self-cycles are not permitted")
        if len(set(edges)) != len(edges):
            raise ValueError("duplicate causal edges are not permitted")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        self.topological_order()  # Reject cyclic structures at construction.

    def parents(self, node: str) -> tuple[str, ...]:
        if node not in self.nodes:
            raise KeyError(node)
        return tuple(source for source, target in self.edges if target == node)

    def children(self, node: str) -> tuple[str, ...]:
        if node not in self.nodes:
            raise KeyError(node)
        return tuple(target for source, target in self.edges if source == node)

    def topological_order(self) -> tuple[str, ...]:
        incoming = {node: 0 for node in self.nodes}
        for source, target in self.edges:
            incoming[target] += 1
        queue = [node for node in self.nodes if incoming[node] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.children(node):
                incoming[child] -= 1
                if incoming[child] == 0:
                    queue.append(child)
        if len(order) != len(self.nodes):
            raise ValueError("causal dependency graph must be acyclic")
        return tuple(order)

    def descendants(self, node: str) -> tuple[str, ...]:
        if node not in self.nodes:
            raise KeyError(node)
        found, frontier = set(), list(self.children(node))
        while frontier:
            item = frontier.pop(0)
            if item not in found:
                found.add(item)
                frontier.extend(self.children(item))
        order = self.topological_order()
        return tuple(item for item in order if item in found)
