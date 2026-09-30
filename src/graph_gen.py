"""Seeded random microservice call-graph generator.

Convention: nodes are services labelled 0..n-1. A directed edge u -> v means
service u *calls* service v. Slowness therefore cascades from callee to caller
(v slow  =>  its callers u slow down).
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def generate_service_graph(
    n_nodes: int = 40, seed: int = 7, avg_out_degree: float = 3.0
) -> nx.DiGraph:
    """Build a weakly-connected random call graph.

    A random Hamiltonian path guarantees weak connectivity; extra
    Erdos-Renyi edges raise the mean out-degree toward ``avg_out_degree``.
    Fully seeded: the same ``seed`` always yields the same graph.
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes - 1):  # backbone keeps the graph weakly connected
        G.add_edge(int(order[i]), int(order[i + 1]))

    extra_p = max(0.0, (avg_out_degree - 1.0) / max(1, n_nodes - 1))
    if extra_p > 0:
        roll = rng.random((n_nodes, n_nodes))
        for u in range(n_nodes):
            for v in range(n_nodes):
                if u != v and not G.has_edge(u, v) and roll[u, v] < extra_p:
                    G.add_edge(u, v)
    return G


def edge_list(graph: nx.DiGraph) -> list[tuple[int, int]]:
    """Canonical edge ordering used to align theta vectors with the graph."""
    return list(graph.edges())
