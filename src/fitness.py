"""Objective function: mean scenario cost of a threshold vector.

F = cascade_size + 2 * false_trips + 0.1 * latency_penalty
averaged over a list of scenario seeds. Lower is better.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .simulator import simulate_many

W_FALSE_TRIPS = 2.0
W_LATENCY = 0.1


def scenario_cost(metrics: dict) -> float:
    """Cost of a single simulated scenario."""
    return (
        metrics["cascade_size"]
        + W_FALSE_TRIPS * metrics["false_trips"]
        + W_LATENCY * metrics["latency_penalty"]
    )


def evaluate(
    graph: nx.DiGraph,
    theta,
    seeds,
    **sim_kwargs,
) -> dict:
    """Evaluate ``theta`` over ``seeds``; return mean F and mean components."""
    results = simulate_many(graph, theta, seeds, **sim_kwargs)
    costs = np.array([scenario_cost(r) for r in results])
    return {
        "F": float(costs.mean()),
        "F_std": float(costs.std()),
        "cascade_size": float(np.mean([r["cascade_size"] for r in results])),
        "false_trips": float(np.mean([r["false_trips"] for r in results])),
        "latency_penalty": float(np.mean([r["latency_penalty"] for r in results])),
        "n_seeds": len(seeds),
    }
