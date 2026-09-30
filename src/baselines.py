"""Baselines: a fixed threshold and random search at the same eval budget.

``theta = 2.0`` on every edge disables breakers entirely (the noisy observed
stress never exceeds 1 + noise_amp).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from . import fitness as fitmod


def fixed_threshold(
    graph: nx.DiGraph,
    theta_value: float,
    train_seeds: list[int],
    test_seeds: list[int],
    **sim_kwargs,
) -> dict:
    """One constant threshold on every edge. Evaluated on train and test."""
    n_edges = graph.number_of_edges()
    theta = np.full(n_edges, theta_value)
    train = fitmod.evaluate(graph, theta, train_seeds, **sim_kwargs)
    test = fitmod.evaluate(graph, theta, test_seeds, **sim_kwargs)
    return {
        "method": f"fixed-{theta_value}",
        "theta": theta,
        "train_F": train["F"],
        "test": test,
        "evals": 1,
        "gens_to_target": None,
    }


def random_search(
    graph: nx.DiGraph,
    train_seeds: list[int],
    test_seeds: list[int],
    budget: int,
    rng: np.random.Generator,
    init_lo: float = 0.2,
    init_hi: float = 0.7,
    **sim_kwargs,
) -> dict:
    """Random thresholds in [init_lo, init_hi]; best on train, reported on test.

    Shares the GA's informed init range so the comparison isolates the
    optimizer, not the prior.
    """
    n_edges = graph.number_of_edges()
    best, best_f = None, float("inf")
    for _ in range(budget):
        theta = rng.uniform(init_lo, init_hi, n_edges)
        f = fitmod.evaluate(graph, theta, train_seeds, **sim_kwargs)["F"]
        if f < best_f:
            best_f, best = f, theta
    test = fitmod.evaluate(graph, best, test_seeds, **sim_kwargs)
    return {
        "method": "random-search",
        "theta": best,
        "train_F": best_f,
        "test": test,
        "evals": budget,
        "gens_to_target": None,
    }
