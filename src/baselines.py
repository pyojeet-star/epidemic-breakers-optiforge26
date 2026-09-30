"""Baselines: a fixed threshold and random search at the same eval budget.

``theta = 2.0`` on every edge disables breakers entirely (the noisy observed
stress never exceeds 1 + noise_amp).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from . import fitness as fitmod
from .graph_gen import edge_list
from .simulator import _per_edge_params  # intra-package use of the env sampler


def fixed_threshold(
    graph: nx.DiGraph,
    theta_value: float,
    train_seeds: list[int],
    test_seeds: list[int],
    w_false_trips: float = fitmod.W_FALSE_TRIPS,
    w_latency: float = fitmod.W_LATENCY,
    **sim_kwargs,
) -> dict:
    """One constant threshold on every edge. Evaluated on train and test."""
    n_edges = graph.number_of_edges()
    theta = np.full(n_edges, theta_value)
    kw = dict(w_false_trips=w_false_trips, w_latency=w_latency)
    train = fitmod.evaluate(graph, theta, train_seeds, **kw, **sim_kwargs)
    test = fitmod.evaluate(graph, theta, test_seeds, **kw, **sim_kwargs)
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
    w_false_trips: float = fitmod.W_FALSE_TRIPS,
    w_latency: float = fitmod.W_LATENCY,
    **sim_kwargs,
) -> dict:
    """Random thresholds in [init_lo, init_hi]; best on train, reported on test.

    Shares the GA's informed init range so the comparison isolates the
    optimizer, not the prior.
    """
    n_edges = graph.number_of_edges()
    kw = dict(w_false_trips=w_false_trips, w_latency=w_latency)
    best, best_f = None, float("inf")
    for _ in range(budget):
        theta = rng.uniform(init_lo, init_hi, n_edges)
        f = fitmod.evaluate(graph, theta, train_seeds, **kw, **sim_kwargs)["F"]
        if f < best_f:
            best_f, best = f, theta
    test = fitmod.evaluate(graph, best, test_seeds, **kw, **sim_kwargs)
    return {
        "method": "random-search",
        "theta": best,
        "train_F": best_f,
        "test": test,
        "evals": budget,
        "gens_to_target": None,
    }


def oracle_thresholds(
    graph: nx.DiGraph,
    noise_amp: float = 0.3,
    spread_p: float = 0.25,
    margin: float = 0.10,
) -> np.ndarray:
    """Reference heuristic using hidden per-edge noise values.

    The single canonical definition, used by the code, the README, and the
    demos: ``theta_e = clip(noise_amp_e + margin, 0, 1)`` -- each breaker's
    threshold is set just above its edge's *true* noise amplitude. Those
    amplitudes are hidden from every real tuner (the GA, random search, and
    any fixed threshold), so this is a reference point for how much
    headroom the noise level leaves -- not a ceiling a method is expected
    to reach, and not an achievable baseline.
    """
    edges = edge_list(graph)
    n = graph.number_of_nodes()
    noise_mat, _ = _per_edge_params(edges, n, noise_amp, spread_p)
    amps = np.array([noise_mat[u, v] for (u, v) in edges])
    return np.clip(amps + margin, 0.0, 1.0)
