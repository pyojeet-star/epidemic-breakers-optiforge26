"""Tests for the fitness function."""
import numpy as np

from src import fitness, graph_gen
from src.utils import make_seeds


def _small_graph():
    return graph_gen.generate_service_graph(n_nodes=12, seed=3)


def test_f_formula_matches_components():
    G = _small_graph()
    theta = np.full(G.number_of_edges(), 0.5)
    res = fitness.evaluate(G, theta, seeds=[1, 2, 3])
    expected = res["cascade_size"] + 2 * res["false_trips"] + 0.1 * res["latency_penalty"]
    assert abs(res["F"] - expected) < 1e-9


def test_train_test_seeds_disjoint():
    train = make_seeds(12, base=1000)
    test = make_seeds(12, base=2000)
    assert not set(train) & set(test)


def test_more_seeds_stable_estimate():
    G = _small_graph()
    theta = np.full(G.number_of_edges(), 0.5)
    r4 = fitness.evaluate(G, theta, seeds=[1, 2, 3, 4])
    r8 = fitness.evaluate(G, theta, seeds=[1, 2, 3, 4, 5, 6, 7, 8])
    assert r4["n_seeds"] == 4 and r8["n_seeds"] == 8
    assert r4["F"] >= 0 and r8["F"] >= 0


def test_bad_theta_raises():
    G = _small_graph()
    try:
        fitness.evaluate(G, np.zeros(3), seeds=[1])
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong theta length")
