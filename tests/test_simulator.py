"""Tests for the cascade simulator."""
import numpy as np

from src import graph_gen, simulator


def _small_graph():
    return graph_gen.generate_service_graph(n_nodes=12, seed=3)


def test_deterministic_same_seed():
    G = _small_graph()
    theta = np.full(G.number_of_edges(), 0.5)
    r1 = simulator.simulate(G, theta, seed=11)
    r2 = simulator.simulate(G, theta, seed=11)
    assert r1 == r2


def test_different_seeds_can_differ():
    G = _small_graph()
    theta = np.full(G.number_of_edges(), 0.5)
    r1 = simulator.simulate(G, theta, seed=11)
    r2 = simulator.simulate(G, theta, seed=12)
    # injection node differs almost surely; cascades should differ sometimes
    assert r1["cascade_size"] >= 1 and r2["cascade_size"] >= 1


def test_no_breakers_full_spread():
    # theta=2.0 disables breakers; p=1.0 on uniform edges infects every
    # caller every step -> the whole weakly-connected graph goes slow.
    G = _small_graph()
    theta = np.full(G.number_of_edges(), 2.0)
    r = simulator.simulate(G, theta, seed=5, spread_p=1.0, n_steps=30,
                           heterogeneous=False)
    assert r["cascade_size"] == G.number_of_nodes()
    assert r["false_trips"] == 0


def test_lower_theta_cannot_spread_more():
    # Same seed => same noise/infection draws, so lowering thresholds can only
    # block more edges and never grow the cascade.
    G = _small_graph()
    n = G.number_of_edges()
    lo = simulator.simulate(G, np.zeros(n), seed=9, n_steps=30)
    hi = simulator.simulate(G, np.full(n, 2.0), seed=9, n_steps=30)
    assert lo["cascade_size"] <= hi["cascade_size"]


def test_zero_theta_causes_false_trips():
    # theta=0 with healthy callees: noise alone trips breakers -> false trips.
    G = _small_graph()
    n = G.number_of_edges()
    r = simulator.simulate(G, np.zeros(n), seed=9, n_steps=30)
    assert r["false_trips"] >= 1


def test_theta_length_mismatch_raises():
    G = _small_graph()
    try:
        simulator.simulate(G, np.zeros(3), seed=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong theta length")


def test_simulate_many_reuses_matrices_and_matches():
    G = _small_graph()
    theta = np.full(G.number_of_edges(), 0.5)
    seeds = [21, 22, 23]
    many = simulator.simulate_many(G, theta, seeds)
    single = [simulator.simulate(G, theta, s) for s in seeds]
    assert many == single


def test_heterogeneous_params_deterministic_and_varied():
    G = _small_graph()
    n = G.number_of_nodes()
    edges = list(G.edges())
    m1 = simulator._per_edge_params(edges, n, 0.3, 0.25)
    m2 = simulator._per_edge_params(edges, n, 0.3, 0.25)
    assert np.array_equal(m1[0], m2[0]) and np.array_equal(m1[1], m2[1])
    amps = m1[0][m1[0] > 0]
    assert amps.min() < amps.max()  # edges genuinely differ
