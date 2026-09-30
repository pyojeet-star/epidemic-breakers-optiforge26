"""Tests for the baselines (fixed threshold, random search, oracle)."""
import numpy as np

from src import baselines
from src.graph_gen import edge_list, generate_service_graph
from src.simulator import _per_edge_params


def _small():
    return generate_service_graph(n_nodes=12, seed=3)


def test_oracle_formula_is_canonical():
    """The one oracle definition: theta = clip(noise_amp + 0.10, 0, 1)."""
    G = _small()
    edges = edge_list(G)
    n = G.number_of_nodes()
    noise_mat, _ = _per_edge_params(edges, n, 0.3, 0.25)
    amps = np.array([noise_mat[u, v] for (u, v) in edges])
    th = baselines.oracle_thresholds(G, noise_amp=0.3, spread_p=0.25)
    assert th.shape == (len(edges),)
    assert np.allclose(th, np.clip(amps + 0.10, 0.0, 1.0))
    assert np.all((th >= 0.0) & (th <= 1.0))


def test_oracle_margin_parameter():
    G = _small()
    th = baselines.oracle_thresholds(G, margin=0.0)
    edges = edge_list(G)
    noise_mat, _ = _per_edge_params(edges, G.number_of_nodes(), 0.3, 0.25)
    amps = np.array([noise_mat[u, v] for (u, v) in edges])
    assert np.allclose(th, np.clip(amps, 0.0, 1.0))
