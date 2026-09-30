"""Interactive Epidemic Breakers demo.

Deployed on Streamlit Community Cloud straight from this repo
(main file: streamlit_app.py). Lets anyone play with the cascade
simulator: pick a threshold strategy, tweak the environment, and watch
the epidemic curve and the cost breakdown.
"""
from pathlib import Path

import numpy as np
import streamlit as st

from src import fitness as fitmod
from src import graph_gen
from src import simulator
from src.baselines import oracle_thresholds
from src.graph_gen import edge_list
from src.utils import make_seeds

st.set_page_config(page_title="Epidemic Breakers", page_icon="🦠", layout="wide")

REPO = Path(__file__).resolve().parent
BEST_THETA = REPO / "results" / "best_theta.npy"


@st.cache_resource
def get_graph():
    return graph_gen.generate_service_graph(n_nodes=40, seed=7)


@st.cache_resource
def get_best_theta(n_edges: int) -> np.ndarray | None:
    if BEST_THETA.exists():
        theta = np.load(BEST_THETA)
        if theta.shape == (n_edges,):
            return theta
    return None


def build_theta(strategy: str, fixed: float, best: np.ndarray | None,
                graph, noise_amp: float, spread_p: float, n_edges: int):
    if strategy == "GA-optimized (precomputed)":
        if best is None:
            st.warning("best_theta.npy not found — falling back to fixed 0.5.")
            return np.full(n_edges, 0.5), "fixed-0.5 (fallback)"
        return best, "guided GA"
    if strategy == "Oracle (noise amplitude + 0.1)":
        return oracle_thresholds(graph, noise_amp, spread_p), "oracle"
    return np.full(n_edges, fixed), f"fixed-{fixed:.2f}"


G = get_graph()
edges = edge_list(G)
N_EDGES = len(edges)
best_theta = get_best_theta(N_EDGES)

st.title("🦠 Epidemic Breakers")
st.markdown(
    "Circuit-breaker thresholds on a microservice call graph. One service "
    "slows down, the slowness spreads to callers — unless a breaker's noisy "
    "stress signal crosses its threshold first. Tune `theta` per edge to "
    f"minimize `F = cascade + 2·false_trips + 0.1·latency` "
    f"({G.number_of_nodes()} services, {N_EDGES} breakers)."
)

with st.sidebar:
    st.header("Environment")
    seed = st.number_input("Scenario seed", value=7, step=1)
    spread_p = st.slider("Spread probability", 0.05, 0.60, 0.25, 0.01)
    cooldown = st.slider("Breaker cooldown (steps)", 1, 20, 5, 1)
    noise_amp = st.slider("Monitoring noise amplitude", 0.0, 0.6, 0.3, 0.05)
    n_steps = st.slider("Simulation steps", 10, 80, 40, 5)
    st.header("Thresholds")
    strategy = st.radio(
        "Strategy",
        ["Fixed (slider)", "GA-optimized (precomputed)", "Oracle (noise amplitude + 0.1)"],
        help="Oracle: reference heuristic using hidden per-edge noise values "
             "(θ = clip(noise_amp + 0.10)); not available to any real tuner.",
    )
    fixed_theta = st.slider("Fixed theta", 0.05, 0.95, 0.5, 0.05)

sim_kw = dict(spread_p=spread_p, cooldown=cooldown, n_steps=n_steps,
              noise_amp=noise_amp)

theta, label = build_theta(strategy, fixed_theta, best_theta, G,
                           noise_amp, spread_p, N_EDGES)

col_run, _ = st.columns([1, 4])
run = col_run.button("▶ Run scenario", type="primary")

if run:
    res = simulator.simulate(G, theta, int(seed), record_history=True, **sim_kw)
    f = fitmod.scenario_cost(res)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F (lower is better)", f"{f:.2f}")
    m2.metric("Cascade size", f"{res['cascade_size']} / {G.number_of_nodes()}")
    m3.metric("False trips", res["false_trips"])
    m4.metric("Latency penalty", f"{res['latency_penalty']:.1f}%")
    st.subheader(f"Epidemic curve — {label}, seed {int(seed)}")
    st.line_chart({"slow services": res["slow_history"]})
    st.caption(
        "Try dragging the fixed-theta slider: too low → false trips explode, "
        "too high → the cascade outruns the breakers."
    )

st.divider()
st.subheader("⚔️ Compare strategies")
n_cmp = st.slider("Scenarios per strategy", 4, 16, 8, 1)
if st.button("Compare on fresh seeds"):
    cmp_seeds = make_seeds(int(n_cmp), base=9000)
    strategies = {
        "fixed-0.5": np.full(N_EDGES, 0.5),
        "oracle": oracle_thresholds(G, noise_amp, spread_p),
    }
    if best_theta is not None:
        strategies["guided GA"] = best_theta
    rows = []
    prog = st.progress(0.0)
    for i, (name, th) in enumerate(strategies.items()):
        r = fitmod.evaluate(G, th, cmp_seeds, **sim_kw)
        rows.append({"strategy": name, "mean F": round(r["F"], 2),
                     "cascade": round(r["cascade_size"], 1),
                     "false trips": round(r["false_trips"], 1)})
        prog.progress((i + 1) / len(strategies))
    st.bar_chart({r["strategy"]: r["mean F"] for r in rows})
    st.table(rows)
    st.caption("Mean F over held-out scenarios — lower wins. "
               "Full study: see the repo README.")
