"""Entry point: baselines vs GA with a strict train/test seed split.

Runs (default): fixed threshold, random search, vanilla GA, guided GA --
each over several seeded runs. Prints a results table, saves CSV + an SVG
convergence plot under results/.

Round 2 mode (--round2): harder environment (higher spread probability,
longer breaker cooldown). Compares warm-starting the GA from the Round 1
population (with a temporary mutation boost) against restarting from
scratch, measuring recovery on Round 2 test seeds.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from src import baselines as blmod
from src import fitness as fitmod
from src import ga as gamod
from src import graph_gen
from src.utils import (
    ensure_dir,
    get_logger,
    make_rng,
    make_seeds,
    save_convergence_svg,
    write_csv,
)

log = get_logger()

ROUND1_KW = dict(spread_p=0.25, cooldown=5, n_steps=40)
ROUND2_KW = dict(spread_p=0.45, cooldown=12, n_steps=40)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Epidemic Breakers - OptiForge 2026")
    p.add_argument("--runs", type=int, default=3, help="independent runs per method")
    p.add_argument("--train-seeds", type=int, default=24)
    p.add_argument("--test-seeds", type=int, default=12)
    p.add_argument("--pop", type=int, default=30, help="GA population size")
    p.add_argument("--gens", type=int, default=40, help="GA generations")
    p.add_argument("--seed", type=int, default=42, help="master seed")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--quick", action="store_true", help="tiny budgets for smoke tests")
    p.add_argument("--round2", action="store_true", help="run the Round 2 shift experiment")
    return p


def summarise(method: str, run: int, res: dict, train_f: float, secs: float) -> dict:
    t = res["test"]
    return {
        "method": method,
        "run": run,
        "test_F_mean": round(t["F"], 3),
        "test_F_std": round(t["F_std"], 3),
        "cascade_size": round(t["cascade_size"], 2),
        "false_trips": round(t["false_trips"], 2),
        "latency_penalty": round(t["latency_penalty"], 2),
        "train_F": round(train_f, 3),
        "gens_to_target": res.get("gens_to_target") if res.get("gens_to_target") is not None else "-",
        "evals": res.get("evals", ""),
        "secs": round(secs, 1),
    }


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print("-" * 108)
    hdr = (f"{'method':<16}{'run':<5}{'test F':<16}{'cascade':<10}"
           f"{'false':<8}{'latency':<10}{'gens->tgt':<10}{'secs':<6}")
    print(hdr)
    print("-" * 108)
    for r in rows:
        fstr = f"{r['test_F_mean']:.2f} +/- {r['test_F_std']:.2f}"
        print(f"{r['method']:<16}{r['run']:<5}{fstr:<16}{r['cascade_size']:<10.1f}"
              f"{r['false_trips']:<8.1f}{r['latency_penalty']:<10.1f}"
              f"{str(r['gens_to_target']):<10}{r['secs']:<6.1f}")
    print("-" * 108)


def run_comparison(args) -> list[dict]:
    G = graph_gen.generate_service_graph(n_nodes=40, seed=7)
    n_edges = G.number_of_edges()
    log.info("graph: %d nodes, %d edges", G.number_of_nodes(), n_edges)
    train_seeds = make_seeds(args.train_seeds, base=1000)
    test_seeds = make_seeds(args.test_seeds, base=2000)
    assert not set(train_seeds) & set(test_seeds), "train/test seeds must be disjoint"

    cfg_v = gamod.GAConfig(pop_size=args.pop, generations=args.gens, mutation_mode="uniform")
    cfg_g = gamod.GAConfig(pop_size=args.pop, generations=args.gens, mutation_mode="guided")
    budget = args.pop * (args.gens + 1)  # == GA evals: initial pop + one per gen

    # fixed baseline; the target is beating it by 15% on train (demanding,
    # since the informed init already starts near the baseline)
    t0 = time.time()
    fixed = blmod.fixed_threshold(G, 0.5, train_seeds, test_seeds, **ROUND1_KW)
    target = 0.85 * fixed["train_F"]
    rows = [summarise(fixed["method"], 0, fixed, fixed["train_F"], time.time() - t0)]
    log.info("fixed-0.5 train F = %.3f -> target %.3f", fixed["train_F"], target)

    histories = {"vanilla GA": [], "guided GA": []}
    for run in range(args.runs):
        run_rng = make_rng(args.seed + 100 + run)

        t0 = time.time()
        rs = blmod.random_search(G, train_seeds, test_seeds, budget, run_rng, **ROUND1_KW)
        rows.append(summarise("random-search", run, rs, rs["train_F"], time.time() - t0))

        for name, cfg in (("vanilla GA", cfg_v), ("guided GA", cfg_g)):
            t0 = time.time()
            ga_rng = make_rng(args.seed + 1000 + run * 10 + (0 if name == "vanilla GA" else 1))
            res = gamod.run_ga(G, train_seeds, cfg, ga_rng,
                               target_score=target, **ROUND1_KW)
            test = fitmod.evaluate(G, res["best"], test_seeds, **ROUND1_KW)
            res["test"] = test
            rows.append(summarise(name, run, res, res["best_F"], time.time() - t0))
            histories[name].append(res["history"])
            log.info("%s run %d: train F=%.3f test F=%.3f", name, run, res["best_F"], test["F"])

    print_table(rows, "ROUND 1 RESULTS (test seeds, lower F is better)")

    outdir = ensure_dir(args.outdir)
    write_csv(outdir / "round1_results.csv", rows, list(rows[0].keys()))
    mean_hist = {
        name: list(np.mean(h, axis=0)) for name, h in histories.items() if h
    }
    if mean_hist:
        save_convergence_svg(outdir / "convergence.svg", mean_hist)
        log.info("wrote %s", outdir / "convergence.svg")
    log.info("wrote %s", outdir / "round1_results.csv")
    return rows


def run_round2(args) -> list[dict]:
    """Shift the environment, then compare warm-start vs scratch recovery.

    The honest metric for adaptation is *speed*: generations needed to
    recover past the stale (non-adapted) Round 1 thresholds, measured on
    Round 2 test seeds. Warm-start = Round 1 final population + diversity
    noise + temporary mutation boost; scratch = fresh population.
    """
    G = graph_gen.generate_service_graph(n_nodes=40, seed=7)
    train_seeds = make_seeds(args.train_seeds, base=1000)
    test_seeds = make_seeds(args.test_seeds, base=2000)
    assert not set(train_seeds) & set(test_seeds)

    adapt_gens = max(10, args.gens // 2)
    n_reps = 2
    rows: list[dict] = []
    curves: dict[str, list[list[float]]] = {"warm-start": [], "from-scratch": []}
    stale_fs: list[float] = []

    for rep in range(n_reps):
        log.info("Round 2 rep %d: training guided GA on Round 1 env ...", rep)
        r1 = gamod.run_ga(
            G, train_seeds,
            gamod.GAConfig(pop_size=args.pop, generations=args.gens,
                           mutation_mode="guided"),
            make_rng(args.seed + rep), **ROUND1_KW,
        )
        stale = fitmod.evaluate(G, r1["best"], test_seeds, **ROUND2_KW)
        stale_fs.append(stale["F"])
        log.info("rep %d: Round 1 train F=%.3f; stale in Round 2: test F=%.3f",
                 rep, r1["best_F"], stale["F"])
        log.info("rep %d: shift spread_p %.2f -> %.2f, cooldown %d -> %d", rep,
                 ROUND1_KW["spread_p"], ROUND2_KW["spread_p"],
                 ROUND1_KW["cooldown"], ROUND2_KW["cooldown"])

        adapt_cfg = gamod.GAConfig(pop_size=args.pop, generations=adapt_gens,
                                   mutation_mode="guided")
        # warm start: previous population + diversity injection + boost
        wrng = make_rng(args.seed + 100 + rep)
        init = np.clip(
            r1["final_pop"] + wrng.normal(0.0, 0.08, r1["final_pop"].shape),
            0.0, 1.0,
        )
        warm = gamod.run_ga(G, train_seeds, adapt_cfg, wrng, init_pop=init,
                            mutation_boost=3.0, boost_gens=10, **ROUND2_KW)
        scratch = gamod.run_ga(G, train_seeds, adapt_cfg,
                               make_rng(args.seed + 200 + rep), **ROUND2_KW)

        for name, res in (("warm-start", warm), ("from-scratch", scratch)):
            test_curve = [fitmod.evaluate(G, ind, test_seeds, **ROUND2_KW)["F"]
                          for ind in res["best_per_gen"]]
            curves[name].append(test_curve)
            rec = next((g for g, f in enumerate(test_curve) if f <= stale["F"]),
                       None)
            rows.append({
                "method": name, "rep": rep,
                "final_test_F": round(test_curve[-1], 3),
                "stale_test_F": round(stale["F"], 3),
                "gens_to_recover": rec if rec is not None else "-",
                "evals": res["evals"],
            })
            log.info("rep %d %s: final test F=%.3f, gens_to_recover=%s",
                     rep, name, test_curve[-1], rec)

    mean_curves = {k: list(np.mean(v, axis=0)) for k, v in curves.items()}
    mean_stale = float(np.mean(stale_fs))

    print("\nROUND 2 RECOVERY (shifted env; lower test F is better)")
    print("-" * 70)
    print(f"{'method':<14}{'rep':<5}{'final test F':<14}{'stale F':<10}{'gens_to_recover'}")
    print("-" * 70)
    for r in rows:
        print(f"{r['method']:<14}{r['rep']:<5}{r['final_test_F']:<14.2f}"
              f"{r['stale_test_F']:<10.2f}{r['gens_to_recover']}")
    print("-" * 70)
    print(f"stale (no adaptation) mean test F: {mean_stale:.2f}")

    outdir = ensure_dir(args.outdir)
    write_csv(outdir / "round2_results.csv", rows, list(rows[0].keys()))
    curve_rows = [
        {"gen": g, "warm_start": round(w, 3), "from_scratch": round(s, 3)}
        for g, (w, s) in enumerate(zip(mean_curves["warm-start"],
                                       mean_curves["from-scratch"]))
    ]
    write_csv(outdir / "round2_curves.csv", curve_rows,
              ["gen", "warm_start", "from_scratch"])
    save_convergence_svg(
        outdir / "recovery.svg",
        {"warm-start": mean_curves["warm-start"],
         "from-scratch": mean_curves["from-scratch"]},
        title="Round 2 recovery (test F per adaptation generation)",
        xlabel="adaptation generation",
        ylabel="test F (lower is better)",
    )
    log.info("wrote %s, %s, %s", outdir / "round2_results.csv",
             outdir / "round2_curves.csv", outdir / "recovery.svg")
    return rows


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:  # tiny budgets for smoke tests
        args.pop, args.gens = 8, 6
        args.train_seeds, args.test_seeds, args.runs = 4, 4, 2
    if args.round2:
        run_round2(args)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
