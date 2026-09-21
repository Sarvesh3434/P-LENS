"""Run simulations and experiments.

`run_simulation` runs one seed/policy. `run_experiment` runs P-LENS and all
baselines over multiple seeds with common random numbers (same seed ⇒ same
initial topology, traffic, failure draws — [ASSUMPTION] fairness per prompt §10),
reports mean ± 95% CI and paired Wilcoxon signed-rank tests vs P-LENS, and saves
per-run JSON files. Per-step topology snapshots are kept in every run so the
evolution can be replayed and visualized.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from experiments.config import Config
from experiments.seeds import demo_seeds
from network.graph import EvolvingNetwork
from network.metrics import snapshot
from plens.controller import PLENSController


def run_simulation(
    cfg: Config,
    demand_series: np.ndarray,
    scaled_feat: np.ndarray | None = None,
    predictor=None,
    collect_topology: bool = True,
) -> dict:
    rng = np.random.default_rng(cfg.seed)
    net = EvolvingNetwork(cfg, rng)
    base_rng = np.random.default_rng(cfg.seed + 1_000_003)  # dedicated baseline stream [SIM]
    ctl = PLENSController(cfg, net, ml_predictor=predictor, base_rng=base_rng)
    T = min(cfg.simulation_steps, len(demand_series))
    L = cfg.window_length_L
    for t in range(T):
        hist = None
        if scaled_feat is not None and t >= L:
            hist = scaled_feat[t - L : t]
        ctl.step(t, demand_series[t], hist)
    out = {
        "seed": cfg.seed,
        "policy": cfg.policy,
        "ablation": cfg.ablation,
        "dataset": cfg.dataset,
        "history": ctl.history,
        "rejection_log": ctl.rejection_log,
        "final_edges": list(map(list, net.G.edges())),
        "initial_edges": list(map(list, net.reference.edges())),
        "n_nodes": cfg.network_size,
    }
    if collect_topology:
        out["topology_snapshots"] = [list(map(list, s)) for s in ctl.topology_snapshots]
    return out


def summarize(run: dict) -> dict:
    h = run["history"]

    def mean(key):
        return float(np.nanmean([x["metrics"].get(key, np.nan) for x in h]))

    n_steps = max(len(h), 1)
    n_mut_steps = sum(1 for x in h if x["mutations"])
    return {
        "seed": run["seed"],
        "policy": run["policy"],
        "ablation": run.get("ablation", "none"),
        "latency_mean": mean("latency"),
        "throughput_mean": mean("throughput"),
        "loss_mean": mean("loss"),
        "congestion_mean": mean("congestion"),
        "connected_frac": mean("connected"),
        "avg_path_length_mean": mean("avg_path_length"),
        "mutations": int(sum(len(x["mutations"]) for x in h)),
        "add": int(sum(x["n_add"] for x in h)),
        "prune": int(sum(x["n_prune"] for x in h)),
        "rewire": int(sum(x["n_rewire"] for x in h)),
        "mutation_rate": float(n_mut_steps / n_steps),
        "pp_mean": float(np.mean([x["pp_mean"] for x in h])),
        "pd_mean": float(np.mean([x["pd_mean"] for x in h])),
        "budget_mean": float(np.mean([x["budget"] for x in h])),
        "mc_mean": float(np.mean([x["mc_mean"] for x in h])),
    }


def _policy_run(cfg: Config, demand, scaled_feat, predictor):
    return run_simulation(cfg, demand, scaled_feat, predictor)


def run_experiment(
    cfg: Config,
    demand_series: np.ndarray,
    scaled_feat: np.ndarray | None = None,
    predictor=None,
    policies=("plens", "static", "random", "greedy"),
    seeds=None,
    out_dir: str | Path | None = None,
    save_runs: bool = True,
    verbose: bool = True,
    return_runs: bool = False,
) -> dict:
    """Run all policies × seeds. Same seed = same traffic/failure draws [CRN].

    return_runs=True additionally returns the raw per-policy run dicts
    (history, topology snapshots) so figures can be drawn without re-reading
    the saved per-run JSONs [ASSUMPTION: convenience only].
    """
    seeds = seeds if seeds is not None else demo_seeds(cfg.n_seeds)
    out_dir = Path(out_dir) if out_dir else cfg.artifacts_dir() / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_policy = {p: [] for p in policies}
    raw_runs: dict = {p: [] for p in policies}
    for seed in seeds:
        for policy in policies:
            run_cfg = Config(**{**cfg.__dict__, "seed": seed, "policy": policy})
            t0 = time.time()
            run = _policy_run(run_cfg, demand_series, scaled_feat, predictor)
            dt = time.time() - t0
            s = summarize(run)
            per_policy[policy].append(s)
            if return_runs:
                raw_runs[policy].append(run)
            if save_runs:
                with open(out_dir / f"run_{policy}_seed{seed}.json", "w") as f:
                    json.dump(run, f)
            if verbose:
                print(
                    f"  seed={seed} policy={policy:7s} "
                    f"lat={s['latency_mean']:.5f} loss={s['loss_mean']:.4f} "
                    f"thr={s['throughput_mean']:.4f} mut={s['mutations']:4d} ({dt:.1f}s)"
                )
    # aggregate stats
    agg = {}
    for p, rows in per_policy.items():
        agg[p] = {}
        for key in rows[0]:
            if key in ("seed", "policy", "ablation"):
                continue
            vals = np.array([r[key] for r in rows], dtype=float)
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci = 1.96 * sd / np.sqrt(max(len(vals), 1))
            agg[p][key] = {"mean": mu, "ci95": float(ci), "values": vals.tolist()}
    # paired Wilcoxon signed-rank vs P-LENS on per-seed means [ASSUMPTION §10]
    tests = {}
    plens_rows = per_policy.get("plens", [])
    for p in policies:
        if p == "plens":
            continue
        tests[p] = {}
        for key in ("latency_mean", "throughput_mean", "loss_mean", "congestion_mean"):
            a = np.array([r[key] for r in plens_rows], dtype=float)
            b = np.array([r[key] for r in per_policy[p]], dtype=float)
            d = a - b
            if np.allclose(d, 0.0):
                tests[p][key] = {"stat": None, "p": 1.0, "note": "no differences"}
                continue
            try:
                stat, pv = wilcoxon(a, b)
                tests[p][key] = {"stat": float(stat), "p": float(pv)}
            except Exception as e:  # tiny samples etc.
                tests[p][key] = {"stat": None, "p": None, "note": str(e)}
    result = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
        "seeds": list(seeds),
        "per_seed": per_policy,
        "aggregate": agg,
        "wilcoxon_vs_plens": tests,
    }
    if return_runs:
        result["runs"] = raw_runs
    with open(out_dir / "experiment_summary.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result
