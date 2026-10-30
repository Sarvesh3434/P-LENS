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
from experiments.stats import holm_family, paired_tests
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
    from experiments.traffic import intensity_series

    intensities = intensity_series(cfg, T)
    for t in range(T):
        hist = None
        if scaled_feat is not None and t >= L:
            hist = scaled_feat[t - L : t]
        ctl.step(t, demand_series[t], hist, float(intensities[t]))
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
    n_success = int(sum(x.get("n_success", 0) for x in h))
    n_fail = int(sum(x.get("n_fail", 0) for x in h))
    evaluated = n_success + n_fail
    return {
        "seed": run["seed"],
        "policy": run["policy"],
        "ablation": run.get("ablation", "none"),
        "latency_mean": mean("latency"),
        "latency_p95_mean": mean("latency_p95"),
        "latency_p99_mean": mean("latency_p99"),
        "throughput_mean": mean("throughput"),
        "loss_mean": mean("loss"),
        "congestion_mean": mean("congestion"),
        "sla_violation_mean": mean("sla_violation"),
        "sla_latency_violation_mean": mean("sla_latency_violation"),
        "sla_loss_violation_mean": mean("sla_loss_violation"),
        "connected_frac": mean("connected"),
        "avg_path_length_mean": mean("avg_path_length"),
        "mutations": int(sum(len(x["mutations"]) for x in h)),
        "add": int(sum(x["n_add"] for x in h)),
        "prune": int(sum(x["n_prune"] for x in h)),
        "rewire": int(sum(x["n_rewire"] for x in h)),
        "mutation_rate": float(n_mut_steps / n_steps),
        "n_success": n_success,
        "n_fail": n_fail,
        "success_rate": float(n_success / evaluated) if evaluated else 0.0,
        "pp_mean": float(np.mean([x["pp_mean"] for x in h])),
        "pd_mean": float(np.mean([x["pd_mean"] for x in h])),
        "debt_persistence_mean": float(np.mean([x.get("debt_persistence_mean", 0.0) for x in h])),
        "criticality_mean": float(np.mean([x.get("criticality_mean", 0.0) for x in h])),
        "budget_mean": float(np.mean([x["budget"] for x in h])),
        "budget_spent_mean": float(np.mean([x.get("budget_spent", 0.0) for x in h])),
        "overload_mean": float(np.mean([x.get("overload", 0.0) for x in h])),
        "eval_uncertainty_mean": float(np.mean([x.get("eval_uncertainty", 0.0) for x in h])),
        "mc_mean": float(np.mean([x["mc_mean"] for x in h])),
        "control_ms_mean": 1000.0 * float(np.mean([x.get("control_seconds", 0.0) for x in h])),
        "eval_ms_mean": 1000.0 * float(np.mean([x.get("eval_seconds", 0.0) for x in h])),
        "step_ms_mean": 1000.0 * float(np.mean([x.get("step_seconds", 0.0) for x in h])),
    }


def _policy_run(cfg: Config, demand, scaled_feat, predictor):
    return run_simulation(cfg, demand, scaled_feat, predictor)


def _run_one(payload: dict) -> tuple:
    """Worker for the process pool: one (seed, policy) run.

    Runs are deterministic in ``cfg.seed`` (common random numbers), so the
    scheduling of workers cannot change any result — only the wall-clock time.
    """
    run_cfg = Config(**{**payload["cfg"], "seed": payload["seed"], "policy": payload["policy"]})
    run = run_simulation(
        run_cfg,
        payload["demand"],
        payload["scaled_feat"],
        payload["predictor"],
        collect_topology=payload["collect_topology"],
    )
    s = summarize(run)
    if payload["save_runs"]:
        with open(Path(payload["out_dir"]) / f"run_{payload['policy']}_seed{payload['seed']}.json", "w") as f:
            json.dump(run, f)
    return s, (run if payload["return_run"] else None)


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
    n_jobs: int = 1,
    collect_topology: bool = True,
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

    # Plan every (seed, policy) run, then execute them in a process pool when
    # n_jobs > 1.  Results are identical either way (CRN + seeded RNG).
    plan = [
        {
            "cfg": cfg.__dict__,
            "demand": demand_series,
            "scaled_feat": scaled_feat,
            "predictor": predictor,
            "policy": policy,
            "seed": seed,
            "out_dir": str(out_dir),
            "save_runs": save_runs,
            "return_run": return_runs,
            "collect_topology": collect_topology,
        }
        for seed in seeds
        for policy in policies
    ]

    t_start = time.time()
    if n_jobs > 1 and len(plan) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=int(n_jobs)) as pool:
            outcomes = list(pool.map(_run_one, plan, chunksize=1))
    else:
        outcomes = [_run_one(p) for p in plan]

    for payload, (s, run) in zip(plan, outcomes):
        per_policy[payload["policy"]].append(s)
        if return_runs:
            raw_runs[payload["policy"]].append(run)
        if verbose:
            print(
                f"  seed={payload['seed']} policy={payload['policy']:15s} "
                f"lat={s['latency_mean']:.5f} loss={s['loss_mean']:.4f} "
                f"thr={s['throughput_mean']:.4f} mut={s['mutations']:4d}"
            )
    if verbose:
        print(f"  {len(plan)} runs in {time.time() - t_start:.1f}s (n_jobs={n_jobs})")
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
    # Paired tests with Holm correction and effect sizes vs P-LENS [§10]
    tests: dict = {}
    plens_rows = per_policy.get("plens", [])
    metric_keys = [k for k in per_policy[policies[0]][0] if k not in ("seed", "policy", "ablation")]
    for p in policies:
        if p == "plens":
            continue
        tests[p] = {}
        for key in metric_keys:
            a = np.array([r[key] for r in plens_rows], dtype=float)
            b = np.array([r[key] for r in per_policy[p]], dtype=float)
            if np.allclose(a, b):
                tests[p][key] = {
                    "metric": key,
                    "n": len(a),
                    "mean_diff": 0.0,
                    "ttest_p": 1.0,
                    "wilcoxon_p": 1.0,
                    "note": "no differences",
                }
                continue
            tests[p][key] = paired_tests(a, b, key)
        holm = holm_family(tests[p], key="wilcoxon_p")
        for key, adj in holm.items():
            tests[p][key]["p_holm"] = adj
            tests[p][key]["significant_holm"] = bool(adj < 0.05)
    result = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
        "seeds": list(seeds),
        "per_seed": per_policy,
        "aggregate": agg,
        "wilcoxon_vs_plens": tests,
        "statistics_vs_plens": tests,
    }
    if return_runs:
        result["runs"] = raw_runs
    with open(out_dir / "experiment_summary.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result

