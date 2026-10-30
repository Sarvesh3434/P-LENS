"""[SIM] Multi-scale / multi-topology / traffic / failure scenario study.

Each cell is a full paired experiment: identical seeds, traffic matrices and
failure streams for P-LENS and every baseline, with the statistics from
:mod:`experiments.stats`.  Results are written per cell and aggregated into
``study_summary.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from experiments.config import Config
from experiments.runner import run_experiment


def _demand_for(cfg: Config, bundle) -> tuple[np.ndarray, np.ndarray | None]:
    steps = cfg.simulation_steps
    if bundle is None:
        from ml.synthetic import synthetic_windows

        _, dem, _, _ = synthetic_windows(cfg, n_windows=steps, seed=cfg.seed + 5_000)
        return dem, None
    from experiments.traffic import replay_series

    return replay_series(bundle, steps)


def run_study(
    base_cfg: Config,
    bundle=None,
    predictor=None,
    scales: Sequence[int] = (24, 50, 100, 200),
    topologies: Sequence[str] = ("ring", "erdos_renyi", "barabasi_albert", "small_world"),
    patterns: Sequence[str] = ("stationary", "shift", "periodic", "bursty"),
    failures: Sequence[str] = ("random", "targeted", "multi", "recovery"),
    seeds: Iterable[int] | None = None,
    policies: Sequence[str] = ("plens", "static", "random", "greedy", "threshold", "fixed_budget", "adaptive_budget", "optimization", "bandit"),
    out_dir: str | Path | None = None,
    save_runs: bool = False,
    verbose: bool = True,
    n_jobs: int = 1,
) -> dict:
    out = Path(out_dir) if out_dir else base_cfg.artifacts_dir() / "study"
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds) if seeds is not None else list(range(7, 7 + max(base_cfg.n_seeds, 30)))
    summary: dict = {"seeds": seeds, "cells": {}}

    for scale in scales:
        for topo in topologies:
            for pattern in patterns:
                for failure in failures:
                    key = f"n{scale}_{topo}_{pattern}_{failure}"
                    cfg = Config(
                        **{
                            **base_cfg.__dict__,
                            "network_size": int(scale),
                            "initial_topology": topo,
                            "traffic_pattern": pattern,
                            "failure_mode": failure,
                        }
                    )
                    dem, scaled_feat = _demand_for(cfg, bundle)
                    cell_dir = out / key
                    if verbose:
                        print(f"[study] {key} ({len(seeds)} seeds x {len(policies)} policies)")
                    res = run_experiment(
                        cfg,
                        dem,
                        scaled_feat,
                        predictor,
                        policies=policies,
                        seeds=seeds,
                        out_dir=cell_dir,
                        save_runs=save_runs,
                        verbose=verbose,
                        n_jobs=n_jobs,
                    )
                    summary["cells"][key] = {
                        "config": {
                            "network_size": scale,
                            "initial_topology": topo,
                            "traffic_pattern": pattern,
                            "failure_mode": failure,
                        },
                        "aggregate": res["aggregate"],
                        "statistics_vs_plens": res.get("statistics_vs_plens", {}),
                    }
                    with open(out / "study_summary.json", "w") as f:
                        json.dump(summary, f, indent=2, default=str)
    return summary
