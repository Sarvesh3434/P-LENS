"""[DIAGRAM]+[ASSUMPTION] Ablation study: P-LENS with one component disabled.

Runs the same seeds/traffic/failures as the main experiment (common random
numbers) with `ablation` set to one of:
    no_ml     – PP/PD/B/MF computed from observed indicators only
    no_debt   – PD forced to 0 (debt term drops out of MF, refill loses mean(PD))
    no_credit – MC stays empty (credit term drops out of MF)
    no_budget – budget pinned to B_max (no throttle; still capped per step)
Each ablation runs the full P-LENS machinery — none is replaced by another
policy — so the comparison isolates exactly one mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.config import Config
from experiments.runner import run_experiment


def run_ablations(
    cfg: Config,
    demand_series: np.ndarray,
    scaled_feat: np.ndarray | None = None,
    predictor=None,
    seeds=None,
    out_dir: str | Path | None = None,
    save_runs: bool = False,
    verbose: bool = True,
) -> dict:
    seeds = seeds if seeds is not None else list(range(7, 7 + cfg.n_seeds))
    out_dir = Path(out_dir) if out_dir else cfg.artifacts_dir() / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    ablations = ["none", "no_ml", "no_debt", "no_credit", "no_budget"]
    all_rows: dict = {}
    for abl in ablations:
        if verbose:
            print(f"[ablation] {abl}")
        run_cfg = Config(**{**cfg.__dict__, "ablation": abl, "policy": "plens"})
        res = run_experiment(
            run_cfg,
            demand_series,
            scaled_feat,
            predictor,
            policies=("plens",),
            seeds=seeds,
            out_dir=out_dir / abl,
            save_runs=save_runs,
            verbose=verbose,
        )
        all_rows[abl] = res["aggregate"]["plens"]
    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    return all_rows
